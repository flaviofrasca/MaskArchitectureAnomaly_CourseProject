"""
Step 8 — Mask-based Anomaly Baselines with EoMT.

Evaluates MSP, MaxLogit, MaxEntropy and RbA on 5 anomaly datasets
using an EoMT checkpoint (Cityscapes, COCO, or fine-tuned).

Run from eomt/ directory:
    python eval_anomaly_eomt.py --model_type cityscapes --method MSP
    python eval_anomaly_eomt.py --model_type coco       --method RbA
    python eval_anomaly_eomt.py --model_type finetuned  --method MaxLogit
"""

import argparse
import glob
import importlib
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import average_precision_score
from ood_metrics import fpr_at_95_tpr
from torch.amp.autocast_mode import autocast
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = "/content/drive/.shortcut-targets-by-id/1osgiWms0a4SYz1evCZNwMV-jv--0I0RU/MaskArch_Shared"

CHECKPOINTS = {
    "cityscapes": BASE + "/checkpoints/cityscapes/eomt_cityscapes.bin",
    "coco":       BASE + "/checkpoints/coco/eomt_coco.bin",
    "finetuned":  BASE + "/checkpoints/finetuned/eomt_finetuned.bin",
}

CONFIGS = {
    "cityscapes": "configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
    "coco":       "configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml",
    "finetuned":  "configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
}

NUM_CLASSES = {
    "cityscapes": 19,
    "coco":       133,
    "finetuned":  19,
}

IMG_SIZE = (640, 640)

ANOMALY_DIR = BASE + "/datasets/anomaly/"

DATASETS = [
    ("SMIYC RA-21",  "RoadAnomaly21/images/*.png"),
    ("SMIYC RO-21",  "RoadObsticle21/images/*.webp"),
    ("FS L&F",       "LostAndFound/images/*.png"),
    ("FS Static",    "fs_static/images/*.jpg"),
    ("Road Anomaly", "RoadAnomaly/images/*.jpg"),
]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_and_load_model(model_type: str, device):
    config_path = CONFIGS[model_type]
    ckpt_path   = CHECKPOINTS[model_type]
    num_classes = NUM_CLASSES[model_type]

    with open(config_path) as f:
        config = yaml.safe_load(f)

    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_mod, enc_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder = getattr(importlib.import_module(enc_mod), enc_name)(
        img_size=IMG_SIZE, **encoder_cfg.get("init_args", {})
    )

    net_cfg = config["model"]["init_args"]["network"]
    net_mod, net_name = net_cfg["class_path"].rsplit(".", 1)
    net_kw = {k: v for k, v in net_cfg["init_args"].items() if k != "encoder"}
    network = getattr(importlib.import_module(net_mod), net_name)(
        masked_attn_enabled=False, num_classes=num_classes, encoder=encoder, **net_kw
    )

    lit_mod, lit_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_mod), lit_name)
    model_kw = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    if "stuff_classes" in config.get("data", {}).get("init_args", {}):
        model_kw["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

    model = lit_cls(img_size=IMG_SIZE, num_classes=num_classes, network=network, **model_kw)

    if os.path.isfile(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights: {ckpt_path}")
    else:
        print(f"[WARNING] Checkpoint not found: {ckpt_path} — using random weights")

    return model.eval().to(device)


# ---------------------------------------------------------------------------
# Anomaly scoring
# ---------------------------------------------------------------------------

def compute_anomaly_score(pixel_logits, mask_logits, class_logits, method):
    """
    Args:
        pixel_logits : [C, H, W]  — combined per-pixel semantic logits
        mask_logits  : [B, Q, H, W] — raw mask logits (needed for RbA)
        class_logits : [B, Q, C+1] — raw class logits (last = background)
        method       : 'MSP' | 'MaxLogit' | 'MaxEntropy' | 'RbA'
    Returns:
        anomaly_score : np.ndarray [H, W], higher = more anomalous
    """
    if method == 'MaxLogit':
        return -pixel_logits.max(dim=0)[0].cpu().numpy()

    elif method == 'MSP':
        probs = pixel_logits.softmax(dim=0)
        return (1.0 - probs.max(dim=0)[0]).cpu().numpy()

    elif method == 'MaxEntropy':
        probs = pixel_logits.softmax(dim=0).clamp(min=1e-8)
        entropy = -(probs * probs.log()).sum(dim=0)
        return entropy.cpu().numpy()

    elif method == 'RbA':
        # inlier_score(p) = max_q( sigmoid(mask(p,q)) * (1 - P(background|q)) )
        background_prob  = class_logits.softmax(dim=-1)[..., -1]   # [B, Q]
        foreground_score = 1.0 - background_prob                   # [B, Q]
        mask_probs = mask_logits.sigmoid()                         # [B, Q, H, W]
        fs = foreground_score[:, :, None, None]                    # [B, Q, 1, 1]
        inlier_score = (mask_probs * fs).max(dim=1)[0]             # [B, H, W]
        return (1.0 - inlier_score)[0].cpu().numpy()

    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Per-image inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_image(model, img_tensor, method, device):
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    imgs      = [img_tensor.to(device)]
    img_sizes = [img_tensor.shape[-2:]]

    with autocast(dtype=torch.float16, device_type=device_type):
        crops, origins = model.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = model(crops)

        mask_logits  = F.interpolate(
            mask_logits_per_layer[-1], model.img_size, mode="bilinear"
        )
        class_logits = class_logits_per_layer[-1]

        if method == 'RbA':
            background_prob  = class_logits.softmax(dim=-1)[..., -1]
            foreground_score = 1.0 - background_prob
            mask_probs       = mask_logits.sigmoid()
            fs               = foreground_score[:, :, None, None]
            inlier_crops     = (mask_probs * fs).max(dim=1)[0]
            anomaly_crops    = 1.0 - inlier_crops

            reverted = model.revert_window_logits_semantic(
                anomaly_crops.unsqueeze(1), origins, img_sizes
            )
            return reverted[0].squeeze(0).float().cpu().numpy()

        else:
            crop_logits  = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
            pixel_logits = model.revert_window_logits_semantic(
                crop_logits, origins, img_sizes
            )[0].float()

            return compute_anomaly_score(pixel_logits, mask_logits, class_logits, method)


# ---------------------------------------------------------------------------
# Dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(dataset_path, model, method, device):
    image_paths = sorted(glob.glob(os.path.expanduser(dataset_path)))
    if not image_paths:
        print(f"  [WARNING] No images found: {dataset_path}")
        return None

    anomaly_score_list, ood_gts_list = [], []

    for path in tqdm(image_paths, desc="  images", leave=False):
        img_np     = np.array(Image.open(path).convert('RGB'))
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        anomaly_result = infer_image(model, img_tensor, method, device)

        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace(".webp", ".png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace(".jpg", ".png")
        if "RoadAnomaly" in pathGT and "RoadAnomaly21" not in pathGT:
            pathGT = pathGT.replace(".jpg", ".png")

        if not os.path.exists(pathGT):
            continue

        ood_gts = np.array(Image.open(pathGT).convert('L'))

        if ood_gts.shape != anomaly_result.shape:
            ood_gts = np.array(
                Image.fromarray(ood_gts).resize(
                    (anomaly_result.shape[1], anomaly_result.shape[0]),
                    Image.NEAREST
                )
            )

        if "RoadAnomaly" in pathGT and "RoadAnomaly21" not in pathGT:
            ood_gts = np.where(ood_gts == 2, 1, ood_gts)
        if "Streethazard" in pathGT:
            ood_gts = np.where(ood_gts == 14, 255, ood_gts)
            ood_gts = np.where(ood_gts < 20,  0,   ood_gts)
            ood_gts = np.where(ood_gts == 255, 1,   ood_gts)

        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)
        anomaly_score_list.append(anomaly_result)

    if not ood_gts_list:
        print("  [WARNING] No valid images with anomaly labels found.")
        return None

    ood_gts_arr   = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    val_out   = np.concatenate((anomaly_scores[ood_gts_arr == 0], anomaly_scores[ood_gts_arr == 1]))
    val_label = np.concatenate((np.zeros((ood_gts_arr == 0).sum()), np.ones((ood_gts_arr == 1).sum())))

    auprc = average_precision_score(val_label, val_out) * 100.0
    fpr95 = fpr_at_95_tpr(val_out, val_label) * 100.0

    return auprc, fpr95


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='cityscapes',
                        choices=['cityscapes', 'coco', 'finetuned'])
    parser.add_argument('--method', default='MSP',
                        choices=['MSP', 'MaxLogit', 'MaxEntropy', 'RbA'])
    parser.add_argument('--checkpoint', default=None,
                        help='Override checkpoint path (optional)')
    parser.add_argument('--datadir', default=ANOMALY_DIR)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu else 'cuda')

    if args.checkpoint:
        CHECKPOINTS[args.model_type] = args.checkpoint

    print(f"\nLoading EoMT [{args.model_type}] ...")
    model = build_and_load_model(args.model_type, device)

    results_file = open('results_eomt.txt', 'a')
    results_file.write(f"\n=== EoMT [{args.model_type}] | Method: {args.method} ===\n")

    print(f"\n{'='*60}")
    print(f"Model: EoMT [{args.model_type}]  |  Method: {args.method}")
    print(f"{'='*60}")
    print(f"{'Dataset':<20} {'AuPRC':>8} {'FPR95':>8}")
    print(f"{'-'*40}")

    for name, pattern in DATASETS:
        full_pattern = os.path.join(args.datadir, pattern)
        print(f"{name:<20} ", end="", flush=True)
        result = evaluate_dataset(full_pattern, model, args.method, device)
        if result is not None:
            auprc, fpr95 = result
            print(f"{auprc:>7.2f}% {fpr95:>7.2f}%")
            results_file.write(f"  {name:<20} AuPRC: {auprc:.2f}%  FPR@95: {fpr95:.2f}%\n")
        else:
            print("  SKIPPED")
            results_file.write(f"  {name:<20} SKIPPED\n")

    print(f"{'='*60}\n")
    results_file.close()
    print("Results appended to results_eomt.txt")


if __name__ == '__main__':
    main()
