"""
Evaluate BOTH EoMT models on the Cityscapes val set using the same
semantic-inference pipeline, enabling fair mIoU comparison:

  - Cityscapes-trained model  (19 classes  → identity remap)
  - COCO-panoptic model       (133 classes → COCO→Cityscapes remap)

Key design choices
------------------
* Semantic inference (window_imgs_semantic + to_per_pixel_logits_semantic) is
  used for BOTH models — the only fair comparison: panoptic inference adds
  NMS / overlap thresholds that the Cityscapes model never uses.
* Classes with no COCO equivalent are mapped to 255 (ignore_index).
* 'rider' (train_id 12) will always score ~0 for the COCO model — COCO
  annotates riders as 'person'. Mention this limitation in the report.

Usage (run from eomt/ on Colab — no arguments needed if drive is mounted):
    python eval_coco_on_cityscapes.py

Or override any path:
    python eval_coco_on_cityscapes.py \\
        --cityscapes_path /path/to/cityscapes \\
        --city_ckpt /path/to/city.bin \\
        --coco_ckpt /path/to/coco.bin
"""

import argparse
import importlib
import os
import warnings

import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from torch.amp.autocast_mode import autocast
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Default paths (same Drive folder used in inference.ipynb)
# ---------------------------------------------------------------------------

BASE = "/content/drive/.shortcut-targets-by-id/1osgiWms0a4SYz1evCZNwMV-jv--0I0RU/MaskArch_Shared"

DEFAULT_CITYSCAPES_PATH = BASE + "/datasets/cityscapes"
DEFAULT_CITY_CKPT       = BASE + "/checkpoints/cityscapes/eomt_cityscapes.bin"
DEFAULT_COCO_CKPT       = BASE + "/checkpoints/coco/eomt_coco.bin"
DEFAULT_CITY_CONFIG     = "configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"
DEFAULT_COCO_CONFIG     = "configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml"

COCO_IMG_SIZE    = (640, 640)
COCO_NUM_CLASSES = 133

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IGNORE_INDEX = 255
NUM_CITYSCAPES_CLASSES = 19

CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]

# COCO contiguous index (0-132) → Cityscapes train_id (0-18).
# 255 = void / ignore (no matching Cityscapes class).
COCO_TO_CITYSCAPES = torch.full((133,), IGNORE_INDEX, dtype=torch.long)

# --- exact matches ---
COCO_TO_CITYSCAPES[0]   = 11  # person
COCO_TO_CITYSCAPES[1]   = 18  # bicycle
COCO_TO_CITYSCAPES[2]   = 13  # car
COCO_TO_CITYSCAPES[3]   = 17  # motorcycle
COCO_TO_CITYSCAPES[5]   = 15  # bus
COCO_TO_CITYSCAPES[6]   = 16  # train
COCO_TO_CITYSCAPES[7]   = 14  # truck
COCO_TO_CITYSCAPES[9]   = 6   # traffic light
COCO_TO_CITYSCAPES[99]  = 0   # road (stuff)
COCO_TO_CITYSCAPES[118] = 4   # fence-merged
COCO_TO_CITYSCAPES[130] = 2   # building-other-merged

# --- synonyms / groupings ---
COCO_TO_CITYSCAPES[11]  = 7   # stop sign       → traffic sign
COCO_TO_CITYSCAPES[12]  = 5   # parking meter   → pole
COCO_TO_CITYSCAPES[82]  = 2   # bridge          → building
COCO_TO_CITYSCAPES[86]  = 2   # door-stuff      → building
COCO_TO_CITYSCAPES[88]  = 8   # flower          → vegetation
COCO_TO_CITYSCAPES[90]  = 9   # gravel          → terrain
COCO_TO_CITYSCAPES[91]  = 2   # house           → building
COCO_TO_CITYSCAPES[92]  = 5   # light           → pole
COCO_TO_CITYSCAPES[96]  = 1   # platform        → sidewalk
COCO_TO_CITYSCAPES[97]  = 9   # playingfield    → terrain
COCO_TO_CITYSCAPES[100] = 2   # roof            → building
COCO_TO_CITYSCAPES[101] = 9   # sand            → terrain
COCO_TO_CITYSCAPES[104] = 9   # snow            → terrain
COCO_TO_CITYSCAPES[105] = 1   # stairs          → sidewalk
COCO_TO_CITYSCAPES[108] = 3   # wall-brick      → wall
COCO_TO_CITYSCAPES[109] = 3   # wall-concrete   → wall
COCO_TO_CITYSCAPES[110] = 3   # wall-panel      → wall
COCO_TO_CITYSCAPES[111] = 3   # wall-stone      → wall
COCO_TO_CITYSCAPES[112] = 3   # wall-tile       → wall
COCO_TO_CITYSCAPES[113] = 3   # wall-wood       → wall
COCO_TO_CITYSCAPES[115] = 2   # window-blind    → building
COCO_TO_CITYSCAPES[116] = 2   # window-other    → building
COCO_TO_CITYSCAPES[117] = 8   # tree-merged     → vegetation
COCO_TO_CITYSCAPES[120] = 10  # sky-other-merged → sky
COCO_TO_CITYSCAPES[124] = 1   # pavement-merged → sidewalk
COCO_TO_CITYSCAPES[126] = 9   # grass-merged    → terrain
COCO_TO_CITYSCAPES[127] = 9   # dirt-merged     → terrain
COCO_TO_CITYSCAPES[131] = 9   # rock-merged     → terrain
COCO_TO_CITYSCAPES[132] = 3   # wall-other-merged → wall

# --- additional stuff mappings ---
COCO_TO_CITYSCAPES[87]  = 1   # floor-wood         → sidewalk
COCO_TO_CITYSCAPES[98]  = 0   # railroad            → road
COCO_TO_CITYSCAPES[102] = 9   # sea                 → terrain
COCO_TO_CITYSCAPES[114] = 9   # water-other         → terrain
COCO_TO_CITYSCAPES[119] = 2   # ceiling-merged      → building
COCO_TO_CITYSCAPES[121] = 2   # cabinet-merged      → building
COCO_TO_CITYSCAPES[123] = 1   # floor-other-merged  → sidewalk
COCO_TO_CITYSCAPES[125] = 9   # mountain-merged     → terrain

# Identity remap for the Cityscapes model (already predicts 19 classes).
CITY_TO_CITYSCAPES = torch.arange(NUM_CITYSCAPES_CLASSES, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model building and weight loading  (mirrors inference.ipynb)
# ---------------------------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
)


def build_model(config: dict, num_classes: int, img_size: tuple, device):
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_mod, enc_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder = getattr(importlib.import_module(enc_mod), enc_name)(
        img_size=img_size, **encoder_cfg.get("init_args", {})
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
    if "stuff_classes" in config["data"].get("init_args", {}):
        model_kw["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

    model = lit_cls(img_size=img_size, num_classes=num_classes, network=network, **model_kw)
    return model.eval().to(device), lit_cls, model_kw


def load_weights(model, config, lit_cls, model_kw, num_classes, img_size, device,
                 local_ckpt=None):
    """Local checkpoint takes priority; falls back to HuggingFace Hub."""
    if local_ckpt and os.path.isfile(local_ckpt):
        state_dict = torch.load(local_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded local weights: {local_ckpt}")
        return model

    name = config.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name")
    if name is None:
        warnings.warn("No logger name in config; skipping weight download.")
        return model
    try:
        ckpt_path = hf_hub_download(repo_id=f"tue-mps/{name}", filename="pytorch_model.bin")
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded HuggingFace weights: tue-mps/{name}")
    except RepositoryNotFoundError:
        warnings.warn(f"HF repo not found for {name}. Using random weights.")
    return model


# ---------------------------------------------------------------------------
# Evaluation loop  (shared by both models)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_dataset, device, remap: torch.Tensor):
    """
    Runs semantic inference on val_dataset and computes per-class IoU.
    `remap` converts model predictions to Cityscapes train_ids (0-18 or 255).
    """
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    remap = remap.to(device)
    metric = MulticlassJaccardIndex(
        num_classes=NUM_CITYSCAPES_CLASSES,
        validate_args=False,
        ignore_index=IGNORE_INDEX,
        average=None,
    ).to(device)

    for idx in tqdm(range(len(val_dataset)), desc="Evaluating", unit="img"):
        img, target = val_dataset[idx]
        img = img.to(device)

        with autocast(dtype=torch.float16, device_type=device_type):
            imgs = [img]
            img_sizes = [img.shape[-2:]]
            crops, origins = model.window_imgs_semantic(imgs)
            mask_logits_per_layer, class_logits_per_layer = model(crops)
            mask_logits = F.interpolate(
                mask_logits_per_layer[-1], model.img_size, mode="bilinear"
            )
            crop_logits = model.to_per_pixel_logits_semantic(
                mask_logits, class_logits_per_layer[-1]
            )
            logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)

        pred = logits[0].argmax(0)       # (H, W)
        pred_city = remap[pred]          # (H, W), values 0-18 or 255

        gt = model.to_per_pixel_targets_semantic([target], IGNORE_INDEX)[0].to(device)

        # Ignore pixels where EITHER pred or gt maps to void (255).
        # torchmetrics only filters gt==ignore_index, not pred==ignore_index,
        # so pred values of 255 would overflow the confusion matrix.
        ignore_mask = (pred_city == IGNORE_INDEX) | (gt == IGNORE_INDEX)
        gt_upd = gt.clone()
        gt_upd[ignore_mask] = IGNORE_INDEX
        pred_upd = pred_city.clone()
        pred_upd[ignore_mask] = 0  # any valid class; ignored via gt_upd

        metric.update(pred_upd[None], gt_upd[None])

    return metric.compute()


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_results(label: str, iou_per_class: torch.Tensor):
    miou = iou_per_class.mean().item()
    print(f"\n{'=' * 56}")
    print(f"  {label}")
    print(f"  {'Class':<22} {'IoU (%)':>10}")
    print(f"{'-' * 56}")
    for name, iou in zip(CITYSCAPES_CLASS_NAMES, iou_per_class.tolist()):
        print(f"  {name:<22} {iou * 100:>10.1f}")
    print(f"{'-' * 56}")
    print(f"  {'mIoU':<22} {miou * 100:>10.1f}")
    print(f"{'=' * 56}")


def print_comparison(city_iou: torch.Tensor, coco_iou: torch.Tensor):
    print(f"\n{'=' * 70}")
    print("  COMPARISON: Cityscapes model vs COCO model (on Cityscapes val)")
    print(f"  {'Class':<22} {'City (%)':>12} {'COCO (%)':>12} {'Δ':>8}")
    print(f"{'-' * 70}")
    for name, c, k in zip(CITYSCAPES_CLASS_NAMES,
                           city_iou.tolist(), coco_iou.tolist()):
        delta = (k - c) * 100
        print(f"  {name:<22} {c * 100:>12.1f} {k * 100:>12.1f} {delta:>+8.1f}")
    print(f"{'-' * 70}")
    city_m = city_iou.mean().item()
    coco_m = coco_iou.mean().item()
    print(f"  {'mIoU':<22} {city_m * 100:>12.1f} {coco_m * 100:>12.1f} "
          f"{(coco_m - city_m) * 100:>+8.1f}")
    print(f"{'=' * 70}")
    print("\nNote: 'rider' IoU ≈ 0 for COCO model is expected — COCO has no")
    print("      rider class (riders are annotated as 'person').")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate both EoMT models on Cityscapes val using the same "
            "semantic-inference pipeline."
        )
    )
    parser.add_argument("--cityscapes_path", default=DEFAULT_CITYSCAPES_PATH)
    parser.add_argument("--city_config",     default=DEFAULT_CITY_CONFIG)
    parser.add_argument("--city_ckpt",       default=DEFAULT_CITY_CKPT)
    parser.add_argument("--coco_config",     default=DEFAULT_COCO_CONFIG)
    parser.add_argument("--coco_ckpt",       default=DEFAULT_COCO_CKPT)
    parser.add_argument("--device",          type=int, default=0)
    parser.add_argument("--skip_city",       action="store_true")
    parser.add_argument("--skip_coco",       action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Cityscapes dataset (shared by both models) ---
    print("\nLoading Cityscapes val dataset ...")
    with open(args.city_config) as f:
        city_config = yaml.safe_load(f)

    data_mod_name, data_cls_name = city_config["data"]["class_path"].rsplit(".", 1)
    data_cls = getattr(importlib.import_module(data_mod_name), data_cls_name)
    data_kwargs = city_config["data"].get("init_args", {})

    cs_data = data_cls(
        path=args.cityscapes_path,
        batch_size=1,
        num_workers=0,
        check_empty_targets=False,
        **data_kwargs,
    ).setup()

    val_dataset = cs_data.val_dataloader().dataset
    print(f"  {len(val_dataset)} validation images, "
          f"{cs_data.num_classes} classes, img_size={cs_data.img_size}")

    city_iou = coco_iou = None

    # --- Cityscapes model ---
    if not args.skip_city:
        print("\nBuilding Cityscapes model ...")
        city_model, city_lit_cls, city_model_kw = build_model(
            city_config, cs_data.num_classes, cs_data.img_size, device
        )
        city_model = load_weights(
            city_model, city_config, city_lit_cls, city_model_kw,
            cs_data.num_classes, cs_data.img_size, device,
            local_ckpt=args.city_ckpt,
        )
        print("Running evaluation (Cityscapes model) ...")
        city_iou = evaluate(city_model, val_dataset, device, CITY_TO_CITYSCAPES)
        print_results("Cityscapes-trained model", city_iou)
        del city_model
        torch.cuda.empty_cache()

    # --- COCO model ---
    if not args.skip_coco:
        with open(args.coco_config) as f:
            coco_config = yaml.safe_load(f)
        print("\nBuilding COCO model ...")
        coco_model, coco_lit_cls, coco_model_kw = build_model(
            coco_config, COCO_NUM_CLASSES, COCO_IMG_SIZE, device
        )
        coco_model = load_weights(
            coco_model, coco_config, coco_lit_cls, coco_model_kw,
            COCO_NUM_CLASSES, COCO_IMG_SIZE, device,
            local_ckpt=args.coco_ckpt,
        )
        print("Running evaluation (COCO model) ...")
        coco_iou = evaluate(coco_model, val_dataset, device, COCO_TO_CITYSCAPES)
        print_results("COCO-trained model (remapped to Cityscapes)", coco_iou)
        del coco_model
        torch.cuda.empty_cache()

    # --- Side-by-side comparison ---
    if city_iou is not None and coco_iou is not None:
        print_comparison(city_iou, coco_iou)


if __name__ == "__main__":
    main()
