"""
Step 8 — Temperature Scaling baseline with EoMT.

PRO TIP implementation: runs inference once and caches pixel-level logits to disk,
then sweeps temperatures [0.5, 0.75, 1.1] + grid search for best T.

Logits are loaded from disk ONCE per dataset, then all temperatures are evaluated
in memory — no redundant disk I/O during the grid search.

Run from eomt/ directory:
    python eval_temperature_scaling.py --model_type cityscapes
    python eval_temperature_scaling.py --model_type finetuned
    python eval_temperature_scaling.py --model_type cityscapes --force_cache
"""

import argparse
import glob
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score
from ood_metrics import fpr_at_95_tpr
from torch.amp.autocast_mode import autocast

warnings.filterwarnings("ignore")

from eval_anomaly_eomt import (
    build_and_load_model, ANOMALY_DIR, DATASETS, IMG_SIZE
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(name):
    return name.replace(" ", "_").replace("/", "-").replace("&", "and")


def _gt_path(img_path):
    p = img_path.replace("images", "labels_masks")
    if "RoadObsticle21" in p:
        p = p.replace(".webp", ".png")
    if "fs_static" in p:
        p = p.replace(".jpg", ".png")
    if "RoadAnomaly" in p and "RoadAnomaly21" not in p:
        p = p.replace(".jpg", ".png")
    return p


def _remap_gt(ood_gts, pathGT):
    if "RoadAnomaly" in pathGT and "RoadAnomaly21" not in pathGT:
        ood_gts = np.where(ood_gts == 2, 1, ood_gts)
    if "Streethazard" in pathGT:
        ood_gts = np.where(ood_gts == 14, 255, ood_gts)
        ood_gts = np.where(ood_gts < 20,  0,   ood_gts)
        ood_gts = np.where(ood_gts == 255, 1,   ood_gts)
    return ood_gts


# ---------------------------------------------------------------------------
# Step 1: Run model and cache pixel logits
# ---------------------------------------------------------------------------

@torch.no_grad()
def cache_logits(model, dataset_name, dataset_path, cache_dir, device):
    os.makedirs(cache_dir, exist_ok=True)
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    image_paths = sorted(glob.glob(os.path.expanduser(dataset_path)))
    if not image_paths:
        print(f"  [WARNING] No images found: {dataset_path}")
        return 0

    slug = _slug(dataset_name)
    saved = 0
    for i, path in enumerate(image_paths):
        logits_path = os.path.join(cache_dir, f"{slug}_{i:04d}_logits.npy")
        gt_path     = os.path.join(cache_dir, f"{slug}_{i:04d}_gt.npy")

        pathGT = _gt_path(path)
        if not os.path.exists(pathGT):
            continue

        ood_gts = np.array(Image.open(pathGT).convert('L'))
        ood_gts = _remap_gt(ood_gts, pathGT)
        if 1 not in np.unique(ood_gts):
            continue

        img_np     = np.array(Image.open(path).convert('RGB'))
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        imgs       = [img_tensor.to(device)]
        img_sizes  = [img_tensor.shape[-2:]]

        with autocast(dtype=torch.float16, device_type=device_type):
            crops, origins = model.window_imgs_semantic(imgs)
            mask_logits_per_layer, class_logits_per_layer = model(crops)
            mask_logits  = F.interpolate(
                mask_logits_per_layer[-1], model.img_size, mode="bilinear"
            )
            class_logits = class_logits_per_layer[-1]
            crop_logits  = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
            pixel_logits = model.revert_window_logits_semantic(
                crop_logits, origins, img_sizes
            )[0].float()  # [C, H, W]

        H, W = pixel_logits.shape[1], pixel_logits.shape[2]
        if ood_gts.shape != (H, W):
            ood_gts = np.array(
                Image.fromarray(ood_gts).resize((W, H), Image.NEAREST)
            )

        np.save(logits_path, pixel_logits.cpu().numpy().astype(np.float16))
        np.save(gt_path, ood_gts)
        saved += 1

    return saved


# ---------------------------------------------------------------------------
# Step 2: Load logits once, evaluate all temperatures in memory
# ---------------------------------------------------------------------------

_RESIZE_TO = (640, 640)  # resize logits at load time to keep computation fast

def load_dataset_logits(cache_dir, dataset_name):
    """Load cached logits and GTs, resizing to _RESIZE_TO for fast temperature sweep."""
    slug = _slug(dataset_name)
    logits_files = sorted(glob.glob(os.path.join(cache_dir, f"{slug}_*_logits.npy")))
    all_logits, all_gts = [], []
    for lp in logits_files:
        gp = lp.replace("_logits.npy", "_gt.npy")
        if not os.path.exists(gp):
            continue
        logit = np.load(lp).astype(np.float32)   # [C, H, W]
        gt    = np.load(gp)                        # [H, W]
        H, W  = logit.shape[1], logit.shape[2]
        if (H, W) != _RESIZE_TO:
            t = torch.tensor(logit).unsqueeze(0)  # [1, C, H, W]
            logit = F.interpolate(t, size=_RESIZE_TO, mode='bilinear',
                                  align_corners=False).squeeze(0).numpy()
            gt = np.array(Image.fromarray(gt).resize(
                (_RESIZE_TO[1], _RESIZE_TO[0]), Image.NEAREST))
        all_logits.append(logit)
        all_gts.append(gt)
    return all_logits, all_gts


def msp_score(logits, T):
    """logits: [C, H, W] float32. Returns anomaly score [H, W]."""
    t = torch.tensor(logits, dtype=torch.float32)
    return (1.0 - torch.softmax(t / T, dim=0).max(dim=0)[0]).numpy()


def metrics_from_scores(all_scores, all_gts):
    """Compute AuPRC and FPR95 from lists of score maps and GT masks (handles variable sizes)."""
    val_out_parts, val_label_parts = [], []
    for score, gt in zip(all_scores, all_gts):
        ood = score[gt == 1]
        ind = score[gt == 0]
        val_out_parts.append(np.concatenate([ind, ood]))
        val_label_parts.append(np.concatenate([np.zeros(len(ind)), np.ones(len(ood))]))
    val_out   = np.concatenate(val_out_parts)
    val_label = np.concatenate(val_label_parts)
    return (average_precision_score(val_label, val_out) * 100.0,
            fpr_at_95_tpr(val_out, val_label) * 100.0)


def evaluate_temperatures_for_dataset(all_logits, all_gts, T_list):
    """Evaluate all temperatures in T_list using pre-loaded logits. Returns dict T -> (auprc, fpr95)."""
    if not all_logits:
        return {T: None for T in T_list}
    results = {}
    for T in T_list:
        scores = [msp_score(logits, T) for logits in all_logits]
        results[T] = metrics_from_scores(scores, all_gts)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='cityscapes',
                        choices=['cityscapes', 'coco', 'finetuned'])
    parser.add_argument('--datadir',     default=ANOMALY_DIR)
    parser.add_argument('--cpu',         action='store_true')
    parser.add_argument('--force_cache', action='store_true',
                        help='Re-run inference even if cache exists')
    args = parser.parse_args()

    device    = torch.device('cpu' if args.cpu else 'cuda')
    cache_dir = f"logits_cache_{args.model_type}"

    # ---- Step 1: cache logits if needed ----
    cached = glob.glob(os.path.join(cache_dir, "*_logits.npy"))
    if not cached or args.force_cache:
        print(f"\nLoading EoMT [{args.model_type}] for logits caching ...")
        model = build_and_load_model(args.model_type, device)
        for name, pattern in DATASETS:
            full_pattern = os.path.join(args.datadir, pattern)
            print(f"  Caching: {name} ...", flush=True)
            n = cache_logits(model, name, full_pattern, cache_dir, device)
            print(f"    {n} images saved.")
        del model
        torch.cuda.empty_cache()
        print(f"\nLogits cached in: {cache_dir}/")
    else:
        print(f"\nUsing cached logits: {cache_dir}/  ({len(cached)} files)")

    # ---- Step 2: evaluate temperatures ----
    T_fixed  = [1.0, 0.5, 0.75, 1.1]
    T_labels = ["MSP", "MSP(t=0.5)", "MSP(t=0.75)", "MSP(t=1.1)"]
    T_grid   = list(np.round(np.arange(0.1, 2.05, 0.05), 2))
    T_all    = list(dict.fromkeys(T_fixed + T_grid))  # all unique temperatures

    dataset_names   = [name for name, _ in DATASETS]
    dataset_display = ["RA-21", "RO-21", "L&F", "Static", "RoadAnom"]

    results_file = open('results_temperature_scaling.txt', 'a')
    results_file.write(f"\n=== EoMT [{args.model_type}] | Temperature Scaling ===\n")

    print(f"\n{'='*75}")
    print(f"Model: EoMT [{args.model_type}]  |  Temperature Scaling on MSP")
    print(f"{'='*75}")

    # Load all logits once per dataset, evaluate all temperatures in memory
    print("Loading logits into memory and evaluating all temperatures ...")
    all_results = {}   # dataset_name -> {T -> (auprc, fpr95)}

    for name, disp in zip(dataset_names, dataset_display):
        print(f"  {disp} ...", flush=True)
        logits_list, gts_list = load_dataset_logits(cache_dir, name)
        all_results[name] = evaluate_temperatures_for_dataset(logits_list, gts_list, T_all)

    # ---- Print fixed temperatures ----
    header = f"{'Method':<16}  {'mIoU':>6}"
    for d in dataset_display:
        header += f"  {d+' AuPRC':>14}  {d+' FPR95':>14}"
    print(f"\n{header}\n{'-'*75}")
    results_file.write(header + "\n" + "-"*75 + "\n")

    def print_row(label, T):
        row = f"{label:<16}  {'N/A':>6}"
        for name in dataset_names:
            r = all_results[name].get(T)
            if r:
                row += f"  {r[0]:>14.2f}  {r[1]:>14.2f}"
            else:
                row += f"  {'SKIP':>14}  {'SKIP':>14}"
        print(row)
        results_file.write(row + "\n")

    for T, label in zip(T_fixed, T_labels):
        print_row(label, T)

    # ---- Best T per dataset ----
    best = {}
    for name in dataset_names:
        best_auprc, best_T, best_fpr95 = -1.0, 1.0, 100.0
        for T in T_grid:
            r = all_results[name].get(T)
            if r and r[0] > best_auprc:
                best_auprc, best_fpr95, best_T = r[0], r[1], T
        best[name] = (best_auprc, best_fpr95, best_T)

    row = f"{'MSP (best t)':<16}  {'N/A':>6}"
    for name in dataset_names:
        a, f, _ = best[name]
        row += f"  {a:>14.2f}  {f:>14.2f}" if a >= 0 else f"  {'SKIP':>14}  {'SKIP':>14}"
    print(row)
    results_file.write(row + "\n")

    print(f"\n  Best T per dataset:")
    results_file.write("\n  Best T:\n")
    for name, disp in zip(dataset_names, dataset_display):
        a, f, t = best[name]
        print(f"    {disp}: T = {t}  (AuPRC={a:.2f}%  FPR95={f:.2f}%)")
        results_file.write(f"    {disp}: T = {t}\n")

    print(f"{'='*75}\n")
    results_file.close()
    print("Results appended to results_temperature_scaling.txt")


if __name__ == '__main__':
    main()
