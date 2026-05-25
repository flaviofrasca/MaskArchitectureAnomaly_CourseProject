"""
Step 8 — Temperature Scaling baseline with EoMT.

PRO TIP implementation: runs inference once and caches pixel-level logits to disk,
then sweeps temperatures [0.5, 0.75, 1.1] + grid search for best T,
all without re-running the model forward pass.

Run from eomt/ directory:
    python eval_temperature_scaling.py --model_type cityscapes
    python eval_temperature_scaling.py --model_type cityscapes --force_cache  # re-run inference
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
    build_and_load_model, ANOMALY_DIR, DATASETS, CHECKPOINTS, IMG_SIZE
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

    saved = 0
    for i, path in enumerate(image_paths):
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
            )[0].float()   # [C, H, W]

        # Resize GT to match logits spatial size
        H, W = pixel_logits.shape[1], pixel_logits.shape[2]
        if ood_gts.shape != (H, W):
            ood_gts = np.array(
                Image.fromarray(ood_gts).resize((W, H), Image.NEAREST)
            )

        slug = _slug(dataset_name)
        np.save(os.path.join(cache_dir, f"{slug}_{i:04d}_logits.npy"),
                pixel_logits.cpu().numpy().astype(np.float16))
        np.save(os.path.join(cache_dir, f"{slug}_{i:04d}_gt.npy"), ood_gts)
        saved += 1

    return saved


# ---------------------------------------------------------------------------
# Step 2: Evaluate at a given temperature
# ---------------------------------------------------------------------------

def msp_at_temperature(logits, T):
    """logits: [C, H, W] float32 numpy. Returns anomaly score [H, W]."""
    t = torch.tensor(logits, dtype=torch.float32)
    return (1.0 - torch.softmax(t / T, dim=0).max(dim=0)[0]).numpy()


def evaluate_temperature(cache_dir, dataset_name, T):
    slug = _slug(dataset_name)
    logits_files = sorted(glob.glob(os.path.join(cache_dir, f"{slug}_*_logits.npy")))
    if not logits_files:
        return None

    all_scores, all_gts = [], []
    for lp in logits_files:
        gp = lp.replace("_logits.npy", "_gt.npy")
        if not os.path.exists(gp):
            continue
        logits = np.load(lp).astype(np.float32)
        gt     = np.load(gp)
        all_scores.append(msp_at_temperature(logits, T))
        all_gts.append(gt)

    if not all_scores:
        return None

    gts    = np.array(all_scores)   # reuse variable for brevity
    scores = np.array(all_scores)
    gts    = np.array(all_gts)

    val_out   = np.concatenate((scores[gts == 0], scores[gts == 1]))
    val_label = np.concatenate((np.zeros((gts == 0).sum()), np.ones((gts == 1).sum())))

    return average_precision_score(val_label, val_out) * 100.0, \
           fpr_at_95_tpr(val_out, val_label) * 100.0


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

    dataset_names   = [name for name, _ in DATASETS]
    dataset_display = ["RA-21", "RO-21", "L&F", "Static", "RoadAnom"]

    results_file = open('results_temperature_scaling.txt', 'a')
    results_file.write(f"\n=== EoMT [{args.model_type}] | Temperature Scaling ===\n")
    results_file.write(f"{'Method':<16}  {'mIoU':>6}  ")
    for d in dataset_display:
        results_file.write(f"  {d} AuPRC  {d} FPR95  ")
    results_file.write("\n" + "-" * 100 + "\n")

    print(f"\n{'='*75}")
    print(f"Model: EoMT [{args.model_type}]  |  Temperature Scaling on MSP")
    print(f"{'='*75}")
    print(f"{'Method':<16}  {'mIoU':>6}", end="")
    for d in dataset_display:
        print(f"  {d+' AuPRC':>12}  {d+' FPR95':>12}", end="")
    print(f"\n{'-'*75}")

    def print_and_write_row(label, results_per_ds):
        line = f"  {label:<14}  {'N/A':>6}"
        print(f"{label:<16}  {'N/A':>6}", end="")
        for r in results_per_ds:
            if r is not None:
                line += f"  {r[0]:>12.2f}  {r[1]:>12.2f}"
                print(f"  {r[0]:>12.2f}  {r[1]:>12.2f}", end="")
            else:
                line += f"  {'SKIP':>12}  {'SKIP':>12}"
                print(f"  {'SKIP':>12}  {'SKIP':>12}", end="")
        print()
        results_file.write(line + "\n")

    # Fixed temperatures
    for T, label in zip(T_fixed, T_labels):
        row = [evaluate_temperature(cache_dir, n, T) for n in dataset_names]
        print_and_write_row(label, row)

    # Best T per dataset via grid search
    print(f"\n  Grid search best T (T ∈ [0.10, 2.00] step 0.05) ...")
    best = {}
    for name in dataset_names:
        best_auprc, best_T, best_fpr95 = -1.0, 1.0, 100.0
        for T in T_grid:
            r = evaluate_temperature(cache_dir, name, T)
            if r and r[0] > best_auprc:
                best_auprc, best_fpr95, best_T = r[0], r[1], T
        best[name] = (best_auprc, best_fpr95, best_T)

    row = [(best[n][0], best[n][1]) if best[n][0] >= 0 else None for n in dataset_names]
    print_and_write_row("MSP (best t)", row)

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
