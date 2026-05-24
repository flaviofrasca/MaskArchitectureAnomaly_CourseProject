# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def compute_anomaly_score(result, method):
    """
    Compute per-pixel anomaly score from model logits.

    Args:
        result: raw logits tensor, shape [1, C, H, W]
        method: 'MSP' | 'MaxLogit' | 'MaxEntropy'

    Returns:
        anomaly_score: numpy array [H, W], higher = more anomalous
    """
    logits = result.squeeze(0).data.cpu().numpy()  # [C, H, W]

    if method == 'MaxLogit':
        # Anomaly score: negative of max logit (lower max logit = more anomalous)
        anomaly_score = -np.max(logits, axis=0)

    elif method == 'MSP':
        # Maximum Softmax Probability: anomaly = 1 - max(softmax(logits))
        logits_t = torch.tensor(logits)
        softmax_probs = torch.softmax(logits_t, dim=0).numpy()
        anomaly_score = 1.0 - np.max(softmax_probs, axis=0)

    elif method == 'MaxEntropy':
        # Entropy of softmax distribution: higher entropy = more uncertain = more anomalous
        logits_t = torch.tensor(logits)
        softmax_probs = torch.softmax(logits_t, dim=0).numpy()
        # Shannon entropy: -sum(p * log(p)), clamp p to avoid log(0)
        eps = 1e-8
        anomaly_score = -np.sum(softmax_probs * np.log(softmax_probs + eps), axis=0)

    else:
        raise ValueError(f"Unknown method: {method}. Choose from MSP, MaxLogit, MaxEntropy.")

    return anomaly_score


def evaluate_dataset(dataset_path, model, method, device):
    """
    Evaluate anomaly detection on a single dataset split.

    Args:
        dataset_path: glob pattern pointing to images (e.g. '/path/images/*.jpg')
        model: loaded ERFNet model in eval mode
        method: anomaly scoring method name
        device: torch device

    Returns:
        (auprc, fpr95) tuple, or None if no valid images found
    """
    anomaly_score_list = []
    ood_gts_list = []

    image_paths = glob.glob(os.path.expanduser(dataset_path))
    if not image_paths:
        print(f"  [WARNING] No images found at: {dataset_path}")
        return None

    for path in image_paths:
        # Load and preprocess image — ToTensor gives [C,H,W], unsqueeze gives [1,C,H,W]
        image = input_transform(Image.open(path).convert('RGB')).unsqueeze(0).float().to(device)

        with torch.no_grad():
            result = model(image)

        anomaly_result = compute_anomaly_score(result, method)

        # Derive ground truth mask path from image path
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace(".webp", ".png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace(".jpg", ".png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace(".jpg", ".png")

        if not os.path.exists(pathGT):
            print(f"  [WARNING] GT mask not found: {pathGT}")
            continue

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        # Remap anomaly labels per dataset convention
        # RoadAnomaly (original): anomaly label = 2, must remap to 1
        # RoadAnomaly21 (SMIYC): anomaly label already = 1, no remap needed
        if "RoadAnomaly" in pathGT and "RoadAnomaly21" not in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)

        # Skip images with no anomaly pixels
        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)
        anomaly_score_list.append(anomaly_result)

        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    if len(ood_gts_list) == 0:
        print("  [WARNING] No valid images with anomaly labels found.")
        return None

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((np.zeros(len(ind_out)), np.ones(len(ood_out))))

    auprc = average_precision_score(val_label, val_out)
    fpr95 = fpr_at_95_tpr(val_out, val_label)

    return auprc * 100.0, fpr95 * 100.0


def main():
    parser = ArgumentParser()
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument(
        '--method',
        default='MSP',
        choices=['MSP', 'MaxLogit', 'MaxEntropy'],
        help='Anomaly scoring method to use',
    )
    parser.add_argument(
        '--datadir',
        default="/content/drive/.shortcut-targets-by-id/1osgiWms0a4SYz1evCZNwMV-jv--0I0RU/MaskArch_Shared/datasets/anomaly/",
        help='Root directory containing all anomaly dataset folders',
    )
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu else 'cuda')

    # Dataset definitions: (display_name, glob_pattern_relative_to_datadir)
    # Adjust folder names and extensions to match your actual extracted archive
    datasets = [
        ("SMIYC RA-21",  "RoadAnomaly21/images/*.png"),
        ("SMIYC RO-21",  "RoadObsticle21/images/*.webp"),
        ("FS L&F",       "LostAndFound/images/*.png"),
        ("FS Static",    "fs_static/images/*.jpg"),
        ("Road Anomaly", "RoadAnomaly/images/*.jpg"),
    ]

    weightspath = os.path.join(args.loadDir, args.loadWeights)
    print(f"Loading weights: {weightspath}")

    model = ERFNet(NUM_CLASSES)
    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(f"{name} not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print("Model and weights LOADED successfully")
    model.eval()

    results_file = open('results.txt', 'a')
    results_file.write(f"\n=== ERFNet | Method: {args.method} ===\n")
    print(f"\n{'='*60}")
    print(f"Method: {args.method}")
    print(f"{'='*60}")
    print(f"{'Dataset':<20} {'AuPRC':>8} {'FPR95':>8}")
    print(f"{'-'*40}")

    for name, pattern in datasets:
        full_pattern = os.path.join(args.datadir, pattern)
        print(f"{name:<20} ", end="", flush=True)
        result = evaluate_dataset(full_pattern, model, args.method, device)
        if result is not None:
            auprc, fpr95 = result
            print(f"{auprc:>7.2f}% {fpr95:>7.2f}%")
            results_file.write(f"  {name:<20} AuPRC: {auprc:.2f}%  FPR@95: {fpr95:.2f}%\n")
        else:
            print("  SKIPPED (no data)")
            results_file.write(f"  {name:<20} SKIPPED\n")

    print(f"{'='*60}\n")
    results_file.close()
    print("Results appended to results.txt")


if __name__ == '__main__':
    main()