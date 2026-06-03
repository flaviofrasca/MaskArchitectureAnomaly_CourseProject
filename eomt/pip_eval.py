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
    python eval_coco_on_cityscapes.py \
        --cityscapes_path /path/to/cityscapes \
        --city_ckpt /path/to/city.bin \
        --coco_ckpt /path/to/coco.bin
"""

import argparse
import glob
import importlib
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image as PILImage
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from torch.amp.autocast_mode import autocast
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

# ERFNet lives in ../eval/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'eval'))

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

BASE = "/content/drive/.shortcut-targets-by-id/1osgiWms0a4SYz1evCZNwMV-jv--0I0RU/MaskArch_Shared"

DEFAULT_CITYSCAPES_PATH   = BASE + "/datasets/cityscapes"
DEFAULT_CITY_CKPT         = BASE + "/checkpoints/cityscapes/eomt_cityscapes.bin"
DEFAULT_COCO_CKPT         = BASE + "/checkpoints/coco/eomt_coco.bin"
DEFAULT_FINETUNED_CKPT    = BASE + "/checkpoints/finetuned_v2/phase3_unfreeze_more/epoch=2-step=2232.ckpt"
DEFAULT_ERFNET_CKPT       = "../trained_models/erfnet_pretrained.pth"

DEFAULT_CITY_CONFIG       = "configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"
DEFAULT_COCO_CONFIG       = "configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml"
DEFAULT_FINETUNED_CONFIG  = DEFAULT_CITY_CONFIG  # same architecture as cityscapes

COCO_IMG_SIZE    = (640, 640)
COCO_NUM_CLASSES = 133

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IGNORE_INDEX           = 255
NUM_CITYSCAPES_CLASSES = 19
ERF_NUM_CLASSES        = 20  # 0-18 valid, 19 = unlabeled → ignore

CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]

# COCO contiguous index (0-132) → Cityscapes train_id (0-18 or 255)
DENSE_COCO_TO_CITYSCAPES_TRAINID = {
    0: 11, 1: 18, 2: 13, 3: 17, 4: 255, 5: 15, 6: 16, 7: 14, 8: 255,
    9: 6, 10: 255, 11: 7, 12: 255, 13: 255, 14: 255, 15: 255, 16: 255,
    17: 255, 18: 255, 19: 255, 20: 255, 21: 255, 22: 255, 23: 255,
    24: 255, 25: 255, 26: 255, 27: 255, 28: 255, 29: 255, 30: 255,
    31: 255, 32: 255, 33: 255, 34: 255, 35: 255, 36: 255, 37: 255,
    38: 255, 39: 255, 40: 255, 41: 255, 42: 255, 43: 255, 44: 255,
    45: 255, 46: 255, 47: 255, 48: 255, 49: 255, 50: 255, 51: 255,
    52: 255, 53: 255, 54: 255, 55: 255, 56: 255, 57: 255, 58: 255,
    59: 255, 60: 255, 61: 255, 62: 255, 63: 255, 64: 255, 65: 255,
    66: 255, 67: 255, 68: 255, 69: 255, 70: 255, 71: 255, 72: 255,
    73: 255, 74: 255, 75: 255, 76: 255, 77: 255, 78: 255, 79: 255,
    80: 255, 81: 255, 82: 255, 83: 255, 84: 255, 85: 255, 86: 255,
    87: 255, 88: 8,  89: 255, 90: 255, 91: 2,  92: 255, 93: 255,
    94: 255, 95: 255, 96: 255, 97: 255, 98: 255, 99: 255, 100: 0,
    101: 2, 102: 255, 103: 255, 104: 255, 105: 255, 106: 255, 107: 255,
    108: 255, 109: 3,  110: 3,  111: 3,  112: 3,  113: 255, 114: 255,
    115: 255, 116: 8,  117: 4,  118: 255, 119: 10, 120: 255, 121: 255,
    122: 255, 123: 1,  124: 255, 125: 9,  126: 9,  127: 255, 128: 255,
    129: 2,  130: 255, 131: 3,  132: 255,
}

COCO_TO_CITYSCAPES = torch.full((133,), IGNORE_INDEX, dtype=torch.long)
for coco_id, city_id in DENSE_COCO_TO_CITYSCAPES_TRAINID.items():
    COCO_TO_CITYSCAPES[coco_id] = city_id

DENSE_COCO_TO_CITYSCAPES_TRAINID = {
    0: 11,   # person -> person
    1: 18,   # bicycle -> bicycle
    2: 13,   # car -> car
    3: 17,   # motorcycle -> motorcycle
    4: 255,  # airplane -> void
    5: 15,   # bus -> bus
    6: 16,   # train -> train
    7: 14,   # truck -> truck
    8: 255,  # boat -> void
    9: 6,    # traffic light -> traffic light
    10: 255, # fire hydrant -> void
    11: 7,   # stop sign -> traffic sign
    12: 255, # parking meter -> void
    13: 255, # bench -> void
    14: 255, # bird -> void
    15: 255, # cat -> void
    16: 255, # dog -> void
    17: 255, # horse -> void
    18: 255, # sheep -> void
    19: 255, # cow -> void
    20: 255, # elephant -> void
    21: 255, # bear -> void
    22: 255, # zebra -> void
    23: 255, # giraffe -> void
    24: 255, # backpack -> void
    25: 255, # umbrella -> void
    26: 255, # handbag -> void
    27: 255, # tie -> void
    28: 255, # suitcase -> void
    29: 255, # frisbee -> void
    30: 255, # skis -> void
    31: 255, # snowboard -> void
    32: 255, # sports ball -> void
    33: 255, # kite -> void
    34: 255, # baseball bat -> void
    35: 255, # baseball glove -> void
    36: 255, # skateboard -> void
    37: 255, # surfboard -> void
    38: 255, # tennis racket -> void
    39: 255, # bottle -> void
    40: 255, # wine glass -> void
    41: 255, # cup -> void
    42: 255, # fork -> void
    43: 255, # knife -> void
    44: 255, # spoon -> void
    45: 255, # bowl -> void
    46: 255, # banana -> void
    47: 255, # apple -> void
    48: 255, # sandwich -> void
    49: 255, # orange -> void
    50: 255, # broccoli -> void
    51: 255, # carrot -> void
    52: 255, # hot dog -> void
    53: 255, # pizza -> void
    54: 255, # donut -> void
    55: 255, # cake -> void
    56: 255, # chair -> void
    57: 255, # couch -> void
    58: 255, # potted plant -> void
    59: 255, # bed -> void
    60: 255, # dining table -> void
    61: 255, # toilet -> void
    62: 255, # tv -> void
    63: 255, # laptop -> void
    64: 255, # mouse -> void
    65: 255, # remote -> void
    66: 255, # keyboard -> void
    67: 255, # cell phone -> void
    68: 255, # microwave -> void
    69: 255, # oven -> void
    70: 255, # toaster -> void
    71: 255, # sink -> void
    72: 255, # refrigerator -> void
    73: 255, # book -> void
    74: 255, # clock -> void
    75: 255, # vase -> void
    76: 255, # scissors -> void
    77: 255, # teddy bear -> void
    78: 255, # hair drier -> void
    79: 255, # toothbrush -> void
    80: 255, # banner -> void
    81: 255, # blanket -> void
    82: 255, # bridge -> void
    83: 255, # cardboard -> void
    84: 255, # counter -> void
    85: 255, # curtain -> void
    86: 255, # door-stuff -> void
    87: 255, # floor-wood -> void
    88: 8,   # flower -> vegetation
    89: 255, # fruit -> void
    90: 255, # gravel -> void
    91: 2,   # house -> building
    92: 255, # light -> void
    93: 255, # mirror-stuff -> void
    94: 255, # net -> void
    95: 255, # pillow -> void
    96: 255, # platform -> void
    97: 255, # playingfield -> void
    98: 255, # railroad -> void
    99: 255, # river -> void
    100: 0,  # road -> road
    101: 2,  # roof -> building
    102: 255, # sand -> void
    103: 255, # sea -> void
    104: 255, # shelf -> void
    105: 255, # snow -> void
    106: 255, # stairs -> void
    107: 255, # tent -> void
    108: 255, # towel -> void
    109: 3,  # wall-brick -> wall
    110: 3,  # wall-stone -> wall
    111: 3,  # wall-tile -> wall
    112: 3,  # wall-wood -> wall
    113: 255, # water-other -> void
    114: 255, # window-blind -> void
    115: 255, # window-other -> void
    116: 8,  # tree-merged -> vegetation
    117: 4,  # fence-merged -> fence
    118: 255, # ceiling-merged -> void
    119: 10, # sky-other-merged -> sky
    120: 255, # cabinet-merged -> void
    121: 255, # table-merged -> void
    122: 255, # floor-other-merged -> void
    123: 1,  # pavement-merged -> sidewalk
    124: 255, # mountain-merged -> void
    125: 9,  # grass-merged -> terrain
    126: 9,  # dirt-merged -> terrain
    127: 255, # paper-merged -> void
    128: 255, # food-other-merged -> void
    129: 2,  # building-other-merged -> building
    130: 255, # rock-merged -> void
    131: 3,  # wall-other-merged -> wall
    132: 255  # rug-merged -> void
}

# Populate the tensor using the new mapping dictionary
for coco_idx, city_idx in DENSE_COCO_TO_CITYSCAPES_TRAINID.items():
    COCO_TO_CITYSCAPES[coco_idx] = city_idx

# Identity remap for the Cityscapes model (already predicts 19 classes).
CITY_TO_CITYSCAPES = torch.arange(NUM_CITYSCAPES_CLASSES, dtype=torch.long)

# ERFNet: class 19 (unlabeled) → ignore, classes 0-18 → identity
ERFNET_TO_CITYSCAPES = torch.full((ERF_NUM_CLASSES,), IGNORE_INDEX, dtype=torch.long)
ERFNET_TO_CITYSCAPES[:NUM_CITYSCAPES_CLASSES] = torch.arange(NUM_CITYSCAPES_CLASSES)

# Cityscapes labelId (0-33) → ERFNet trainId (0-18), 19 = ignore
LABEL_ID_TO_TRAIN = torch.full((256,), 19, dtype=torch.long)
for _lid, _tid in {7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7, 21:8,
                   22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15,
                   31:16, 32:17, 33:18}.items():
    LABEL_ID_TO_TRAIN[_lid] = _tid

# ---------------------------------------------------------------------------
# Suppress Lightning checkpoint warning
# ---------------------------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
)

# ---------------------------------------------------------------------------
# EoMT model building and weight loading
# ---------------------------------------------------------------------------

def build_eomt_model(config: dict, num_classes: int, img_size: tuple, device):
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
    if "stuff_classes" in config.get("data", {}).get("init_args", {}):
        model_kw["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]
    model = lit_cls(img_size=img_size, num_classes=num_classes, network=network, **model_kw)
    return model.eval().to(device), lit_cls, model_kw


def load_eomt_weights(model, config, lit_cls, model_kw, num_classes, img_size,
                      device, local_ckpt=None):
    """Load weights from a local .bin or .ckpt file, or fall back to HuggingFace Hub."""
    if local_ckpt and os.path.isfile(local_ckpt):
        raw = torch.load(local_ckpt, map_location=device, weights_only=False)
        # Lightning .ckpt files wrap weights under 'state_dict'
        state_dict = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw
        model_state = model.state_dict()
        state_dict = {
            k: v for k, v in state_dict.items()
            if k not in model_state or v.shape == model_state[k].shape
        }
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded weights: {local_ckpt}")
        return model

    name = config.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name")
    if name is None:
        warnings.warn("No logger name in config; skipping weight download.")
        return model
    try:
        ckpt_path = hf_hub_download(repo_id=f"tue-mps/{name}", filename="pytorch_model.bin")
        raw = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded HuggingFace weights: tue-mps/{name}")
    except RepositoryNotFoundError:
        warnings.warn(f"HF repo not found for {name}. Using random weights.")
    return model


# ---------------------------------------------------------------------------
# ERFNet model loading
# ---------------------------------------------------------------------------

def load_erfnet(ckpt_path: str, device):
    from erfnet import ERFNet

    model = ERFNet(ERF_NUM_CLASSES)
    if device.type != 'cpu':
        model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        own_state  = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                # handle 'module.' prefix mismatch
                alt = name.split("module.")[-1]
                if alt in own_state:
                    own_state[alt].copy_(param)
            else:
                own_state[name].copy_(param)
        print(f"  Loaded ERFNet weights: {ckpt_path}")
    else:
        print(f"  [WARNING] ERFNet checkpoint not found: {ckpt_path}")

    return model.eval()


# ---------------------------------------------------------------------------
# Evaluation loop — EoMT (mask-based)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_eomt(model, val_dataset, device, remap: torch.Tensor):
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    remap  = remap.to(device)
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
            imgs      = [img]
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

        pred      = logits[0].argmax(0)
        pred_city = remap[pred]

        gt = model.to_per_pixel_targets_semantic([target], IGNORE_INDEX)[0].to(device)

        ignore_mask = (pred_city == IGNORE_INDEX) | (gt == IGNORE_INDEX)
        gt_upd   = gt.clone();        gt_upd[ignore_mask]   = IGNORE_INDEX
        pred_upd = pred_city.clone(); pred_upd[ignore_mask] = 0

        metric.update(pred_upd[None], gt_upd[None])

    return metric.compute()


# ---------------------------------------------------------------------------
# Evaluation loop — ERFNet (pixel-based, uses iouEval as in the original repo)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_erfnet(model, cityscapes_path: str, device):
    import zipfile
    from io import BytesIO
    from iouEval import iouEval
    from transform import Relabel, ToLabel

    img_transform = Compose([
        Resize(512, PILImage.BILINEAR),
        ToTensor(),
    ])
    target_transform = Compose([
        Resize(512, PILImage.NEAREST),
        ToLabel(),
        Relabel(255, 19),  # ignore label → 19 (as in original eval_iou.py)
    ])

    img_zip_path = os.path.join(cityscapes_path, "leftImg8bit_trainvaltest.zip")
    gt_zip_path  = os.path.join(cityscapes_path, "gtFine_trainvaltest.zip")

    if not os.path.isfile(img_zip_path) or not os.path.isfile(gt_zip_path):
        print("  [WARNING] Cityscapes zip files not found — check --cityscapes_path")
        return None

    iou_eval = iouEval(ERF_NUM_CLASSES)  # ignoreIndex=19 by default

    with zipfile.ZipFile(img_zip_path, 'r') as img_zip, \
         zipfile.ZipFile(gt_zip_path,  'r') as gt_zip:

        # Normalize: strip leading './' so lookups are consistent
        def norm(p): return p.lstrip('./')

        img_names   = sorted([n for n in img_zip.namelist()
                              if 'leftImg8bit/val' in norm(n) and n.endswith('.png')])
        gt_name_map = {norm(n): n for n in gt_zip.namelist()}

        if not img_names:
            print("  [WARNING] No val images found inside zip")
            return None

        for img_name in tqdm(img_names, desc="Evaluating ERFNet", unit="img"):
            gt_name_norm = norm(img_name).replace('leftImg8bit/', 'gtFine/').replace(
                '_leftImg8bit.png', '_gtFine_labelIds.png'
            )
            if gt_name_norm not in gt_name_map:
                continue

            with img_zip.open(img_name) as f:
                img = PILImage.open(BytesIO(f.read())).convert('RGB')
            with gt_zip.open(gt_name_map[gt_name_norm]) as f:
                gt_np = np.array(PILImage.open(BytesIO(f.read())).resize(
                    (1024, 512), PILImage.NEAREST))  # [512, 1024]

            img_t = img_transform(img).unsqueeze(0).float().to(device)
            # Convert labelIds → trainIds (0-18, 19=ignore)
            gt_t  = LABEL_ID_TO_TRAIN[torch.from_numpy(gt_np.astype(np.int64))] \
                        .unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 512, 1024]

            logits = model(img_t)
            pred   = logits.max(1)[1].unsqueeze(1).data

            iou_eval.addBatch(pred, gt_t)

    miou, iou_per_class = iou_eval.getIoU()
    return iou_per_class


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


def print_comparison(results: dict):
    """results: {label: iou_tensor or None}"""
    labels = [k for k, v in results.items() if v is not None]
    ious   = [results[k] for k in labels]

    width  = max(12, max(len(l) for l in labels))
    header = f"  {'Class':<22}" + "".join(f" {l:>{width}}" for l in labels)
    sep    = "=" * (24 + (width + 1) * len(labels))

    print(f"\n{sep}")
    print("  COMPARISON — all models on Cityscapes val")
    print(header)
    print("-" * (24 + (width + 1) * len(labels)))

    for cls_name, *cls_ious in zip(CITYSCAPES_CLASS_NAMES, *[i.tolist() for i in ious]):
        row = f"  {cls_name:<22}" + "".join(f" {v * 100:>{width}.1f}" for v in cls_ious)
        print(row)

    print("-" * (24 + (width + 1) * len(labels)))
    miou_row = f"  {'mIoU':<22}" + "".join(
        f" {i.mean().item() * 100:>{width}.1f}" for i in ious
    )
    print(miou_row)
    print(sep)

    if "EoMT COCO" in results and results["EoMT COCO"] is not None:
        print("\nNote: 'rider' IoU ≈ 0 for COCO model — COCO annotates riders as 'person'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes_path",  default=DEFAULT_CITYSCAPES_PATH)
    parser.add_argument("--city_config",      default=DEFAULT_CITY_CONFIG)
    parser.add_argument("--city_ckpt",        default=DEFAULT_CITY_CKPT)
    parser.add_argument("--coco_config",      default=DEFAULT_COCO_CONFIG)
    parser.add_argument("--coco_ckpt",        default=DEFAULT_COCO_CKPT)
    parser.add_argument("--finetuned_config", default=DEFAULT_FINETUNED_CONFIG)
    parser.add_argument("--finetuned_ckpt",   default=DEFAULT_FINETUNED_CKPT)
    parser.add_argument("--erfnet_ckpt",      default=DEFAULT_ERFNET_CKPT)
    parser.add_argument("--device",           type=int, default=0)
    parser.add_argument("--skip_city",        action="store_true")
    parser.add_argument("--skip_coco",        action="store_true")
    parser.add_argument("--skip_finetuned",   action="store_true")
    parser.add_argument("--skip_erfnet",      action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Cityscapes val dataset (shared by EoMT models) ---
    print("\nLoading Cityscapes val dataset ...")
    with open(args.city_config) as f:
        city_config = yaml.safe_load(f)

    data_mod_name, data_cls_name = city_config["data"]["class_path"].rsplit(".", 1)
    data_cls    = getattr(importlib.import_module(data_mod_name), data_cls_name)
    data_kwargs = city_config["data"].get("init_args", {})
    cs_data = data_cls(
        path=args.cityscapes_path, batch_size=1, num_workers=0,
        check_empty_targets=False, **data_kwargs,
    ).setup()
    val_dataset = cs_data.val_dataloader().dataset
    print(f"  {len(val_dataset)} val images")

    all_results = {}

    # --- 1. EoMT Cityscapes ---
    if not args.skip_city:
        print("\n[1/4] EoMT Cityscapes ...")
        model, lit_cls, model_kw = build_eomt_model(
            city_config, cs_data.num_classes, cs_data.img_size, device
        )
        model = load_eomt_weights(model, city_config, lit_cls, model_kw,
                                  cs_data.num_classes, cs_data.img_size, device,
                                  local_ckpt=args.city_ckpt)
        iou = evaluate_eomt(model, val_dataset, device, CITY_TO_CITYSCAPES)
        all_results["EoMT CS"] = iou
        print_results("EoMT Cityscapes", iou)
        del model; torch.cuda.empty_cache()

    # --- 2. EoMT COCO ---
    if not args.skip_coco:
        print("\n[2/4] EoMT COCO ...")
        with open(args.coco_config) as f:
            coco_config = yaml.safe_load(f)
        model, lit_cls, model_kw = build_eomt_model(
            coco_config, COCO_NUM_CLASSES, COCO_IMG_SIZE, device
        )
        model = load_eomt_weights(model, coco_config, lit_cls, model_kw,
                                  COCO_NUM_CLASSES, COCO_IMG_SIZE, device,
                                  local_ckpt=args.coco_ckpt)
        iou = evaluate_eomt(model, val_dataset, device, COCO_TO_CITYSCAPES)
        all_results["EoMT COCO"] = iou
        print_results("EoMT COCO (remapped)", iou)
        del model; torch.cuda.empty_cache()

    # --- 3. EoMT Finetuned ---
    if not args.skip_finetuned:
        print("\n[3/4] EoMT Finetuned ...")
        with open(args.finetuned_config) as f:
            ft_config = yaml.safe_load(f)
        model, lit_cls, model_kw = build_eomt_model(
            ft_config, cs_data.num_classes, cs_data.img_size, device
        )
        model = load_eomt_weights(model, ft_config, lit_cls, model_kw,
                                  cs_data.num_classes, cs_data.img_size, device,
                                  local_ckpt=args.finetuned_ckpt)
        iou = evaluate_eomt(model, val_dataset, device, CITY_TO_CITYSCAPES)
        all_results["EoMT FT"] = iou
        print_results("EoMT Finetuned", iou)
        del model; torch.cuda.empty_cache()

    # --- 4. ERFNet ---
    if not args.skip_erfnet:
        print("\n[4/4] ERFNet ...")
        erf_model = load_erfnet(args.erfnet_ckpt, device)
        iou = evaluate_erfnet(erf_model, args.cityscapes_path, device)
        if iou is not None:
            all_results["ERFNet"] = iou
            print_results("ERFNet", iou)
        del erf_model; torch.cuda.empty_cache()

    # --- Final comparison ---
    if len(all_results) > 1:
        print_comparison(all_results)


if __name__ == "__main__":
    main()
