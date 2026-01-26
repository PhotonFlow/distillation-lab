import csv
import inspect
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets import CocoDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from models.SDG_FRCNN import SDGFasterRCNN

# ----------------------------------------
# Config
# ----------------------------------------
BENCHMARK_ROOT = Path("./datasets/custom_benchmark_splits")
OUTPUT_CSV = Path("./custom_benchmark_inference_results.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIXED_SIZE = (640, 640)

SDG_CHECKPOINT = Path("./checkpoints/unbiased_frcnn_best.pth")

RFDETR_VARIANT = "base"  # "base", "small", "medium"
RFDETR_CHECKPOINT = Path("./checkpoints/custom_training/rfdetr/rfdetr_best.pth")
RFDETR_SCORE_THRESHOLD = 0.001

MODEL_SPECS = [
    {
        "name": "SDG Faster R-CNN (train_benchmark)",
        "type": "sdg_frcnn",
        "checkpoint": SDG_CHECKPOINT,
        "label_offset": 1,  # background class at 0
        "pred_label_offset": 0,
        "resize_to": FIXED_SIZE,
        "batch_size": 4,
    },
    {
        "name": f"RF-DETR ({RFDETR_VARIANT}) (train_custom_models)",
        "type": "rfdetr",
        "checkpoint": RFDETR_CHECKPOINT,
        "variant": RFDETR_VARIANT,
        "label_offset": 0,
        "pred_label_offset": 0,
        "resize_to": None,
        "batch_size": 1,
    },
]


def get_transforms():
    return torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return tuple(zip(*batch))


class COCOWrapper(Dataset):
    def __init__(self, root, ann, transforms=None, resize_to=None, label_offset=1):
        self.coco = CocoDetection(root, ann)
        self.transforms = transforms
        self.resize_to = resize_to
        self.label_offset = label_offset

        cat_ids = sorted(self.coco.coco.getCatIds())
        self.cat_id_to_contiguous_id = {
            cat_id: idx + label_offset for idx, cat_id in enumerate(cat_ids)
        }

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        img, target = self.coco[idx]
        if self.transforms:
            img = self.transforms(img)

        original_h, original_w = img.shape[-2:]
        if self.resize_to is not None:
            img = F.interpolate(
                img.unsqueeze(0), size=self.resize_to, mode="bilinear", align_corners=False
            ).squeeze(0)
            scale_x = self.resize_to[1] / original_w
            scale_y = self.resize_to[0] / original_h
        else:
            scale_x = 1.0
            scale_y = 1.0

        boxes = []
        labels = []
        for obj in target:
            x, y, w, h = obj["bbox"]
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y

            if (x2 - x1) > 1 and (y2 - y1) > 1:
                cat_id = obj["category_id"]
                if cat_id in self.cat_id_to_contiguous_id:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(self.cat_id_to_contiguous_id[cat_id])

        if len(boxes) == 0:
            return None

        target_dict = {}
        target_dict["image_id"] = torch.tensor([idx])
        target_dict["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target_dict["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target_dict["area"] = (
            (target_dict["boxes"][:, 2] - target_dict["boxes"][:, 0])
            * (target_dict["boxes"][:, 3] - target_dict["boxes"][:, 1])
        )
        target_dict["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)
        return img, target_dict


def infer_num_categories(ann_path: Path) -> int:
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    categories = data.get("categories", [])
    if not categories:
        raise ValueError(f"No categories found in {ann_path}")
    return len(categories)


@torch.no_grad()
def evaluate_torchvision_detector(model, loader):
    model.eval()
    metric = MeanAveragePrecision()
    updates = 0
    for batch in loader:
        if batch is None:
            continue
        images, targets = batch
        images = [img.to(DEVICE) for img in images]
        preds = model(images)

        preds_formatted = [
            dict(
                boxes=p["boxes"].detach().cpu(),
                scores=p["scores"].detach().cpu(),
                labels=p["labels"].detach().cpu(),
            )
            for p in preds
        ]
        targets_formatted = [
            dict(
                boxes=t["boxes"].detach().cpu(),
                labels=t["labels"].detach().cpu(),
            )
            for t in targets
        ]
        metric.update(preds_formatted, targets_formatted)
        updates += 1

    if updates == 0:
        return {"mAP_50": float("nan"), "mAP_50_95": float("nan"), "recall": float("nan")}
    result = metric.compute()
    return {
        "mAP_50": result["map_50"].item(),
        "mAP_50_95": result["map"].item(),
        "recall": result["mar_100"].item(),
    }


def _load_state_dict(checkpoint_path: Path) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        if "model" in state and isinstance(state["model"], dict):
            return state["model"]
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            return state["state_dict"]
    return state


def build_sdg_model(num_classes: int, checkpoint_path: Path):
    model = SDGFasterRCNN(
        num_classes=num_classes,
        use_sam=False,
        use_attention_rpn=True,
        rpn_binarize=False,
        explicit_rpn_thresh=0.7,
    )
    state_dict = _load_state_dict(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[SDG] Missing keys: {missing}")
        print(f"[SDG] Unexpected keys: {unexpected}")
    model.to(DEVICE)
    model.eval()
    return model


def build_rfdetr_model(variant: str, num_classes: int, checkpoint_path: Path):
    try:
        from rfdetr import RFDETRBase, RFDETRMedium, RFDETRSmall
    except ImportError as exc:
        raise RuntimeError("RF-DETR not installed. Install with: pip install rfdetr") from exc

    if variant == "small":
        cls = RFDETRSmall
    elif variant == "medium":
        cls = RFDETRMedium
    else:
        cls = RFDETRBase

    kwargs = {}
    try:
        sig = inspect.signature(cls)
        if "num_classes" in sig.parameters:
            kwargs["num_classes"] = num_classes
    except (TypeError, ValueError):
        pass

    model = cls(**kwargs)
    state_dict = _load_state_dict(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[RF-DETR] Missing keys: {missing}")
        print(f"[RF-DETR] Unexpected keys: {unexpected}")
    if hasattr(model, "to"):
        model.to(DEVICE)
    if hasattr(model, "eval"):
        model.eval()
    return model


def _tensor_to_numpy_image(image_tensor: torch.Tensor) -> np.ndarray:
    img = image_tensor.detach().cpu().clamp(0, 1)
    img = (img * 255.0).byte().permute(1, 2, 0).numpy()
    return img


def _call_rfdetr_predict(model, image_np: np.ndarray):
    if not hasattr(model, "predict"):
        raise RuntimeError("RF-DETR model has no predict() method.")
    predict = model.predict
    kwargs = {}
    try:
        sig = inspect.signature(predict)
        for key in ("threshold", "conf", "confidence", "score_threshold", "score"):
            if key in sig.parameters:
                kwargs[key] = RFDETR_SCORE_THRESHOLD
                break
    except (TypeError, ValueError):
        pass
    return predict(image_np, **kwargs)


def _normalize_rfdetr_output(output, pred_label_offset: int):
    if isinstance(output, list):
        output = output[0] if output else {}

    if isinstance(output, dict):
        boxes = output.get("boxes") or output.get("xyxy")
        scores = output.get("scores") or output.get("confidence")
        labels = output.get("labels") or output.get("class_id") or output.get("class_ids")
    else:
        boxes = getattr(output, "boxes", None) or getattr(output, "xyxy", None)
        scores = getattr(output, "scores", None) or getattr(output, "confidence", None)
        labels = getattr(output, "labels", None) or getattr(output, "class_id", None) or getattr(
            output, "class_ids", None
        )

    if boxes is None or scores is None or labels is None:
        raise ValueError("Unable to parse RF-DETR predictions. Inspect model.predict output.")

    boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
    scores_t = torch.as_tensor(scores, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.int64)

    if boxes_t.numel() == 0:
        boxes_t = boxes_t.reshape(0, 4)
    if scores_t.numel() == 0:
        scores_t = scores_t.reshape(0)
    if labels_t.numel() == 0:
        labels_t = labels_t.reshape(0)

    if pred_label_offset != 0:
        labels_t = labels_t + int(pred_label_offset)

    return {"boxes": boxes_t, "scores": scores_t, "labels": labels_t}


@torch.no_grad()
def evaluate_rfdetr(model, dataset: Dataset, pred_label_offset: int):
    if hasattr(model, "eval"):
        model.eval()
    metric = MeanAveragePrecision()
    updates = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample is None:
            continue
        image_tensor, target = sample

        image_np = _tensor_to_numpy_image(image_tensor)
        output = _call_rfdetr_predict(model, image_np)
        pred = _normalize_rfdetr_output(output, pred_label_offset=pred_label_offset)

        metric.update(
            [dict(boxes=pred["boxes"].cpu(), scores=pred["scores"].cpu(), labels=pred["labels"].cpu())],
            [dict(boxes=target["boxes"].cpu(), labels=target["labels"].cpu())],
        )
        updates += 1

    if updates == 0:
        return {"mAP_50": float("nan"), "mAP_50_95": float("nan"), "recall": float("nan")}
    result = metric.compute()
    return {
        "mAP_50": result["map_50"].item(),
        "mAP_50_95": result["map"].item(),
        "recall": result["mar_100"].item(),
    }


def parse_severity(name: str) -> int | None:
    if "severity_" not in name:
        return None
    try:
        return int(name.split("severity_")[-1])
    except ValueError:
        return None


def find_annotation_example(root: Path) -> Path:
    for subset in sorted(root.iterdir()):
        ann = subset / "clean" / "annotations" / "data.json"
        if ann.exists():
            return ann
    raise FileNotFoundError(f"No annotation file found under {root}")


def main():
    if not BENCHMARK_ROOT.exists():
        raise FileNotFoundError(f"Benchmark root not found: {BENCHMARK_ROOT}")

    subsets = [p for p in BENCHMARK_ROOT.iterdir() if p.is_dir() and "subset_" in p.name]
    if not subsets:
        raise FileNotFoundError(f"No subset folders found under {BENCHMARK_ROOT}")
    subsets = sorted(subsets, key=lambda p: (parse_severity(p.name) or 0, p.name))

    ann_example = find_annotation_example(BENCHMARK_ROOT)
    num_categories = infer_num_categories(ann_example)

    built_models = {}
    for spec in MODEL_SPECS:
        if spec["type"] == "sdg_frcnn":
            built_models[spec["name"]] = build_sdg_model(num_categories + 1, spec["checkpoint"])
        elif spec["type"] == "rfdetr":
            built_models[spec["name"]] = build_rfdetr_model(
                spec.get("variant", "base"), num_categories, spec["checkpoint"]
            )
        else:
            raise ValueError(f"Unknown model type: {spec['type']}")

    results = []
    for subset in subsets:
        subset_name = subset.name
        severity = parse_severity(subset_name)

        for condition in ["clean", "ood"]:
            data_dir = subset / condition
            img_dir = data_dir / "images"
            ann_file = data_dir / "annotations" / "data.json"
            if not img_dir.exists() or not ann_file.exists():
                print(f"Skipping {data_dir} (missing images or annotations)")
                continue

            for spec in MODEL_SPECS:
                print(f"--> Eval: {spec['name']} on {subset_name}/{condition}")
                dataset = COCOWrapper(
                    str(img_dir),
                    str(ann_file),
                    transforms=get_transforms(),
                    resize_to=spec["resize_to"],
                    label_offset=spec["label_offset"],
                )

                if spec["type"] == "sdg_frcnn":
                    loader = DataLoader(
                        dataset,
                        batch_size=spec["batch_size"],
                        shuffle=False,
                        collate_fn=collate_fn,
                        num_workers=0,
                    )
                    metrics = evaluate_torchvision_detector(built_models[spec["name"]], loader)
                else:
                    metrics = evaluate_rfdetr(
                        built_models[spec["name"]],
                        dataset,
                        pred_label_offset=spec["pred_label_offset"],
                    )

                results.append(
                    {
                        "Model": spec["name"],
                        "Subset": subset_name,
                        "Severity": severity if severity is not None else "",
                        "Condition": condition,
                        "mAP_50": metrics["mAP_50"],
                        "mAP_50_95": metrics["mAP_50_95"],
                        "Recall": metrics["recall"],
                    }
                )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Model", "Subset", "Severity", "Condition", "mAP_50", "mAP_50_95", "Recall"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Done. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
