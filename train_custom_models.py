import csv
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parent
SUPPORTED_MODELS = ["rfdetr", "yolov8"]


# =====================================
# User-configurable variables (edit here)
# =====================================
MODELS_TO_TRAIN = ["rfdetr", "yolov8"]

TRAIN_IMG = REPO_ROOT / "datasets/custom_benchmark/train/images"
TRAIN_ANN = REPO_ROOT / "datasets/custom_benchmark/train/annotations/train.json"
VAL_IMG = REPO_ROOT / "datasets/custom_benchmark/val_ood/images"
VAL_ANN = REPO_ROOT / "datasets/custom_benchmark/val_ood/annotations/val.json"
OUTPUT_DIR = REPO_ROOT / "checkpoints/custom_training"
RESULTS_CSV = REPO_ROOT / "benchmark_final_results.csv"

EPOCHS = 4
BATCH_SIZE = 4
LR = 1e-3
DEVICE = "cuda"  # Set to "cuda", "cpu", or None for auto-select

# RF-DETR options
RFDETR_VARIANT = "base"  # "base", "small", "medium"
RFDETR_COPY_IMAGES = True  # If symlink fails, set True to copy images

# YOLOv8 options
YOLOV8_MODEL = "yolov8n.pt"
YOLOV8_IMGSZ = 640


def get_settings() -> SimpleNamespace:
    return SimpleNamespace(
        models=MODELS_TO_TRAIN,
        train_img=str(TRAIN_IMG),
        train_ann=str(TRAIN_ANN),
        val_img=str(VAL_IMG),
        val_ann=str(VAL_ANN),
        output_dir=str(OUTPUT_DIR),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        device=DEVICE,
        rfdetr_variant=RFDETR_VARIANT,
        rfdetr_copy_images=RFDETR_COPY_IMAGES,
        yolov8_model=YOLOV8_MODEL,
        yolov8_imgsz=YOLOV8_IMGSZ,
    )


def resolve_models(value) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        models = [str(m).strip().lower() for m in value if str(m).strip()]
        if not models:
            raise ValueError("MODELS_TO_TRAIN is empty.")
        if "all" in models:
            return SUPPORTED_MODELS
    else:
        text = str(value).strip().lower()
        if text == "all":
            return SUPPORTED_MODELS
        models = [m.strip().lower() for m in text.split(",") if m.strip()]
    unknown = sorted(set(models) - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(unknown)}. Choose from: {SUPPORTED_MODELS}")
    return models


def ensure_exists(path: Path, kind: str) -> None:
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")


def load_coco_categories(ann_path: Path) -> tuple[list[str], dict[int, int]]:
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    categories = sorted(data.get("categories", []), key=lambda c: c["id"])
    if not categories:
        raise ValueError(f"No categories found in {ann_path}")
    class_names = [c["name"] for c in categories]
    id_to_index = {c["id"]: idx for idx, c in enumerate(categories)}
    return class_names, id_to_index


def infer_dataset_root(train_img: Path) -> Path:
    if train_img.name == "images" and train_img.parent.name == "train":
        return train_img.parent.parent
    return train_img.parent


def _dump_yaml(data: dict, path: Path) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to edit config files. Install with: pip install pyyaml") from exc
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _safe_index(values: list, idx: int) -> float | None:
    if not isinstance(values, list):
        return None
    if idx < 0 or idx >= len(values):
        return None
    try:
        return float(values[idx])
    except (TypeError, ValueError):
        return None


def _write_results_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Model", "Epoch", "Loss", "mAP_50", "mAP_50_95", "Recall"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def _collect_rfdetr_results(log_path: Path, model_name: str) -> list[dict]:
    if not log_path.exists():
        return []
    results: list[dict] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            epoch = entry.get("epoch")
            if epoch is None:
                continue
            loss = entry.get("train_loss")
            if loss is None:
                loss = entry.get("train_loss_unscaled")
            test_eval = entry.get("test_coco_eval_bbox")
            results.append({
                "Model": model_name,
                "Epoch": int(epoch) + 1,
                "Loss": loss,
                "mAP_50": _safe_index(test_eval, 1),
                "mAP_50_95": _safe_index(test_eval, 0),
                "Recall": _safe_index(test_eval, 8),
            })
    return results


def _collect_yolov8_results(results_csv: Path, model_name: str) -> list[dict]:
    if not results_csv.exists():
        return []
    results: list[dict] = []
    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = row.get("epoch")
            if epoch is None or epoch == "":
                continue
            try:
                epoch_val = int(float(epoch)) + 1
            except ValueError:
                continue
            def _get_float(key: str) -> float | None:
                value = row.get(key)
                if value is None or value == "":
                    return None
                try:
                    return float(value)
                except ValueError:
                    return None
            train_box = _get_float("train/box_loss")
            train_cls = _get_float("train/cls_loss")
            train_dfl = _get_float("train/dfl_loss")
            loss = None
            if train_box is not None and train_cls is not None and train_dfl is not None:
                loss = train_box + train_cls + train_dfl
            results.append({
                "Model": model_name,
                "Epoch": epoch_val,
                "Loss": loss,
                "mAP_50": _get_float("metrics/mAP50(B)"),
                "mAP_50_95": _get_float("metrics/mAP50-95(B)"),
                "Recall": _get_float("metrics/recall(B)"),
            })
    return results


def find_checkpoint(root: Path, prefer_tokens: list[str]) -> Path | None:
    if not root.exists():
        return None
    candidates = list(root.rglob("*.pth")) + list(root.rglob("*.pt"))
    if not candidates:
        return None
    preferred = [
        p for p in candidates
        if any(token in p.name.lower() for token in prefer_tokens)
    ]
    picks = preferred or candidates
    picks.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return picks[0]


def export_state_dict(weights_path: Path, output_path: Path) -> None:
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict):
        if "model" in state and hasattr(state["model"], "state_dict"):
            state_dict = state["model"].state_dict()
        elif "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state
    elif hasattr(state, "state_dict"):
        state_dict = state.state_dict()
    else:
        state_dict = state
    torch.save(state_dict, output_path)


def _populate_images(src_dir: Path, dst_dir: Path, copy_images: bool) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img_path in src_dir.iterdir():
        if not img_path.is_file():
            continue
        dst_path = dst_dir / img_path.name
        if dst_path.exists():
            continue
        if copy_images:
            shutil.copy2(img_path, dst_path)
        else:
            try:
                os.symlink(img_path, dst_path)
            except OSError as exc:
                raise RuntimeError(
                    "RF-DETR dataset prep could not create a symlink for images. "
                    "Set RFDETR_COPY_IMAGES = True to copy images instead."
                ) from exc


def _load_coco_category_mapping(ann_path: Path) -> tuple[dict[int, int], list[dict]]:
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    categories = data.get("categories", [])
    if not categories:
        raise ValueError(f"No categories found in {ann_path}")

    id_map: dict[int, int] = {}
    new_categories: list[dict] = []
    for new_id, cat in enumerate(categories):
        old_id = cat.get("id")
        if old_id is None:
            raise ValueError(f"Category missing 'id' in {ann_path}")
        if old_id in id_map:
            raise ValueError(f"Duplicate category id {old_id} in {ann_path}")
        id_map[int(old_id)] = new_id
        new_cat = dict(cat)
        new_cat["id"] = new_id
        new_categories.append(new_cat)
    return id_map, new_categories


def _rewrite_coco_annotations(
    src_ann: Path,
    dst_ann: Path,
    id_map: dict[int, int],
    categories: list[dict],
) -> None:
    with src_ann.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["categories"] = categories
    for ann in data.get("annotations", []):
        old_id = ann.get("category_id")
        if old_id is None:
            raise ValueError(f"Annotation missing 'category_id' in {src_ann}")
        if int(old_id) not in id_map:
            raise ValueError(f"Unknown category_id {old_id} in {src_ann}")
        ann["category_id"] = id_map[int(old_id)]
    dst_ann.parent.mkdir(parents=True, exist_ok=True)
    with dst_ann.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def prepare_rfdetr_dataset(args) -> Path:
    dataset_root = Path(args.output_dir) / "rfdetr_dataset"
    train_dir = dataset_root / "train"
    valid_dir = dataset_root / "valid"
    test_dir = dataset_root / "test"
    for d in [train_dir, valid_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_img = Path(args.train_img)
    val_img = Path(args.val_img)

    _populate_images(train_img, train_dir, args.rfdetr_copy_images)
    _populate_images(val_img, valid_dir, args.rfdetr_copy_images)
    _populate_images(val_img, test_dir, args.rfdetr_copy_images)

    id_map, categories = _load_coco_category_mapping(Path(args.train_ann))
    _rewrite_coco_annotations(Path(args.train_ann), train_dir / "_annotations.coco.json", id_map, categories)
    _rewrite_coco_annotations(Path(args.val_ann), valid_dir / "_annotations.coco.json", id_map, categories)
    _rewrite_coco_annotations(Path(args.val_ann), test_dir / "_annotations.coco.json", id_map, categories)
    return dataset_root


def train_rfdetr(args, results: list[dict]) -> None:
    try:
        from rfdetr import RFDETRBase, RFDETRMedium, RFDETRSmall
    except ImportError as exc:
        raise RuntimeError("RF-DETR not installed. Install with: pip install rfdetr") from exc

    output_dir = Path(args.output_dir) / "rfdetr"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = prepare_rfdetr_dataset(args)

    if args.rfdetr_variant == "small":
        model = RFDETRSmall()
    elif args.rfdetr_variant == "medium":
        model = RFDETRMedium()
    else:
        model = RFDETRBase()

    model.train(
        dataset_dir=str(dataset_root),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=str(output_dir),
        device=args.device,
    )

    ckpt = find_checkpoint(output_dir, prefer_tokens=["best", "model_best"])
    if ckpt:
        export_state_dict(ckpt, output_dir / "rfdetr_best.pth")

    model_label = f"RF-DETR ({args.rfdetr_variant})"
    log_path = output_dir / "log.txt"
    results.extend(_collect_rfdetr_results(log_path, model_label))
    _write_results_csv(RESULTS_CSV, results)


def convert_coco_to_yolo(
    ann_path: Path,
    images_dir: Path,
    labels_dir: Path,
    class_id_map: dict[int, int],
) -> None:
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data.get("images", [])}
    annotations = data.get("annotations", [])
    labels_dir.mkdir(parents=True, exist_ok=True)

    label_cache: dict[str, list[str]] = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in images:
            continue
        img_info = images[img_id]
        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]
        category_id = ann["category_id"]
        if category_id not in class_id_map:
            continue

        x, y, w, h = ann["bbox"]
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        w_norm = w / width
        h_norm = h / height
        cls = class_id_map[category_id]

        label_line = f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
        label_cache.setdefault(file_name, []).append(label_line)

    for img in images.values():
        label_path = labels_dir / (Path(img["file_name"]).stem + ".txt")
        lines = label_cache.get(img["file_name"], [])
        with label_path.open("w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines))

    # sanity check for missing images
    for img_name in label_cache:
        if not (images_dir / img_name).exists():
            print(f"[yolov8] Warning: missing image file for label: {img_name}")


def write_yolo_data_yaml(
    dataset_root: Path,
    train_rel: str,
    val_rel: str,
    names: list[str],
    output_path: Path,
) -> None:
    data = {
        "path": str(dataset_root),
        "train": train_rel,
        "val": val_rel,
        "names": names,
    }
    _dump_yaml(data, output_path)


def train_yolov8(args, class_names: list[str], class_id_map: dict[int, int], results: list[dict]) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("YOLOv8 not installed. Install with: pip install ultralytics") from exc

    train_img = Path(args.train_img)
    val_img = Path(args.val_img)

    train_labels = train_img.parent / "labels"
    val_labels = val_img.parent / "labels"
    convert_coco_to_yolo(Path(args.train_ann), train_img, train_labels, class_id_map)
    convert_coco_to_yolo(Path(args.val_ann), val_img, val_labels, class_id_map)

    dataset_root = infer_dataset_root(train_img)
    data_yaml = Path(args.output_dir) / "yolov8_data.yaml"
    write_yolo_data_yaml(
        dataset_root=dataset_root,
        train_rel=str(train_img.relative_to(dataset_root)),
        val_rel=str(val_img.relative_to(dataset_root)),
        names=class_names,
        output_path=data_yaml,
    )

    output_dir = Path(args.output_dir) / "yolov8"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.yolov8_model)
    train_result = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.yolov8_imgsz,
        batch=args.batch_size,
        device=args.device,
        project=str(Path(args.output_dir)),
        name="yolov8",
    )

    save_dir = getattr(train_result, "save_dir", None)
    if save_dir is None and hasattr(model, "trainer"):
        save_dir = getattr(model.trainer, "save_dir", None)
    if save_dir is None:
        save_dir = output_dir
    save_dir = Path(save_dir)
    best_pt = save_dir / "weights/best.pt"
    if best_pt.exists():
        export_state_dict(best_pt, output_dir / "yolov8_best.pth")

    results_csv = save_dir / "results.csv"
    model_label = f"YOLOv8 ({args.yolov8_model})"
    results.extend(_collect_yolov8_results(results_csv, model_label))
    _write_results_csv(RESULTS_CSV, results)


def main() -> None:
    settings = get_settings()
    if settings.device is None:
        settings.device = "cuda" if torch.cuda.is_available() else "cpu"

    settings.train_img = str(Path(settings.train_img))
    settings.train_ann = str(Path(settings.train_ann))
    settings.val_img = str(Path(settings.val_img))
    settings.val_ann = str(Path(settings.val_ann))

    selected = resolve_models(settings.models)
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names, class_id_map = load_coco_categories(Path(settings.train_ann))

    results: list[dict] = []
    for model_name in selected:
        print(f"\n=== Training: {model_name} ===")
        if model_name == "rfdetr":
            train_rfdetr(settings, results)
        elif model_name == "yolov8":
            train_yolov8(settings, class_names, class_id_map, results)
        else:
            raise ValueError(f"Unsupported model: {model_name}")


if __name__ == "__main__":
    main()
