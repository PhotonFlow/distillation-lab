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

EPOCHS = 4
BATCH_SIZE = 4
LR = 1e-3
DEVICE = "cuda"  # Set to "cuda", "cpu", or None for auto-select

# RF-DETR options
RFDETR_VARIANT = "base"  # "base", "small", "medium"
RFDETR_COPY_IMAGES = False  # If symlink fails, set True to copy images

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


def _try_symlink(src: Path, dst: Path) -> bool:
    try:
        if dst.exists():
            return True
        os.symlink(src, dst, target_is_directory=True)
        return True
    except OSError:
        return False


def _copy_images(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img_path in src_dir.iterdir():
        if img_path.is_file():
            shutil.copy2(img_path, dst_dir / img_path.name)


def prepare_rfdetr_dataset(args) -> Path:
    dataset_root = Path(args.output_dir) / "rfdetr_dataset"
    train_dir = dataset_root / "train"
    valid_dir = dataset_root / "valid"
    test_dir = dataset_root / "test"
    for d in [train_dir, valid_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_img = Path(args.train_img)
    val_img = Path(args.val_img)

    if not _try_symlink(train_img, train_dir / "images"):
        if args.rfdetr_copy_images:
            _copy_images(train_img, train_dir / "images")
        else:
            raise RuntimeError(
                "RF-DETR dataset prep could not create a symlink for train images. "
                "Set RFDETR_COPY_IMAGES = True to copy images instead."
            )
    if not _try_symlink(val_img, valid_dir / "images"):
        if args.rfdetr_copy_images:
            _copy_images(val_img, valid_dir / "images")
        else:
            raise RuntimeError(
                "RF-DETR dataset prep could not create a symlink for val images. "
                "Set RFDETR_COPY_IMAGES = True to copy images instead."
            )
    if not _try_symlink(val_img, test_dir / "images"):
        if args.rfdetr_copy_images:
            _copy_images(val_img, test_dir / "images")
        else:
            raise RuntimeError(
                "RF-DETR dataset prep could not create a symlink for test images. "
                "Set RFDETR_COPY_IMAGES = True to copy images instead."
            )

    shutil.copy2(Path(args.train_ann), train_dir / "_annotations.coco.json")
    shutil.copy2(Path(args.val_ann), valid_dir / "_annotations.coco.json")
    shutil.copy2(Path(args.val_ann), test_dir / "_annotations.coco.json")
    return dataset_root


def train_rfdetr(args) -> None:
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


def train_yolov8(args, class_names: list[str], class_id_map: dict[int, int]) -> None:
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

    for model_name in selected:
        print(f"\n=== Training: {model_name} ===")
        if model_name == "rfdetr":
            train_rfdetr(settings)
        elif model_name == "yolov8":
            train_yolov8(settings, class_names, class_id_map)
        else:
            raise ValueError(f"Unsupported model: {model_name}")


if __name__ == "__main__":
    main()
