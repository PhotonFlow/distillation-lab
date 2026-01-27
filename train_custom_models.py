import csv
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parent
SUPPORTED_MODELS = ["rfdetr"]


# =====================================
# User-configurable variables (edit here)
# =====================================
MODELS_TO_TRAIN = ["rfdetr"]

TRAIN_IMG = REPO_ROOT / "datasets/custom_benchmark/train/images"
TRAIN_ANN = REPO_ROOT / "datasets/custom_benchmark/train/annotations/train.json"
VAL_IMG = REPO_ROOT / "datasets/custom_benchmark/val_ood/images"
VAL_ANN = REPO_ROOT / "datasets/custom_benchmark/val_ood/annotations/val.json"
OUTPUT_DIR = REPO_ROOT / "checkpoints/custom_training"
RESULTS_CSV = REPO_ROOT / "benchmark_final_results.csv"

EPOCHS = 4
BATCH_SIZE = 4
LR = 1e-3
DEVICE = "cuda"  

# RF-DETR options
RFDETR_VARIANT = "base"  # "base", "small", "medium"
RFDETR_COPY_IMAGES = True  # If symlink fails, set True to copy images
RFDETR_ADV_ENABLED = True
RFDETR_ADV_EPS = 4 / 255
RFDETR_ADV_ALPHA = 2 / 255
RFDETR_ADV_STEPS = 3
RFDETR_ADV_RATIO = 0.5
RFDETR_ADV_RANDOM_START = True
RFDETR_ADV_START_EPOCH = 0

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
        rfdetr_adv_enabled=RFDETR_ADV_ENABLED,
        rfdetr_adv_eps=RFDETR_ADV_EPS,
        rfdetr_adv_alpha=RFDETR_ADV_ALPHA,
        rfdetr_adv_steps=RFDETR_ADV_STEPS,
        rfdetr_adv_ratio=RFDETR_ADV_RATIO,
        rfdetr_adv_random_start=RFDETR_ADV_RANDOM_START,
        rfdetr_adv_start_epoch=RFDETR_ADV_START_EPOCH,
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

    output_dir = Path(args.output_dir) / ("rfdetr_adv" if args.rfdetr_adv_enabled else "rfdetr")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = prepare_rfdetr_dataset(args)

    if args.rfdetr_variant == "small":
        model = RFDETRSmall()
    elif args.rfdetr_variant == "medium":
        model = RFDETRMedium()
    else:
        model = RFDETRBase()

    if args.rfdetr_adv_enabled:
        from rfdetr_adversarial import enable_pgd_training
        enable_pgd_training()

    adv_kwargs = {}
    if args.rfdetr_adv_enabled:
        adv_kwargs = {
            "adv_enabled": True,
            "adv_eps": args.rfdetr_adv_eps,
            "adv_alpha": args.rfdetr_adv_alpha,
            "adv_steps": args.rfdetr_adv_steps,
            "adv_ratio": args.rfdetr_adv_ratio,
            "adv_random_start": args.rfdetr_adv_random_start,
            "adv_start_epoch": args.rfdetr_adv_start_epoch,
        }

    model.train(
        dataset_dir=str(dataset_root),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=str(output_dir),
        device=args.device,
        **adv_kwargs,
    )

    ckpt = find_checkpoint(output_dir, prefer_tokens=["best", "model_best"])
    if ckpt:
        export_state_dict(ckpt, output_dir / "rfdetr_best.pth")

    suffix = " + PGD-3" if args.rfdetr_adv_enabled else ""
    model_label = f"RF-DETR ({args.rfdetr_variant}){suffix}"
    log_path = output_dir / "log.txt"
    results.extend(_collect_rfdetr_results(log_path, model_label))
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

    results: list[dict] = []
    for model_name in selected:
        print(f"\n=== Training: {model_name} ===")
        if model_name == "rfdetr":
            train_rfdetr(settings, results)
        else:
            raise ValueError(f"Unsupported model: {model_name}")


if __name__ == "__main__":
    main()
