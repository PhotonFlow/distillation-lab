import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# --- Import Your Custom Modules ---
from models.SDG_FRCNN import SDGFasterRCNN

# --- Configuration ---
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BATCH_SIZE = 4
NUM_EPOCHS = 4
LR = 0.001
FIXED_SIZE = (640, 640) 
CHECKPOINT_DIR="./checkpoints"
os.makedirs(CHECKPOINT_DIR,exist_ok=True)
WARMUP_EPOCHS = 2
LAMBDA_ATT_START = 0.0
LAMBDA_ATT_END = 0.1
LAMBDA_PROT_START = 0.0
LAMBDA_PROT_END = 0.1


# Paths
TRAIN_IMG = "./datasets/custom_benchmark/train/images"
TRAIN_ANN = "./datasets/custom_benchmark/train/annotations/train.json"
VAL_IMG = "./datasets/custom_benchmark/val_ood/images"
VAL_ANN = "./datasets/custom_benchmark/val_ood/annotations/val.json"

def infer_num_classes(ann_path: str) -> int:
    with open(ann_path, "r") as f:
        data = json.load(f)
    categories = data.get("categories", [])
    if not categories:
        raise ValueError(f"No categories found in {ann_path}")
    return len(categories) + 1  # +1 for background class

NUM_CLASSES = infer_num_classes(TRAIN_ANN)

# --- 1. Data Loading Utils ---

def get_transforms():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

def collate_fn(batch):
    batch=[b for b in batch if b is not None]
    if len(batch)==0:
        return None
    return tuple(zip(*batch))

def linear_warmup(epoch, warmup_epochs, start, end):
    if warmup_epochs <= 0:
        return end
    if epoch >= warmup_epochs:
        return end
    return start + (end - start) * float(epoch) / float(warmup_epochs)
class COCOWrapper(Dataset):
    def __init__(self, root, ann, transforms=None):
        self.coco = CocoDetection(root, ann)
        self.transforms = transforms
        
        # Create map from COCO cat_ids to contiguous [1...80]
        cat_ids = sorted(self.coco.coco.getCatIds())
        self.cat_id_to_contiguous_id = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        img, target = self.coco[idx]
        if self.transforms:
            img = self.transforms(img) # Returns (C, H, W) Tensor

        # 1. Resize Image to Fixed Size
        original_h, original_w = img.shape[-2:]
        img = F.interpolate(img.unsqueeze(0), size=FIXED_SIZE, mode='bilinear', align_corners=False).squeeze(0)
        
        # Calculate scale factors
        scale_x = FIXED_SIZE[1] / original_w
        scale_y = FIXED_SIZE[0] / original_h

        boxes = []
        labels = []
        
        for obj in target:
            x, y, w, h = obj['bbox']
            
            # Convert to [x1, y1, x2, y2] and Scale
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            
            # 2. Logic Fix: Check EVERYTHING before appending ANYTHING
            # Ensure box is valid size AND category is known
            if (x2 - x1) > 1 and (y2 - y1) > 1:
                cat_id = obj['category_id']
                if cat_id in self.cat_id_to_contiguous_id:
                    # Append BOTH together to ensure sync
                    boxes.append([x1, y1, x2, y2])
                    labels.append(self.cat_id_to_contiguous_id[cat_id])

        # 3. Filter empty images
        if len(boxes) == 0:
            return None

        target_dict = {}
        target_dict["image_id"] = torch.tensor([idx])
        target_dict['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target_dict['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        
        # Approximate area for COCO eval compatibility
        target_dict["area"] = (target_dict['boxes'][:, 2] - target_dict['boxes'][:, 0]) * \
                              (target_dict['boxes'][:, 3] - target_dict['boxes'][:, 1])
        target_dict["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)

        return img, target_dict

# --- 2. Evaluation Engine ---

@torch.no_grad()
def evaluate_map(model, loader):
    model.eval()
    metric = MeanAveragePrecision()
    
    for images, targets in tqdm(loader, desc="  Evaluating", leave=False):
        images = list(img.to(DEVICE) for img in images)
        
        preds = model(images)
        
        # Format for torchmetrics
        preds_formatted = [
            dict(boxes=p['boxes'].cpu(), scores=p['scores'].cpu(), labels=p['labels'].cpu())
            for p in preds
        ]
        
        targets_formatted = [
            dict(boxes=t['boxes'].cpu(), labels=t['labels'].cpu())
            for t in targets
        ]
        
        metric.update(preds_formatted, targets_formatted)
        
    result = metric.compute()
    return {"mAP_50":result['map_50'].item(),
            "mAP_50_95":result['map'].item(),
            "recall":result['mar_100'].item()}

# --- 3. Training Functions ---

def train_epoch_baseline(model, optimizer, loader):
    model.train()
    total_loss = 0
    for images, targets in tqdm(loader, desc="  Train Baseline", leave=False):
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    return total_loss / len(loader)

def train_epoch_sdg(model, optimizer, loader):
    model.train()
    total_loss = 0
    prot_loss_sum = 0
    att_loss_sum = 0
    prot_exp_sum = 0
    prot_imp_sum = 0
    exp_empty_batches = 0
    exp_keep_rate_sum = 0.0
    exp_keep_rate_count = 0
    imp_no_common_batches = 0
    
    for images, targets in tqdm(loader, desc="  Train SDG", leave=False):
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        # SDGFasterRCNN returns a dict with component losses
        out = model(images, targets)
        loss = out['total_loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        prot_loss_sum += out['prot_loss'].item()
        att_loss_sum += out['att_loss'].item()
        prot_exp_sum += out.get('prot_exp', torch.tensor(0.0)).item()
        prot_imp_sum += out.get('prot_imp', torch.tensor(0.0)).item()
        exp_empty_batches += int(out.get('exp_empty', 0))
        imp_no_common_batches += int(out.get('imp_no_common', 0))
        exp_keep_rate = out.get('exp_keep_rate', None)
        if exp_keep_rate is not None:
            exp_keep_rate_sum += float(exp_keep_rate)
            exp_keep_rate_count += 1
        
    avg_loss = total_loss / len(loader)
    avg_prot_exp = prot_exp_sum / len(loader)
    avg_prot_imp = prot_imp_sum / len(loader)
    exp_keep_rate_avg = exp_keep_rate_sum / exp_keep_rate_count if exp_keep_rate_count > 0 else 0.0
    print(
        "    [Details] "
        f"Total: {avg_loss:.3f} | Prot: {prot_loss_sum/len(loader):.3f} | "
        f"Att: {att_loss_sum/len(loader):.3f} | "
        f"ProtExp: {avg_prot_exp:.3f} | ProtImp: {avg_prot_imp:.3f} | "
        f"ExpEmpty: {exp_empty_batches}/{len(loader)} | "
        f"ImpNoCommon: {imp_no_common_batches}/{len(loader)} | "
        f"KeepRate: {exp_keep_rate_avg:.3f}"
    )
    return avg_loss

# --- 4. Main Runner ---

def run_benchmark():
    print(f"Loading data from {TRAIN_IMG}...")
    train_set = COCOWrapper(TRAIN_IMG, TRAIN_ANN, get_transforms())
    val_set = COCOWrapper(VAL_IMG, VAL_ANN, get_transforms())
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    results = []

    # ==========================
    # 1. BASELINE A: Standard R-CNN
    # ==========================
    print("\n" + "="*40)
    print("STARTING BASELINE A: Standard Faster R-CNN")
    print("="*40)
    
    model_a = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model_a.roi_heads.box_predictor.cls_score.in_features
    model_a.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model_a.to(DEVICE)
    
    opt_a = optim.SGD(model_a.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)
    scheduler_a=torch.optim.lr_scheduler.MultiStepLR(opt_a, milestones=[3,4],gamma=0.1)

    best_map_baseline=0.0

    # for epoch in range(NUM_EPOCHS):
    #     loss = train_epoch_baseline(model_a, opt_a, train_loader)
    #     metrics= evaluate_map(model_a,val_loader)
    #     current_map_baseline=metrics["mAP_50"]
    #     torch.save(model_a.state_dict(),os.path.join(CHECKPOINT_DIR,'baseline_frcnn_last.pth'))
    #     if current_map_baseline >best_map_baseline:
    #         best_map_baseline=current_map_baseline
    #         torch.save(model_a.state_dict(),os.path.join(CHECKPOINT_DIR,'baseline_frcnn_best.pth'))
    #     scheduler_a.step()
    #     print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {loss:.4f} | Val mAP@50: {metrics['mAP_50']}")
    #     results.append({"Model": "Baseline_A", 
    #                     "Epoch": epoch+1,
    #                      "Loss": loss,
    #                      "mAP_50": metrics['mAP_50'],
    #                      "mAP_50_95":metrics['mAP_50_95'],
    #                      "Recall":metrics['recall']
    #     })
    #     pd.DataFrame(results).to_csv("benchmark_final_results.csv",index=False)

    # ==========================
    # 2. METHOD: SDG R-CNN (Ours)
    # ==========================
    print("\n" + "="*40)
    print("STARTING METHOD: SDG Faster R-CNN (With SAM)")
    print("="*40)
    
    model_sdg = SDGFasterRCNN(
        num_classes=NUM_CLASSES, 
        use_sam=True, 
        sam_checkpoint="sam_vit_h_4b8939.pth",
        use_attention_rpn=True,
        rpn_binarize=False,
        explicit_rpn_thresh=0.7
    )
    model_sdg.to(DEVICE)
    
    opt_sdg = optim.SGD(model_sdg.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)
    scheduler_sdg=torch.optim.lr_scheduler.MultiStepLR(opt_sdg,milestones=[3,4],gamma=0.1)

    best_map_unbiased=0.0

    for epoch in range(NUM_EPOCHS):
        model_sdg.lambda_att = linear_warmup(epoch, WARMUP_EPOCHS, LAMBDA_ATT_START, LAMBDA_ATT_END)
        model_sdg.lambda_prot = linear_warmup(epoch, WARMUP_EPOCHS, LAMBDA_PROT_START, LAMBDA_PROT_END)
        loss = train_epoch_sdg(model_sdg, opt_sdg, train_loader)
        metrics=evaluate_map(model_sdg,val_loader)
        current_map_unbiased=metrics["mAP_50"]
        torch.save(model_sdg.state_dict(),os.path.join(CHECKPOINT_DIR,"unbiased_frcnn_last.pth"))
        if current_map_unbiased>best_map_unbiased:
            best_map_unbiased=current_map_unbiased
            torch.save(model_sdg.state_dict(),os.path.join(CHECKPOINT_DIR,"unbiased_frcnn_best.pth"))
            
        scheduler_sdg.step()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {loss:.4f} | Val mAP@50: {metrics['mAP_50']}")
        results.append({"Model": "Unbiased_RCNN",
                        "Epoch": epoch+1,
                         "Loss": loss,
                         "mAP_50": metrics['mAP_50'],
                         "mAP_50_95":metrics['mAP_50_95'],
                         "Recall":metrics['recall']
        })
        pd.DataFrame(results).to_csv("benchmark_final_results.csv",index=False)
    print("\nBenchmark Results Saved!!!!!!!!!!!!!!")

    

if __name__ == "__main__":
    run_benchmark()