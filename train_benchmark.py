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
from tqdm import tqdm
import time
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch.nn.functional as F

# --- Import Your Modules ---
from models.SDG_FRCNN import SDG_FRCNN
from models.feature_adapter import sdg_total_loss

# --- Configuration ---
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_CLASSES = 81  # 80 COCO classes + 1 Background
BATCH_SIZE = 4
NUM_EPOCHS = 2
LR = 0.005
FIXED_SIZE=(640,640)


# Paths (From your previous data setup)
TRAIN_IMG = "./datasets/custom_benchmark/train/images"
TRAIN_ANN = "./datasets/custom_benchmark/train/annotations/train.json"
VAL_IMG = "./datasets/custom_benchmark/val_ood/images"
VAL_ANN = "./datasets/custom_benchmark/val_ood/annotations/val.json"

# --- 1. Data Loading ---

def get_transforms(train=True):
    # Simple ToTensor. You can add more data aug for Baseline A if you want.
    # GLT handles aug for SDG internally.
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

class COCOWrapper(Dataset):
    """Wrapper to ensure dataset returns (img, target) compatible with R-CNN"""
    def __init__(self, root, ann, transforms=None):
        self.coco = CocoDetection(root, ann)
        self.transforms = transforms
        # Map non-contiguous COCO category ids (e.g., max id 90) to [1..N]
        cat_ids = sorted(self.coco.coco.getCatIds())
        self.cat_id_to_contiguous_id = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        img, target = self.coco[idx]
        
        # Transform image
        if self.transforms:
            img = self.transforms(img)


        original_h, original_w = img.shape[-2:]
        img=F.interpolate(img.unsqueeze(0), size=FIXED_SIZE, mode='bilinear', align_corners=False).squeeze(0)
        scale_x = FIXED_SIZE[1] / original_w
        scale_y = FIXED_SIZE[0] / original_h


        # Reformat Target for Faster R-CNN
        boxes = []
        labels = []
        
        for obj in target:
            bbox = obj['bbox'] # [x, y, w, h]
            # Convert to [x1, y1, x2, y2]
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h

            x1*=scale_x
            y1*=scale_y
            x2*=scale_x
            y2*=scale_y
            
            # Filter tiny/invalid boxes
            if w > 1 and h > 1:
                boxes.append([x1, y1, x2, y2])
                cat_id = obj['category_id']
                if cat_id not in self.cat_id_to_contiguous_id:
                    raise ValueError(f"Unknown category_id {cat_id} in annotations.")
                labels.append(self.cat_id_to_contiguous_id[cat_id])

        target_dict = {}
        target_dict["image_id"] = torch.tensor([idx])
        
        if len(boxes) > 0:
            target_dict['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target_dict['labels'] = torch.as_tensor(labels, dtype=torch.int64)
            target_dict["area"] = torch.tensor([obj['area'] for obj in target]) 
            target_dict["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)
        else:
            # Handle images with no objects
            target_dict['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target_dict['labels'] = torch.zeros((0,), dtype=torch.int64)
            target_dict["area"] = torch.tensor([obj['area'] for obj in target]) 
            target_dict["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)
            
        return img, target_dict

def collate_fn(batch):
    return tuple(zip(*batch))

# --- 2. Evaluation Helper ---

@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()
    metric = MeanAveragePrecision()
    
    print("Evaluating...")
    for images, targets in tqdm(data_loader, desc="Eval"):
        images = list(img.to(DEVICE) for img in images)
        
        # Targets need to be on device for Metric? usually not required but safer
        # Metric expects:
        # preds: list of dicts {boxes, scores, labels}
        # target: list of dicts {boxes, labels}
        
        preds = model(images)
        
        # Prepare for torchmetrics
        formatted_preds = []
        for p in preds:
            formatted_preds.append({
                "boxes": p["boxes"].cpu(),
                "scores": p["scores"].cpu(),
                "labels": p["labels"].cpu()
            })
            
        formatted_targets = []
        for t in targets:
            formatted_targets.append({
                "boxes": t["boxes"].cpu(),
                "labels": t["labels"].cpu()
            })
            
        metric.update(formatted_preds, formatted_targets)
        
    result = metric.compute()
    # Extract scalar map_50
    map_50 = result['map_50'].item()
    return map_50

# --- 3. Training Loops ---

def train_baseline(model, optimizer, loader):
    model.train()
    total_loss = 0
    for images, targets in tqdm(loader, desc="Train Baseline"):
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
        
    return total_loss / len(loader)

def train_sdg(model, optimizer, loader):
    model.train()
    total_loss = 0
    for images, targets in tqdm(loader, desc="Train SDG"):
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        # The SDG model returns a dict with 'total_loss', 'sup_loss', etc.
        out_dict = model(images, targets)
        loss = out_dict['total_loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

# --- 4. Main Experiment Runner ---

def run_benchmark():
    # Setup Data
    train_set = COCOWrapper(TRAIN_IMG, TRAIN_ANN, get_transforms())
    val_set = COCOWrapper(VAL_IMG, VAL_ANN, get_transforms())
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    results = []

    # ==========================================
    # Experiment A: Baseline (Standard R-CNN)
    # ==========================================
    print("\n" + "="*30)
    print("Running Baseline A: Standard Faster R-CNN")
    print("="*30)
    
    model_a = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model_a.roi_heads.box_predictor.cls_score.in_features
    model_a.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model_a.to(DEVICE)
    
    opt_a = optim.SGD(model_a.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)

    # for epoch in range(NUM_EPOCHS):
    #     loss = train_baseline(model_a, opt_a, train_loader)
    #     map_50 = evaluate(model_a, val_loader)
        
    #     print(f"Epoch {epoch+1}: Loss={loss:.4f}, mAP@50={map_50:.4f}")
    #     results.append({
    #         "Model": "Baseline_A",
    #         "Epoch": epoch + 1,
    #         "Train_Loss": loss,
    #         "Val_mAP_50": map_50
    #     })

    # ==========================================
    # Experiment B: SDG Method (Your Implementation)
    # ==========================================
    print("\n" + "="*30)
    print("Running Method: SDG Faster R-CNN (Ours)")
    print("="*30)
    
    # Initialize with SAM enabled
    model_sdg = SDG_FRCNN(NUM_CLASSES, use_sam=True, sam_checkpoint="sam_vit_h_4b8939.pth")
    model_sdg.to(DEVICE)
    
    # Note: SDG usually requires lower LR or careful tuning
    opt_sdg = optim.SGD(model_sdg.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)

    for epoch in range(NUM_EPOCHS):
        loss = train_sdg(model_sdg, opt_sdg, train_loader)
        map_50 = evaluate(model_sdg, val_loader)
        
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, mAP@50={map_50:.4f}")
        results.append({
            "Model": "SDG_RCNN",
            "Epoch": epoch + 1,
            "Train_Loss": loss,
            "Val_mAP_50": map_50
        })

    # Save Results
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results_final.csv", index=False)
    print("\nBenchmark Complete. Saved to benchmark_results_final.csv")

if __name__ == "__main__":
    run_benchmark()