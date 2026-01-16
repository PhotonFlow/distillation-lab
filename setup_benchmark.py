
import os
import json
import shutil
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from imagecorruptions import corrupt

# --- Configuration ---
SOURCE_IMAGES_DIR = Path("./datasets/coco5k/images/train") # Where we put val2017 earlier
SOURCE_ANNS = Path("./datasets/coco5k/annotations/instances_val2017.json")
OUTPUT_ROOT = Path("./datasets/custom_benchmark")

TRAIN_SIZE = 4000
VAL_SIZE = 1000 # The rest (approx 1000)
CORRUPTIONS = ['fog', 'snow', 'frost', 'defocus_blur', 'gaussian_noise']
SEVERITY = 3

def filter_coco_json(data, image_ids, new_json_path):
    """Creates a new COCO JSON file containing only the specified image_ids."""
    
    # 1. Filter Images
    new_images = [img for img in data['images'] if img['id'] in image_ids]
    
    # 2. Filter Annotations
    new_anns = [ann for ann in data['annotations'] if ann['image_id'] in image_ids]
    
    # 3. Construct New JSON
    new_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': data['categories'], # Keep all categories
        'images': new_images,
        'annotations': new_anns
    }
    
    with open(new_json_path, 'w') as f:
        json.dump(new_data, f)
    
    return new_images

def main():
    if not SOURCE_ANNS.exists():
        print(f"Error: Could not find {SOURCE_ANNS}. Did you run setup_coco_5k.py?")
        return

    print("--- 1. Loading Original Annotations ---")
    with open(SOURCE_ANNS, 'r') as f:
        coco_data = json.load(f)
    
    all_img_ids = [img['id'] for img in coco_data['images']]
    print(f"Found {len(all_img_ids)} total images.")
    
    # Shuffle and Split
    np.random.seed(42) # Fixed seed for reproducibility
    np.random.shuffle(all_img_ids)
    
    train_ids = set(all_img_ids[:TRAIN_SIZE])
    val_ids = set(all_img_ids[TRAIN_SIZE:])
    
    print(f"Splitting: {len(train_ids)} Train / {len(val_ids)} Val")
    
    # Setup Directories
    train_dir = OUTPUT_ROOT / "train"
    val_dir = OUTPUT_ROOT / "val_ood" # Out of Domain
    
    (train_dir / "images").mkdir(parents=True, exist_ok=True)
    (train_dir / "annotations").mkdir(parents=True, exist_ok=True)
    (val_dir / "images").mkdir(parents=True, exist_ok=True)
    (val_dir / "annotations").mkdir(parents=True, exist_ok=True)

    # --- 2. Process Training Set (Clean) ---
    print("\n--- Processing Training Set (Clean) ---")
    filter_coco_json(coco_data, train_ids, train_dir / "annotations/train.json")
    
    # Copy clean images
    # We iterate through the JSON 'images' list to get filenames
    train_imgs_data = [img for img in coco_data['images'] if img['id'] in train_ids]
    
    for img_info in tqdm(train_imgs_data, desc="Copying Train"):
        src_path = SOURCE_IMAGES_DIR / img_info['file_name']
        dst_path = train_dir / "images" / img_info['file_name']
        if src_path.exists():
            shutil.copy(src_path, dst_path)
            
    # --- 3. Process Validation Set (Corrupted) ---
    print("\n--- Processing Validation Set (Corrupted Target) ---")
    filter_coco_json(coco_data, val_ids, val_dir / "annotations/val.json")
    
    val_imgs_data = [img for img in coco_data['images'] if img['id'] in val_ids]
    
    for img_info in tqdm(val_imgs_data, desc="Corrupting Val"):
        src_path = SOURCE_IMAGES_DIR / img_info['file_name']
        dst_path = val_dir / "images" / img_info['file_name']
        
        if not src_path.exists(): continue
        
        # Read
        img = cv2.imread(str(src_path))
        if img is None: continue
        
        # Apply Corruption
        corruption = np.random.choice(CORRUPTIONS)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug_img = corrupt(img_rgb, corruption_name=corruption, severity=SEVERITY)
        aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
        
        # Save
        cv2.imwrite(str(dst_path), aug_img_bgr)

    print("\n✅ Custom Benchmark Ready at ./datasets/custom_benchmark/")
    print(f"   Train:   {train_dir}")
    print(f"   Val OOD: {val_dir}")

if __name__ == "__main__":
    main()
    # ... rest of the training loop ...

