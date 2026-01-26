import os
import json
import shutil
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from imagecorruptions import corrupt

# --- Configuration ---
SOURCE_IMAGES_DIR = Path("./datasets/coco5k/images/train") 
SOURCE_ANNS = Path("./datasets/coco5k/annotations/instances_val2017.json")
OUTPUT_ROOT = Path("./datasets/custom_benchmark_splits")

NUM_SPLITS = 5
SPLIT_SIZE = 1000
CORRUPTIONS = ['fog', 'snow', 'frost', 'defocus_blur', 'gaussian_noise']

def filter_coco_json(data, image_ids, new_json_path):
    """Creates a new COCO JSON file containing only the specified image_ids."""
    
    new_images = [img for img in data['images'] if img['id'] in image_ids]
    
    # 2. Filter Annotations
    new_anns = [ann for ann in data['annotations'] if ann['image_id'] in image_ids]
    
    # 3. Construct New JSON
    new_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': data['categories'], 
        'images': new_images,
        'annotations': new_anns
    }
    
    with open(new_json_path, 'w') as f:
        json.dump(new_data, f)
    
    return new_images

def main():
    if not SOURCE_ANNS.exists():
        print(f"Error: Could not find {SOURCE_ANNS}.")
        return

    print("--- 1. Loading Original Annotations ---")
    with open(SOURCE_ANNS, 'r') as f:
        coco_data = json.load(f)
    
    all_img_ids = np.array([img['id'] for img in coco_data['images']])
    print(f"Found {len(all_img_ids)} total images.")
    
    # Shuffle
    np.random.seed(42) 
    np.random.shuffle(all_img_ids)
    
    # Verify we have enough images
    if len(all_img_ids) < NUM_SPLITS * SPLIT_SIZE:
        print("Error: Not enough images for 5 splits of 1000.")
        return

    # Split into 5 chunks of 1000
    # array_split ensures we get 5 chunks
    splits = np.array_split(all_img_ids[:NUM_SPLITS * SPLIT_SIZE], NUM_SPLITS)

    print(f"\nGeneratng {NUM_SPLITS} subsets with sliding severity...")

    # --- 2. Process Each Split ---
    for i, subset_ids in enumerate(splits):
        severity = i + 1 # Severity 1 to 5
        subset_name = f"subset_{severity}_severity_{severity}"
        
        subset_ids_set = set(subset_ids) # Set for faster lookup
        
        # Define paths
        subset_root = OUTPUT_ROOT / subset_name
        
        clean_dir = subset_root / "clean"
        ood_dir = subset_root / "ood"
        
        # Create directories
        for d in [clean_dir, ood_dir]:
            (d / "images").mkdir(parents=True, exist_ok=True)
            (d / "annotations").mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {subset_name} (Severity: {severity})...")

        # 1. Create JSONs (Same annotations for Clean and OOD, just different images)
        filter_coco_json(coco_data, subset_ids_set, clean_dir / "annotations/data.json")
        filter_coco_json(coco_data, subset_ids_set, ood_dir / "annotations/data.json")

        # 2. Process Images
        subset_imgs_data = [img for img in coco_data['images'] if img['id'] in subset_ids_set]

        for img_info in tqdm(subset_imgs_data, desc=f"  Gen Images (Sev {severity})"):
            src_path = SOURCE_IMAGES_DIR / img_info['file_name']
            
            if not src_path.exists(): continue
            
            # --- A. Save Clean ---
            dst_clean = clean_dir / "images" / img_info['file_name']
            shutil.copy(src_path, dst_clean)

            # --- B. Save OOD (Corrupted) ---
            dst_ood = ood_dir / "images" / img_info['file_name']
            
            # Read Image
            img = cv2.imread(str(src_path))
            if img is None: continue
            
            # Apply Corruption
            corruption = np.random.choice(CORRUPTIONS)
            
            # Convert to RGB for imagecorruptions lib, then back to BGR for OpenCV
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                aug_img = corrupt(img_rgb, corruption_name=corruption, severity=severity)
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(dst_ood), aug_img_bgr)
            except Exception as e:
                print(f"Failed to corrupt {src_path}: {e}")
                # Fallback: copy original if corruption fails? 
                # Or just skip. For now, we copy original to avoid crashing benchmark
                shutil.copy(src_path, dst_ood)

    print(f"\n✅ All 5 Subsets Ready at {OUTPUT_ROOT}")
    print("Structure per subset:")
    print("  /subset_X_severity_X")
    print("     /clean/images")
    print("     /clean/annotations")
    print("     /ood/images")
    print("     /ood/annotations")

if __name__ == "__main__":
    main()