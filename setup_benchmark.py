import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics.utils.downloads import download
from imagecorruptions import corrupt
from tqdm import tqdm
import shutil

# Configuration
ROOT = Path("./datasets")
SOURCE_NAME = "coco128"
TARGET_NAME = "coco128_corrupted"
CORRUPTIONS = ['fog', 'snow', 'frost', 'defocus_blur', 'gaussian_noise']
SEVERITY = 3  # 1 (light) to 5 (severe)

def setup_source():
    """Downloads standard COCO128 from Ultralytics (usually unblocked)"""
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
    if not (ROOT / SOURCE_NAME).exists():
        print(f"--- Downloading Source Domain ({SOURCE_NAME}) ---")
        download(url, dir=ROOT, unzip=True)
    else:
        print(f"--- Source Domain ({SOURCE_NAME}) found ---")

def generate_target():
    """Generates the OOD Target Domain using standard ImageNet-C corruptions"""
    src_img_dir = ROOT / SOURCE_NAME / "images/train2017"
    tgt_root = ROOT / TARGET_NAME
    
    print(f"--- Generating Target Domain ({TARGET_NAME}) ---")
    
    # 1. Prepare Directories
    if tgt_root.exists(): shutil.rmtree(tgt_root)
    
    # We will create a folder for each corruption type, or one merged folder
    # For this benchmark, let's create one merged folder for simplicity
    (tgt_root / "images").mkdir(parents=True, exist_ok=True)
    (tgt_root / "labels").mkdir(parents=True, exist_ok=True)

    img_files = list(src_img_dir.glob("*.jpg"))
    
    for img_path in tqdm(img_files, desc="Corrupting Images"):
        # Read
        image = cv2.imread(str(img_path))
        if image is None: continue
        
        # Pick a random corruption from the standard set to simulate domain shift
        corruption_name = np.random.choice(CORRUPTIONS)
        
        # Apply corruption (imagecorruptions expects RGB, OpenCV is BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        corrupted_rgb = corrupt(image_rgb, corruption_name=corruption_name, severity=SEVERITY)
        corrupted_bgr = cv2.cvtColor(corrupted_rgb, cv2.COLOR_RGB2BGR)
        
        # Save
        save_name = f"{img_path.stem}_{corruption_name}.jpg"
        cv2.imwrite(str(tgt_root / "images" / save_name), corrupted_bgr)
        
        # Copy Label (Labels are invariant)
        src_label = ROOT / SOURCE_NAME / "labels/train2017" / (img_path.stem + ".txt")
        if src_label.exists():
            shutil.copy(src_label, tgt_root / "labels" / (img_path.stem + f"_{corruption_name}.txt"))

    print(f"✅ Benchmark Ready.")
    print(f"   Source: {ROOT / SOURCE_NAME}")
    print(f"   Target: {tgt_root}")

if __name__ == "__main__":
    setup_source()
    generate_target()

