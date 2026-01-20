import os
import shutil
from ultralytics.utils.downloads import download
from pathlib import Path
import yaml

# Configuration
DATASET_ROOT = Path("./datasets")
COCO_YAML_URL = "https://github.com/ultralytics/ultralytics/raw/main/ultralytics/cfg/datasets/coco128.yaml" 
# NOTE: Using 'coco128' (128 images) first to test the pipeline. 
# Once this works, you can change URL/download for full 'coco.yaml'.

def setup_coco():
    print("--- 1. Downloading COCO (Source Domain) ---")
    # This downloads the zip to the current dir and unzips
    # coco128 is a tiny subset of COCO train2017
    
    # We use the YOLO utility which often bypasses academic blocks
    if not (DATASET_ROOT / "coco128").exists():
        try:
            # This downloads to ./datasets/coco128
            download(COCO_YAML_URL, dir=DATASET_ROOT, unzip=True)
            print("✅ COCO128 Downloaded successfully.")
        except Exception as e:
            print(f"❌ Failed to download COCO: {e}")
            return
    else:
        print("ℹ️  COCO128 already exists.")

    # Move/Structure standard COCO if needed
    # For this script, Ultralytics structure is: datasets/coco128/images/train2017
    
if __name__ == "__main__":
    os.makedirs(DATASET_ROOT, exist_ok=True)
    setup_coco()

