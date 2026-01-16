import os
import requests
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

# Config
DATASET_ROOT = Path("./datasets/coco5k")
URLS = {
    "images": "http://images.cocodataset.org/zips/val2017.zip",  # 5000 Images (1GB)
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" # (241MB)
}

def download_file(url, dest_dir):
    filename = url.split('/')[-1]
    filepath = dest_dir / filename
    if filepath.exists():
        print(f"Skipping {filename} (already exists)")
        return filepath
        
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        total=total_size, unit='iB', unit_scale=True, unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    return filepath

def unzip_file(zip_path, dest_dir):
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)

def main():
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    
    # 1. Download & Extract
    img_zip = download_file(URLS["images"], DATASET_ROOT)
    ann_zip = download_file(URLS["annotations"], DATASET_ROOT)
    
    if not (DATASET_ROOT / "val2017").exists():
        unzip_file(img_zip, DATASET_ROOT)
    
    if not (DATASET_ROOT / "annotations").exists():
        unzip_file(ann_zip, DATASET_ROOT)

    # 2. Restructure for our Benchmark
    # We rename 'val2017' to 'train' because we are using it for training
    train_dir = DATASET_ROOT / "images" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    source_dir = DATASET_ROOT / "val2017"
    if source_dir.exists():
        print("Moving images to consistent structure...")
        # Move all files from val2017 to images/train
        for file in source_dir.glob("*.jpg"):
            shutil.move(str(file), str(train_dir / file.name))
        shutil.rmtree(source_dir)
        
    print("\n✅ COCO-5k Ready!")
    print(f"   Training Images: {train_dir} (5000 images)")
    print(f"   Annotations:     {DATASET_ROOT}/annotations/instances_val2017.json")

if __name__ == "__main__":
    main()
    