import os
import requests
import tarfile
from tqdm import tqdm

# Configuration
DATASET_ROOT = "./datasets"
VOC_URLS = {
    "VOC2007_TRAINVAL": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
    "VOC2007_TEST": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
    "VOC2012_TRAINVAL": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
}

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return
    
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_tar(tar_path, dest_dir):
    print(f"Extracting {tar_path} to {dest_dir}...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=dest_dir)

def main():
    os.makedirs(DATASET_ROOT, exist_ok=True)
    
    # --- 1. Download PASCAL VOC ---
    print("--- 1. Setting up PASCAL VOC (Source Domain) ---")
    voc_dir = os.path.join(DATASET_ROOT, "VOCdevkit")
    os.makedirs(voc_dir, exist_ok=True)
    
    for name, url in VOC_URLS.items():
        filename = url.split('/')[-1]
        filepath = os.path.join(DATASET_ROOT, filename)
        download_file(url, filepath)
        extract_tar(filepath, DATASET_ROOT) # Extracts into VOCdevkit automatically
        
        # Cleanup tar file to save space (Optional)
        # os.remove(filepath) 

    # --- 2. Instructions for Artistic Datasets ---
    print("\n" + "="*50)
    print("--- 2. Instructions for Target Domains (Clipart, Watercolor, Comic) ---")
    print("="*50)
    print("These datasets are hosted on Google Drive by the original authors (Inoue et al.).")
    print("You must download them manually or use 'gdown'.")
    print("\nDownload Links:")
    print("1. Visit the repository: https://github.com/naoto0804/cross-domain-detection")
    print("2. Look for the 'Dataset' section or 'Cross-Domain Detection Dataset'.")
    print("3. Download: clipart.zip, watercolor.zip, comic.zip")
    print(f"4. Unzip them into: {os.path.abspath(DATASET_ROOT)}")
    print("\nExpected final structure:")
    print(f"  {DATASET_ROOT}/clipart/...")
    print(f"  {DATASET_ROOT}/watercolor/...")
    print(f"  {DATASET_ROOT}/comic/...")
    print("="*50)

if __name__ == "__main__":
    main()