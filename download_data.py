import os
import requests
import tarfile
from tqdm import tqdm

# Configuration
DATASET_ROOT = "./datasets"

# UPDATED MIRRORS (pjreddie is more reliable than the academic server)
VOC_URLS = {
    "VOC2007_TRAINVAL": "http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar",
    "VOC2007_TEST": "http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar",
    "VOC2012_TRAINVAL": "http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar"
}

def download_file(url, dest_path):
    # Check if file exists and has reasonable size (> 10MB) to avoid re-downloading corruption
    if os.path.exists(dest_path):
        if os.path.getsize(dest_path) > 10 * 1024 * 1024: 
            print(f"File already exists and looks valid: {dest_path}")
            return
        else:
            print(f"Found corrupted/small file. Re-downloading: {dest_path}")
            os.remove(dest_path)
    
    print(f"Downloading {url}...")
    # Add headers to mimic a browser, preventing some 403/404 errors
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status() # Check for HTTP errors
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as file, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
                
    except Exception as e:
        print(f"FAILED to download {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path) # Clean up partial file

def extract_tar(tar_path, dest_dir):
    print(f"Extracting {tar_path} to {dest_dir}...")
    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=dest_dir)
    except tarfile.ReadError:
        print(f"CRITICAL ERROR: Could not extract {tar_path}. The file is likely corrupted.")

def main():
    os.makedirs(DATASET_ROOT, exist_ok=True)
    
    # --- 1. Download PASCAL VOC ---
    print("--- 1. Setting up PASCAL VOC (Source Domain) ---")
    voc_dir = os.path.join(DATASET_ROOT, "VOCdevkit")
    
    for name, url in VOC_URLS.items():
        filename = url.split('/')[-1]
        filepath = os.path.join(DATASET_ROOT, filename)
        
        download_file(url, filepath)
        
        # Only extract if download was successful
        if os.path.exists(filepath) and os.path.getsize(filepath) > 10 * 1024 * 1024:
            extract_tar(filepath, DATASET_ROOT) 
        else:
            print(f"Skipping extraction for {filename} (Download failed or file too small)")

    # --- 2. Instructions for Artistic Datasets ---
    print("\n" + "="*50)
    print("--- 2. Instructions for Target Domains (Clipart, Watercolor, Comic) ---")
    print("="*50)
    print("These datasets are hosted on Google Drive by the original authors (Inoue et al.).")
    print("You generally need to download them manually due to GDrive limitations.")
    print("\nDownload Links (GitHub: naoto0804/cross-domain-detection):")
    print("1. Clipart: https://drive.google.com/uc?id=0Bw3mB9T1uVbwd21sV3h3c1h3U28")
    print("2. Watercolor: https://drive.google.com/uc?id=0Bw3mB9T1uVbwWlJjX25wd190T00")
    print("3. Comic: https://drive.google.com/uc?id=0Bw3mB9T1uVbwN0F6X3k1X3h3U28")
    print(f"\nUnzip them into: {os.path.abspath(DATASET_ROOT)}")
    print("="*50)

if __name__ == "__main__":
    main()