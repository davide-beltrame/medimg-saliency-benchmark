# scripts/download_data.py
import kagglehub
import os
import logging
import zipfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define dataset details
# DATASET_SLUG = "prashant268/chest-xray-covid19-pneumonia"
# DATASET_SLUG = "amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset"
# DATASET_SLUG = "tolgadincer/labeled-chest-xray-images"
DATASET_SLUG = "paultimothymooney/chest-xray-pneumonia"
# Define target directory within your project structure
# kagglehub downloads to ~/.cache by default, then we copy/move
TARGET_DATA_DIR = "data/raw/chest-xray-covid19-pneumonia" # Where we want the data finally

def download_and_extract_dataset(slug, final_path):
    """Downloads using kagglehub and handles potential extraction/copying."""
    logging.info(f"Attempting to download dataset '{slug}' using kagglehub...")
    try:
        # Download dataset (returns path to downloaded files/archive)
        # It might download to a cache location like ~/.cache/kagglehub/...
        download_path = kagglehub.dataset_download(slug)
        logging.info(f"Kagglehub reported download path: {download_path}")

        # Ensure the final target directory exists
        os.makedirs(final_path, exist_ok=True)

        # Check if the download path is what we expect or needs handling
        # Often kagglehub downloads a folder structure directly.
        # Let's assume it downloaded the structure we need inside download_path

        # Copy contents from download_path to final_path
        # Be careful with existing files if re-running
        logging.info(f"Copying contents from {download_path} to {final_path}...")
        if os.path.isdir(final_path):
             # Simple copy - might need refinement if structure differs
             for item in os.listdir(download_path):
                 s = os.path.join(download_path, item)
                 d = os.path.join(final_path, item)
                 if os.path.isdir(s):
                     shutil.copytree(s, d, dirs_exist_ok=True) # requires Python 3.8+
                 else:
                     shutil.copy2(s, d)
             logging.info("Dataset contents copied successfully.")
             return final_path

        else:
            logging.error(f"Target path {final_path} is not a directory or does not exist.")
            return None

    except Exception as e:
        logging.error(f"Failed during dataset download or handling: {e}")
        logging.error("Please ensure you have authenticated with Kaggle (e.g., ~/.kaggle/kaggle.json).")
        return None

if __name__ == "__main__":
    final_data_path = download_and_extract_dataset(DATASET_SLUG, TARGET_DATA_DIR)

    if final_data_path:
        print(f"\nDataset should be available at: {final_data_path}")
        print("Please verify the structure: train/{COVID19, NORMAL, PNEUMONIA}, test/{COVID19, NORMAL, PNEUMONIA}")
    else:
        print("\nDataset download/setup failed. Check logs and Kaggle authentication.")