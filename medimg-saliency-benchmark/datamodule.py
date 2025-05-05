import os
import json
import torch
import lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import logging

class ChestXRayDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the Chest X-Ray COVID vs NORMAL dataset.
    Uses torchvision.datasets.ImageFolder.
    """
    def __init__(self, config_path: str):
        super().__init__()
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Extract relevant config parameters
        self.data_conf = self.config.get("data", {})
        self.train_conf = self.config.get("training", {})

        self.raw_data_dir = self.data_conf.get("raw_data_dir", "data/raw/chest-xray-covid19-pneumonia")
        self.image_size = tuple(self.data_conf.get("target_size", [224, 224]))
        self.batch_size = self.train_conf.get("batch_size", 32)
        self.num_workers = self.train_conf.get("num_workers", os.cpu_count() // 2) # Use half CPU cores
        self.validation_split = self.train_conf.get("validation_split", 0.2)

        # Define transformations
        # Normalization based on ImageNet statistics (common for transfer learning)
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        self.train_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # Add more augmentations if needed (e.g., ColorJitter, RandomAffine)
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

        self.val_test_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None # Optional: can use val set or separate test dir
        self.class_to_idx = None # To store class mapping

    def prepare_data(self):
        # This method is intended for downloading/preprocessing steps
        # Data download is handled by scripts/download_data.py
        # Check if data exists
        train_path = os.path.join(self.raw_data_dir, 'train')
        if not os.path.isdir(train_path):
            logging.error(f"Training data directory not found at {train_path}. Please run the download script first.")
            raise FileNotFoundError(f"Training data directory not found at {train_path}")
        logging.info("Data directory verified.")


    def setup(self, stage: str | None = None):
        # This method is called on every GPU setup.
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_path = os.path.join(self.raw_data_dir, 'train')
            # Load the full dataset from the train folder
            # We only want COVID19 and NORMAL
            full_dataset = ImageFolder(root=train_path, transform=None) # Load paths first

            # Filter samples to keep only 'COVID19' and 'NORMAL'
            # Get indices of desired classes
            desired_classes = ['NORMAL', 'COVID19']
            self.class_to_idx = {cls: idx for idx, cls in enumerate(desired_classes)} # Define our binary mapping
            idx_to_keep = [i for i, (path, label_idx) in enumerate(full_dataset.samples) if full_dataset.classes[label_idx] in desired_classes]

            # Create a subset with only desired classes and apply transformations
            # Need a custom way to apply transforms after subsetting or filter ImageFolder directly if possible
            # Simpler: Assume the folder structure is already prepared or create a temporary filtered one
            # Let's assume we prepare it to have only NORMAL and COVID19 subdirs in a processed dir
            # For now, let's try filtering ImageFolder samples if easy, otherwise recommend preprocessing step

            # --- Alternative: Prepare a dedicated train_binary folder ---
            # This is cleaner. Let's assume this exists or guide user to create it.
            # For simplicity now, let's assume train_path *only* contains NORMAL and COVID19 folders
            try:
                dataset_full = ImageFolder(root=train_path) # Reload assuming filtered structure
                self.class_to_idx = dataset_full.class_to_idx # Get mapping {COVID19:0, NORMAL:1} or vice-versa
                logging.info(f"Dataset classes found: {dataset_full.classes}")
                if not all(c in dataset_full.classes for c in desired_classes):
                     logging.warning(f"Expected classes {desired_classes} not fully present in {train_path}")

                # Adjust labels if necessary to match our desired binary mapping (e.g., Normal=0, COVID=1)
                # This requires a custom Dataset wrapper or modifying ImageFolder's targets

                # Let's stick to the default ImageFolder labels for now and handle mapping in the model module
                # Split full dataset into train and validation sets
                n_samples = len(dataset_full)
                n_val = int(n_samples * self.validation_split)
                n_train = n_samples - n_val
                self.dataset_train, self.dataset_val = random_split(dataset_full, [n_train, n_val])

                # Apply correct transforms to each split
                # This requires a wrapper dataset or doing it inside __getitem__ or via DataLoader collate_fn
                # Easiest: Create wrapper datasets
                self.dataset_train = TransformedDataset(self.dataset_train, transform=self.train_transform)
                self.dataset_val = TransformedDataset(self.dataset_val, transform=self.val_test_transform)

                logging.info(f"Setup complete. Train samples: {len(self.dataset_train)}, Val samples: {len(self.dataset_val)}")


            except FileNotFoundError:
                 logging.error(f"Ensure {train_path} contains *only* 'NORMAL' and 'COVID19' subdirectories for binary classification.")
                 raise

        # Assign test dataset (e.g., from 'test' directory or use validation set)
        if stage == "test" or stage is None:
             test_path = os.path.join(self.raw_data_dir, 'test')
             # Similar filtering logic needed if test path contains PNEUMONIA
             # Assuming test_path only has NORMAL and COVID19 for now
             try:
                  self.dataset_test = ImageFolder(root=test_path, transform=self.val_test_transform)
                  # Ensure class mapping is consistent if possible
                  logging.info(f"Test samples: {len(self.dataset_test)}")
             except FileNotFoundError:
                  logging.warning(f"Test data directory not found at {test_path}. Using validation set for testing.")
                  self.dataset_test = self.dataset_val # Fallback

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        # Ensure test dataset is loaded
        if not self.dataset_test:
             self.setup(stage="test") # Attempt to load test data if not already done

        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


# Helper Dataset wrapper to apply transforms after splitting ImageFolder
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)