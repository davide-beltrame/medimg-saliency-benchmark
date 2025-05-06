import os

from PIL import Image
import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

class Dataset(Dataset):
    """
    A custom dataset class to handle data points.
    """

    def __init__(
        self,
        split:str="train"
    ):  
        
        assert split in {"train", "test"}
        # Your data
        path_to_normal = os.path.join(f"./data/{split}/NORMAL")
        path_to_covid = os.path.join(f"./data/{split}/COVID19")
        assert os.path.exists(path_to_normal)
        assert os.path.exists(path_to_covid)
        self.data = [
            (os.path.join(path_to_normal, filename), 0) for filename in os.listdir(path_to_normal)
        ] + [
            (os.path.join(path_to_covid, filename), 1) for filename in os.listdir(path_to_covid)
        ]

        # Transormation to apply to the raw image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()       # divides by 255
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        # Unpack the tuple
        path_to_X, y = self.data[idx]

        # Load the image
        X = Image.open(path_to_X).convert("RGB")

        # Apply transformations
        X = self.transform(X)
        
        # Convert to tensor the label
        y = torch.tensor(y).to(torch.float32)
        
        return X, y


class Datamodule(pl.LightningDataModule):
    """
    A custom datamodule to get the data, prepare it and return batches of X,y.
    """

    def __init__(self):
        super().__init__()

        # Set attributes
        self.batch_size = 16
        self.num_workers = 2

    def prepare_data(self):
        # Download data
        # Process once at the beginning of training
        # Called on only 1 GPU (even in multi-GPU scenarios)
        # Do not set state here (like self.x = y) as it won't be available across GPUs
        pass

    def setup(self, stage=None):

        # Init a dataset
        self.data = Dataset(split="train")

        # Split the dataset
        self.train_ds, self.val_ds = random_split(
            self.data, [0.9, 0.1]
        )

        # Prepare the test split
        self.test_data = Dataset(split="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )