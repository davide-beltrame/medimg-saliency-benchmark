import os
from collections import Counter
from torch.utils.data import WeightedRandomSampler
from PIL import Image
import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from utils import BaseConfig

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
        assert split in {"train", "test"}

        base_path = f"./data/{split}"
        class_dirs = {"NORMAL": 0, "COVID19": 1}

        self.imgs, self.targets = [], []

        for class_name, label in class_dirs.items():
            class_path = os.path.join(base_path, class_name)
            assert os.path.exists(class_path)
            for filename in os.listdir(class_path):
                self.imgs.append(os.path.join(class_path, filename))
                self.targets.append(label)
        
        # Print info about classes
        n_pos = sum(self.targets)
        print(f"Init {split} dataset with {n_pos} covid images and {len(self.targets) - n_pos} normal images.")

        # Transormation to apply to the raw image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()       # also divides by 255
        ])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        # Load the image
        X = Image.open(self.imgs[idx]).convert("RGB")

        # Apply transformations
        X = self.transform(X)

        # Convert to tensor the label
        y = torch.tensor(self.targets[idx]).to(torch.float32)
        
        return X, y


class Datamodule(pl.LightningDataModule):
    """
    A custom datamodule to get the data, prepare it and return batches of X,y.
    """

    def __init__(
            self,
            config:BaseConfig
        ):
        super().__init__()

        # Set attributes
        self.config = config
        self.num_workers = 2
    
    def setup(self, stage=None):

        # Init a dataset
        self.data = Dataset(split="train")

        # Split the dataset
        self.train_ds, self.val_ds = random_split(
            self.data, [0.9, 0.1]
        )

        # Make the dataset balanced using weighted sampling
        ys = [self.data.targets[idx] for idx in self.train_ds.indices]
        class_weights = {y: 1.0 / count for y, count in Counter(ys).items()}    # measure the weight
        sample_weights = [class_weights[label] for label in ys]     # list of weight for each label
        self.train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
       
        # Prepare the test split
        self.test_ds = Dataset(split="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            num_workers=self.num_workers,
            # shuffle=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            sampler=self.train_sampler
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.config.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )