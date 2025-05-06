import json

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, densenet121, efficientnet_b0, mobilenet_v3_large
from torchmetrics.classification import BinaryPrecision, BinaryF1Score, BinaryAccuracy, BinaryRecall
from torch.optim.lr_scheduler import OneCycleLR


class BaseCNN(pl.LightningModule):
    """
    A base class to implement a Pytorch Lightning Model.
    Hyper-parameters are specified in a separate config.json file.
    """

    def __init__(
        self,
        model:str=None,
        pretrained:bool=False
    ):
        super().__init__()

        # For checkpointing
        self.save_hyperparameters()

        # The actual model
        assert model in {"rn", "dn", "en", "mn"}
        
        if model == "rn":
            self.model = ResNet50Binary(pretrained=pretrained)
        elif model == "dn":
            self.model = DenseNet121Binary(pretrained=pretrained)
        elif model == "en":
            self.model = EfficientNetB0Binary(pretrained=pretrained)
        elif model == "mn":
            self.model = MobileNetV3Binary(pretrained=pretrained)

        # To evaluate the model
        self.accuracy = BinaryAccuracy()
        self.recall = BinaryRecall()
        self.precision = BinaryPrecision()
        self.f1 = BinaryF1Score()


    def forward(self, X):
        return self.model(X)

    def _common_step(self, batch):
        """
        Handles one step of the model, from reading in the batch to returning the loss.
        """
        # Read in the batch
        X, y = batch
        
        # Forward pass
        logits = self.forward(X)

        # Compute loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=logits.view(-1),
            target=y.view(-1)
        )

        return loss, logits

    def training_step(self, batch, batch_idx):
        # Get the loss
        loss, logits = self._common_step(batch)
        
        # Logging
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        
        X, y = batch
        
        # Get the loss
        loss, logits = self._common_step(batch)

        # Get predictions
        preds = torch.nn.functional.sigmoid(logits) > 0.5
        preds = preds.view(-1)

        # Compute metrics
        metrics = {
            "valid/accuracy": self.accuracy(preds, y),
            "valid/recall": self.recall(preds, y),
            "valid/precision": self.precision(preds, y),
            "valid/f1": self.f1(preds, y),
            "valid/loss": loss,
        }

        # Logging
        for k, v in metrics.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer and the learning rate scheduling policy.
        """

        # Optimizer
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=0.001,  #  Reset to max_lr / div_factor by the OneCyleLR
        )

        # The 1cycle policy (warm-up + annealing)
        # scheduler = OneCycleLR(
        #     optimizer,
        #     max_lr=0.01,
        #     total_steps=100,
        #     pct_start=0.1,  # Warm-up percentage of total steps
        # )

        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "interval": "step",
            #     "frequency": 1,  # Check after each step
            # },
        }

    @torch.no_grad()
    def predict(self, X):
        
        # Add batch dim
        if X.dim() == 2:
            X = X.unsqueeze(0)

        # Move to correct device
        X = X.to(self.device)

        # Get logits
        logits = self.forward(X)

        # Get probability
        probs = torch.nn.functional.sigmoid(logits)

        return probs.round()


class ResNet50Binary(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = resnet50(pretrained=pretrained)

        # Remove original classifier
        self.features = nn.Sequential(*list(self.model.children())[:-2])  # Up to conv5_x
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # GAP layer
        self.classifier = nn.Linear(2048, 1)  # Binary output

    def forward(self, x):
        x = self.features(x)         # [B, 2048, H, W]                  # For Grad-CAM
        x = self.gap(x)              # [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)    # [B, 2048]
        x = self.classifier(x)       # [B, 1]
        return x

class DenseNet121Binary(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = densenet121(pretrained=pretrained)
        self.features = self.model.features  # [B, 1024, H, W]
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

class EfficientNetB0Binary(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = efficientnet_b0(pretrained=pretrained)
        self.features = self.model.features  # [B, 1280, H, W]
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MobileNetV3Binary(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = mobilenet_v3_large(pretrained=pretrained)
        self.features = self.model.features  # [B, 960, H, W]
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(960, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
