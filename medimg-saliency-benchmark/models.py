import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet50,
    densenet121,
    efficientnet_b0,
    mobilenet_v3_large
)
from torcheval.metrics.functional import (
    binary_accuracy,
    binary_precision,
    binary_recall,
    binary_f1_score
)
from torch.optim.lr_scheduler import OneCycleLR
from utils import BaseConfig

class BaseCNN(pl.LightningModule):
    """
    The base CNN class implementing the training with binary cross entropy.
    """

    def __init__(
        self,
        config: BaseConfig
    ):
        super().__init__()

        # For checkpointing
        self.save_hyperparameters()
        
        # Save configuration file
        self.config = config

        # The actual model
        assert config.model in {"rn", "dn", "en", "mn"}
        
        if config.model == "rn":
            self.model = ResNet50Binary(pretrained=config.pretrained)
        elif config.model == "dn":
            self.model = DenseNet121Binary(pretrained=config.pretrained)
        elif config.model == "en":
            self.model = EfficientNetB0Binary(pretrained=config.pretrained)
        elif config.model == "mn":
            self.model = MobileNetV3Binary(pretrained=config.pretrained)


    def forward(self, X):
        """
        Simply forward the backbone model.
        """
        return self.model(X)

    def _common_step(self, batch):
        """
        Reads batch, gets logits, computes loss & other metrics..
        """
        # Read in the batch
        # float32 & float32
        X, y = batch
        
        # Forward pass
        logits = self.forward(X)

        # Compute loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=logits.view(-1),
            target=y.view(-1)
        )

        # For logging binary metrics
        # [B,] of type float
        preds = torch.nn.functional.sigmoid(logits).view(-1)
        # required to be int
        y = y.to(torch.int32)
        metrics = {
            "{stage}/loss": loss,
            "{stage}/accuracy": binary_accuracy(preds, y),
            "{stage}/recall": binary_recall(preds, y),
            "{stage}/precision": binary_precision(preds, y),
            "{stage}/f1": binary_f1_score(preds, y),   
        }

        return loss, metrics

    def training_step(self, batch, batch_idx):

        # Get the loss
        loss, metrics = self._common_step(batch)
        
        # Log metrics with correct label
        for k, v in metrics.items():
            self.log(
                k.format(stage="train"),
                v,
                on_step=True,
                on_epoch=False,
                prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        
        # Get the loss
        loss, metrics = self._common_step(batch)

        # Log metrics with correct label
        for k, v in metrics.items():
            self.log(
                k.format(stage="valid"),
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True
        )
            
        return loss
    
    def test_step(self, batch, batch_idx):
        # TODO
        pass

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
        scheduler = OneCycleLR(
            optimizer,
            epochs=self.config.epochs,
            steps_per_epoch=self.trainer.estimated_stepping_batches,
            max_lr=self.config.max_lr,
            pct_start=self.config.pct_start,  # Warm-up percentage of total steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,  # Check after each step
            },
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
