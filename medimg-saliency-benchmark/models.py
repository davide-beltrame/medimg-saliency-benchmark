import lightning as pl
import torch
import torch.nn as nn
from torchvision.models import (
    resnet101, ResNet101_Weights,  # Updated import for ResNet101
    vgg16, VGG16_Weights,  # Updated import for VGG16
    googlenet, GoogLeNet_Weights,  # Updated import for Inception V1
    alexnet, AlexNet_Weights
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
        assert config.model in {"an", "vgg", "rn", "in"}
        
        if config.model == "an":
            self.model = AlexNetBinary(pretrained=config.pretrained, linear=config.linear)
        elif config.model == "vgg":
            self.model = VGG16Binary(pretrained=config.pretrained, linear=config.linear)
        elif config.model == "rn":
            assert config.linear # already GAP + FC, cannot be false
            self.model = ResNet101Binary(pretrained=config.pretrained)  
        elif config.model == "in":
            assert config.linear # already GAP + FC, cannot be false
            self.model = InceptionNetBinary(pretrained=config.pretrained)   


    def forward(self, X):
        """
        Simply forward the backbone model.
        """
        return self.model(X)

    def _common_step(self, batch):
        """
        Reads batch, gets logits, computes loss & other metrics.
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
        
        # Train-specific metrics
        metrics["lr"] = self.optimizers().optimizer.param_groups[0]["lr"]
        metrics["batch_pos_perc"] = batch[1].mean()

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
        
        # Get the loss
        loss, metrics = self._common_step(batch)

        # Test-specific metric
        metrics["batch_pos_perc"] = batch[1].mean()

        # Log metrics with correct label
        for k, v in metrics.items():
            self.log(
                k.format(stage="test"),
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True
        )
            
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
        scheduler = OneCycleLR(
            optimizer=optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
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


class AlexNetBinary(nn.Module):
    """
    AlexNet default:
    4x(conv + relu + maxpool)
    + AvgPool (-> 6,6)
    + FC
    + FC
    + FC
    """
    def __init__(self, pretrained=True, linear=True):
        super().__init__()
        weights = AlexNet_Weights.DEFAULT if pretrained else None
        self.model = alexnet(weights=weights)
        self.linear = linear
        
        if self.linear:
            # Extract features (remove classifier)
            self.features = self.model.features  # [B, 256, H, W]
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(256, 1)  # AlexNet has 256 features at the last convolutional layer
        else:
            # Modify original classifier for binary output
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_ftrs, 1)
    
    def forward(self, x):
        if self.linear:
            x = self.features(x)
            x = self.gap(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        else:
            x = self.model(x)
        return x


class VGG16Binary(nn.Module):
    """
    VGG16 default:
    13xConv (+ pool sometimes)
    + AvgPool (-> 7,7)
    + FC
    + FC
    + FC
    """
    def __init__(self, pretrained=True, linear=True):
        super().__init__()
        # Updated to use weights parameter instead of pretrained
        weights = VGG16_Weights.DEFAULT if pretrained else None
        self.model = vgg16(weights=weights)
        self.linear = linear

        if self.linear:
            # Extract features (remove classifier)
            self.features = self.model.features  # [B, 512, H, W]
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(512, 1)
        else:
            # Modify original classifier for binary output
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_ftrs, 1)
    
    def forward(self, x):
        if self.linear:
            x = self.features(x)
            x = self.gap(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        else:
            x = self.model(x)
        return x

class ResNet101Binary(nn.Module):
    """
    ResNet101 default:
    Conv
    + 3 x(Conv + Conv + Downsample)
    + 4 x(Conv + Conv + Downsample)
    + 23x(Conv + Conv + Downsample)
    + 3 x(Conv + Conv + Downsample)
    + AvgPool
    + FC
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # Updated to use weights parameter instead of pretrained
        weights = ResNet101_Weights.DEFAULT if pretrained else None
        self.model = resnet101(weights=weights)

        # Keep original structure but adapt final layer for binary
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)
    
    def forward(self, x):
        return self.model(x)

class InceptionNetBinary(nn.Module):
    """
    InceptionNet (a.k.a. GoogleNet) default:
    2x(Conv+Pool)
    + 9xInceptionBlock
    + AvgPool
    + FC
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # Updated to use weights parameter instead of pretrained
        weights = GoogLeNet_Weights.DEFAULT if pretrained else None
        self.model = googlenet(weights=weights)
        
        # Inception outputs with auxiliary classifier during training
        # Set this to False to get just the main output when calling forward
        self.model.aux_logits = False

        # Adapt for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)
    
    def forward(self, x):
        return self.model(x)