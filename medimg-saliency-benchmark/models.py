import lightning as pl
import torch
import torch.nn as nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    vgg16, VGG16_Weights,
    googlenet, GoogLeNet_Weights,
    alexnet, AlexNet_Weights
)
from torchmetrics import MeanMetric
from torchmetrics.classification import (
    BinaryAccuracy, 
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score
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
            self.model = ResNet50Binary(pretrained=config.pretrained)  
        elif config.model == "in":
            assert config.linear # already GAP + FC, cannot be false
            self.model = InceptionNetBinary(pretrained=config.pretrained)

    def forward(self, X):
        """
        Simply forward the backbone model.
        """
        return self.model(X)

    def _common_step(self, batch, stage):
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

        # Convert predictions to probabilities
        preds = torch.nn.functional.sigmoid(logits).view(-1)
        
        # Update metrics based on stage
        if stage == "train":
            # For training, log batch-level metrics
            batch_metrics = {
                "train/loss": loss,
                "lr": self.optimizers().optimizer.param_groups[0]["lr"],
                "batch_pos_perc": y.mean()
            }
            self.log_dict(batch_metrics, on_step=True, on_epoch=False, prog_bar=True)
            
        elif stage == "valid":
            # For validation, update accumulated metrics
            self.valid_loss(loss)
            self.valid_accuracy(preds, y)
            self.valid_precision(preds, y)
            self.valid_recall(preds, y)
            self.valid_f1(preds, y)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "valid")
    
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")
    
    def on_validation_epoch_start(self):
        # Initialize validation metrics
        self.valid_loss = MeanMetric().to(self.device)
        self.valid_accuracy = BinaryAccuracy().to(self.device)
        self.valid_precision = BinaryPrecision().to(self.device)
        self.valid_recall = BinaryRecall().to(self.device)
        self.valid_f1 = BinaryF1Score().to(self.device)
    
    def on_validation_epoch_end(self):
        # Compute final metrics over the entire validation set
        metrics = {
            "valid/loss": self.valid_loss.compute(),
            "valid/accuracy": self.valid_accuracy.compute(),
            "valid/precision": self.valid_precision.compute(),
            "valid/recall": self.valid_recall.compute(),
            "valid/f1": self.valid_f1.compute()
        }
        
        # Log the epoch-level metrics
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        Configure the optimizer and the learning rate scheduling policy.
        """

        # Optimizer
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=0.001,  #  Reset to max_lr / div_factor by the OneCyleLR
            betas=[0.9,0.95],
            weight_decay=1e-4
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
        
        # Load arch
        weights = AlexNet_Weights.DEFAULT if pretrained else None
        model = alexnet(weights=weights)
        self.linear = linear
        
        # Backbone
        self.features = model.features  # [B, 256, H, W]
        
        # Classifier
        if self.linear:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(256, 1)  # AlexNet has 256 features at the last convolutional layer
        else:
            self.gap = model.avgpool
            self.classifier = model.classifier
            self.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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
        
        # Load arch
        weights = VGG16_Weights.DEFAULT if pretrained else None
        model = vgg16(weights=weights)
        
        # Backbone
        self.features = model.features  # [B, 512, H, W]

        # Classifier
        if linear:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(512, 1)
        else:
            self.classifier = model.classifier
            self.gap = model.avgpool
            self.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResNet50Binary(nn.Module):
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
        
        # Load arch
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.model = resnet50(weights=weights)

        # Adapt for binary classification
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
    
    def forward(self, x):
        return self.model(x)
    

class InceptionNetBinary(nn.Module):
    """
    InceptionNet (a.k.a. GoogleNet) default:
    2x(Conv+Pool)
    + 9xInceptionBlock
    + AvgPool
    + FC

    Unfortunately, GoogleNet implementation does not allow to get intermediate features.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load arch
        weights = GoogLeNet_Weights.DEFAULT if pretrained else None
        self.model = googlenet(weights=weights)
        self.model.aux_logits = False # do not use auxiliary classifier during forward
        
        # Adapt for binary classification
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
    
    def forward(self, x):
        return self.model(x)