import torch
import torch.nn as nn
import torch.optim as optim
import lightning as pl
from torchvision import models
import torchmetrics # Use torchmetrics for accuracy, AUC, etc.
import json
import os

class CNNClassifier(pl.LightningModule):
    """
    PyTorch Lightning Module for CNN-based binary classification
    using transfer learning.
    """
    def __init__(self, config_path: str):
        super().__init__()
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Extract relevant config parameters
        model_conf = self.config.get("models", {})
        train_conf = self.config.get("training", {})

        self.model_name = model_conf.get("cnn_architecture", "ResNet50") # Default if not specified
        self.num_classes = model_conf.get("num_classes", 1) # Binary classification = 1 output neuron
        if self.num_classes != 1:
            raise ValueError("This module currently only supports binary classification (num_classes=1).")

        self.lr = train_conf.get("lr_initial", 1e-3) # Learning rate for head / initial training
        self.lr_finetune = train_conf.get("lr_finetune", 1e-5) # Learning rate for fine-tuning
        self.unfreeze_layers = train_conf.get("unfreeze_layers", 20) # How many layers to unfreeze from end

        # Load pre-trained model
        self.feature_extractor = self._get_base_model()

        # Freeze base model initially
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Get the number of features from the base model's output
        n_features = self._get_feature_output_size()

        # Create classification head
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 512), # Example intermediate layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes) # Output layer
        )

        # Loss function (handles sigmoid internally)
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics using torchmetrics
        self.train_accuracy = torchmetrics.Accuracy(task="binary")
        self.val_accuracy = torchmetrics.Accuracy(task="binary")
        self.train_auc = torchmetrics.AUROC(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")
        self.val_precision = torchmetrics.Precision(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")

        # To control fine-tuning stage
        self.fine_tuning_stage = False
        self.initial_epochs_done = train_conf.get("initial_epochs", 5)

        # Save hyperparameters
        self.save_hyperparameters(self.config)


    def _get_base_model(self):
        """Loads the specified pre-trained base model."""
        if self.model_name == "ResNet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif self.model_name == "DenseNet121":
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        elif self.model_name == "VGG16":
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        elif self.model_name == "MobileNetV2":
            model = models.mobilenet_v2(weights=models.MobileNetV2_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        # Remove the original classifier
        if hasattr(model, 'fc'): # ResNet, MobileNet
            model.fc = nn.Identity()
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential): # VGG
            # Find the last Linear layer and replace it
            # VGG classifier is Sequential(Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear)
             modules = list(model.classifier.children())[:-1] # Remove last Linear layer
             model.classifier = nn.Sequential(*modules) # Rebuild without last layer
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear): # DenseNet
            model.classifier = nn.Identity()
        else:
            raise ValueError(f"Could not automatically remove classifier from {self.model_name}")

        return model

    def _get_feature_output_size(self):
         """Calculates the output feature size of the base model."""
         # Create a dummy input tensor
         dummy_input = torch.randn(1, 3, self.hparams.data['target_size'][0], self.hparams.data['target_size'][1])
         # Pass it through the feature extractor
         self.feature_extractor.eval() # Set to eval mode
         with torch.no_grad():
              output = self.feature_extractor(dummy_input)
         self.feature_extractor.train() # Set back to train mode
         # The output might need pooling if it's not already flattened
         # Check output shape - assume it might need Global Average Pooling
         if output.dim() > 2:
             # Apply Adaptive Average Pooling to get (batch_size, features, 1, 1) -> flatten
              pool = nn.AdaptiveAvgPool2d((1, 1))
              output = pool(output)
              output = torch.flatten(output, 1)

         return output.shape[1]


    def forward(self, x):
        features = self.feature_extractor(x)
        # Apply pooling if needed (depends on base model output)
        if features.dim() > 2:
             pool = nn.AdaptiveAvgPool2d((1, 1))
             features = pool(features)
             features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # Reshape y to match logits if necessary (BCEWithLogitsLoss expects (N,) or (N,1))
        loss = self.criterion(logits.squeeze(), y.float()) # Squeeze logits if (N,1)

        # Log metrics
        preds = torch.sigmoid(logits.squeeze()) # Get probabilities for metrics
        self.train_accuracy(preds, y)
        self.train_auc(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits.squeeze(), y.float())

        # Log metrics
        preds = torch.sigmoid(logits.squeeze())
        self.val_accuracy(preds, y)
        self.val_auc(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_f1(preds, y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.val_accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_auc', self.val_auc, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', self.val_precision, on_epoch=True, logger=True)
        self.log('val_recall', self.val_recall, on_epoch=True, logger=True)
        self.log('val_f1', self.val_f1, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Similar to validation step, often used for final evaluation
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits.squeeze(), y.float())
        # Log test metrics if needed
        self.log('test_loss', loss, on_epoch=True, logger=True)
        # Calculate and log other test metrics (acc, auc, etc.) using separate torchmetrics instances if desired

    def configure_optimizers(self):
        if not self.fine_tuning_stage:
            # Initial stage: optimize only the classifier head
            optimizer = optim.Adam(self.classifier.parameters(), lr=self.lr)
            print("--- Configuring optimizer for HEAD ONLY ---")
        else:
            # Fine-tuning stage: optimize classifier + unfrozen base layers
            # Find unfrozen parameters
            unfrozen_params = [p for p in self.feature_extractor.parameters() if p.requires_grad]
            all_params = list(self.classifier.parameters()) + unfrozen_params
            optimizer = optim.Adam(all_params, lr=self.lr_finetune)
            print(f"--- Configuring optimizer for FINE-TUNING ({len(unfrozen_params)} base params) ---")

        # Optional: Add a learning rate scheduler (e.g., ReduceLROnPlateau)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True) # Monitor val_auc

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_auc", # Metric to monitor for scheduler
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self):
        # Check if it's time to switch to fine-tuning
        if not self.fine_tuning_stage and self.current_epoch == self.initial_epochs_done:
            print(f"\n--- Epoch {self.current_epoch}: Switching to FINE-TUNING ---")
            self.fine_tuning_stage = True
            self._unfreeze_base_layers()
            # Reconfigure optimizers needs trainer support or handle param groups manually
            # Lightning automatically calls configure_optimizers again if trainer.optimizers is reset or state changes
            # A common way is to let configure_optimizers handle the stage logic based on self.fine_tuning_stage
            # We also need to manually update the trainer's optimizer state or reload it
            # Simpler approach: let configure_optimizers return different params based on stage
            # Force re-initialization of optimizer and scheduler
            self.trainer.strategy.setup_optimizers(self.trainer) # May work depending on Lightning version/strategy

    def _unfreeze_base_layers(self):
        """Unfreezes the top N layers of the base model."""
        # Find all layers of the base model
        layers = list(self.feature_extractor.children()) # Or .modules() for deeper nesting
        # Flatten potentially nested structures like Sequential blocks if needed
        all_layers = []
        for layer in self.feature_extractor.modules():
            # Avoid adding top-level container itself or too generic modules
             if isinstance(layer, nn.modules.conv.Conv2d) or \
                isinstance(layer, nn.modules.linear.Linear) or \
                isinstance(layer, nn.modules.batchnorm.BatchNorm2d):
                 all_layers.append(layer)


        num_total_layers = len(all_layers)
        num_to_unfreeze = min(self.unfreeze_layers, num_total_layers)
        print(f"Total layers identified in base model: {num_total_layers}")
        print(f"Attempting to unfreeze last {num_to_unfreeze} layers...")

        unfrozen_count = 0
        # Iterate backwards through the layers
        for layer in reversed(all_layers):
            if unfrozen_count >= num_to_unfreeze:
                break
            # Unfreeze parameters of this layer
            # Make sure BatchNorm layers remain frozen during fine-tuning (common practice)
            if not isinstance(layer, nn.BatchNorm2d):
                 for param in layer.parameters():
                     param.requires_grad = True
                 unfrozen_count += 1
                 # print(f"Unfroze layer: {layer}") # Verbose logging
            # else:
                 # print(f"Keeping BatchNorm frozen: {layer}")

        # Verify how many parameters are now trainable
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters after unfreezing: {trainable_params}")