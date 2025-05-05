import os
import json
import time
import sys
import argparse
from datetime import datetime

import lightning as pl
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Import project modules
from datamodule import ChestXRayDataModule # Use the new CV datamodule
from model_module import CNNClassifier # Use the new CV LightningModule
# from utils import check_config_validity # If you adapt this

def main():
    # --- Configuration ---
    parser = argparse.ArgumentParser(description="Train a CNN model for COVID vs Normal classification using PyTorch Lightning.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration JSON file.")
    # Allow overriding specific config values via command line if needed
    parser.add_argument("--model_name", type=str, help="Override model architecture (e.g., ResNet50)")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--initial_epochs", type=int, help="Override initial epochs for head training")
    parser.add_argument("--fine_tune_epochs", type=int, help="Override epochs for fine-tuning")
    # Add other overrides as needed (lr, unfreeze_layers, etc.)

    args = parser.parse_args()
    CONFIG_PATH = args.config

    # Load configuration file
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {CONFIG_PATH}")
        sys.exit(1)

    # Override config with command-line args if provided
    if args.model_name:
        config["models"]["cnn_architecture"] = args.model_name
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.initial_epochs:
         config["training"]["initial_epochs"] = args.initial_epochs
    if args.fine_tune_epochs:
         config["training"]["fine_tune_epochs"] = args.fine_tune_epochs
    # Add other overrides here...

    # Check validity of configuration (if function is adapted)
    # check_config_validity(config)

    # --- Setup Logging ---
    log_conf = config.get("logging", {})
    model_name_for_log = config.get("models", {}).get("cnn_architecture", "cnn") # Get model name for logging
    run_name = log_conf.get("wandb_run_prefix", "train") + f"_{model_name_for_log}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if log_conf.get("use_wandb", False):
        try:
            logger = WandbLogger(
                name=run_name,
                project=log_conf.get("wandb_project", "cvip-saliency-benchmark"),
                config=config # Log the final config used
            )
        except ImportError:
            print("Wandb not installed. Falling back to CSVLogger. Run `pip install wandb`")
            logger = CSVLogger(save_dir="logs/", name=model_name_for_log)
    else:
        logger = CSVLogger(save_dir="logs/", name=model_name_for_log)


    # --- Load DataModule ---
    print("Loading DataModule...")
    datamodule = ChestXRayDataModule(CONFIG_PATH)

    # --- Instantiate Model ---
    print(f"Instantiating Model: {config.get('models', {}).get('cnn_architecture', 'DefaultCNN')}")
    model = CNNClassifier(CONFIG_PATH)

    # --- Setup Callbacks ---
    callbacks = []
    train_conf = config.get("training", {})
    checkpoint_dir = train_conf.get("checkpoint_dir", "checkpoints")

    # Model Checkpointing
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Save based on validation AUC
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, run_name), # Save checkpoints in run-specific folder
        filename='epoch_{epoch}_val_auc_{val_auc:.4f}',
        save_top_k=2,
        monitor='val_auc', # Monitor validation AUC
        mode='max',
        auto_insert_metric_name=False,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Early Stopping
    early_stopping_callback = EarlyStopping(
        monitor='val_auc', # Monitor validation AUC
        patience=train_conf.get("early_stopping_patience", 7), # Number of epochs with no improvement
        min_delta=train_conf.get("early_stopping_min_delta", 0.001),
        mode='max',
        verbose=True,
    )
    callbacks.append(early_stopping_callback)

    # --- Setup Trainer ---
    # Calculate total epochs
    total_epochs = train_conf.get("initial_epochs", 5) + train_conf.get("fine_tune_epochs", 15)

    trainer = pl.Trainer(
        max_epochs=total_epochs,
        accelerator="auto",
        devices="auto",
        precision=train_conf.get("precision", "16-mixed"), # Use mixed precision
        logger=logger,
        log_every_n_steps=train_conf.get("log_every_n_steps", 10),
        val_check_interval=train_conf.get("val_check_interval", 1.0), # Check validation every epoch by default
        callbacks=callbacks,
        enable_progress_bar=True,
        # Gradient clipping can be added if needed
        # accumulate_grad_batches=train_conf.get("accumulate_grad_batches", 1), # Gradient accumulation
    )

    # --- Train the Model ---
    print(f"--- Starting Training for {model.model_name} ---")
    # Consider adding ckpt_path argument if resuming training is needed later
    trainer.fit(model=model, datamodule=datamodule)

    print(f"--- Training Finished for {model.model_name} ---")
    print(f"Best model checkpoint path: {checkpoint_callback.best_model_path}")

    # Optional: Run testing phase if needed
    # print("--- Running Test Phase ---")
    # trainer.test(model=model, datamodule=datamodule, ckpt_path='best') # Use best checkpoint


if __name__ == "__main__":
    main()