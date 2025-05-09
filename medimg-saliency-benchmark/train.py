import argparse
import os
import time

import lightning as pl
from datamodule import Datamodule
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from models import BaseCNN
from utils import BaseConfig

def main():

    # Load config.json file
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_config", type=str, help="path/to/config.json")
    args = parser.parse_args()
    assert os.path.exists(args.path_to_config)
    config = BaseConfig(args.path_to_config)
    
    # pui
    if hasattr(config, "run_name"):
        run_name = config.run_name
    else:
        run_name = f"{config.model}_{int(time.time())}"

    # Setup logging
    logger =  WandbLogger(
        name=run_name,
        project="CVIP",
        config=config.__dict__
    ) if config.wandb else CSVLogger(save_dir=".")

    # Load the datamodule
    datamodule = Datamodule(config)

    # Instantiate a model
    model = BaseCNN(config=config)

    # Checkpointers
    checkpointer = ModelCheckpoint(
        dirpath="checkpoints",  # Directory to save checkpoints
        filename=f"{run_name}_" + "{valid/loss:.2f}",  # Checkpoint filename format
        save_top_k=1,  # Save the best model
        monitor="valid/loss",  # Metric to monitor
        mode="min",  # Mode ('min' for loss, 'max' for accuracy)
        auto_insert_metric_name=False,  # To avoid issues when "/" in metric name
    )

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="valid/loss",  # Monitor validation cross-entropy loss
        patience=5,  # Number of validation checks with no improvement after which training will stop
        min_delta=0.001,  # Minimum change in monitored value to qualify as improvement
        mode="min",  # We want to minimize the loss
        verbose=True,  # Print message when early stopping is triggered
        check_on_train_epoch_end=False,  # Check at validation time, not training end
    )
    # Init the trainer
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="auto",  # recognizes device
        devices="auto",  # how many devices to use
        precision="16-mixed",  # to use amp 16
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=True,  # saves the most recent model after each epoch (default True)
        callbacks=[checkpointer, early_stopping],
        enable_progress_bar=True,
        val_check_interval=1.0,
    )

    # Train
    trainer.fit(
        model,
        datamodule,
    )

if __name__ == '__main__':
    main()