import argparse
import os

import lightning as pl
from datamodule import Datamodule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from models import BaseCNN
from utils import BaseConfig

def main():

    # Load config.json file
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_config", type=str, help="path/to/config.json")
    args = parser.parse_args()
    assert os.path.exists(args.path_to_config)
    config = BaseConfig(args.path_to_config)
    
    # Setup logging
    logger = CSVLogger(save_dir=".")

    # Load the datamodule
    datamodule = Datamodule(config)
    
    # To get the steps attribute for the annealing scheduler
    config.n_steps = len(datamodule.train_ds) // config.batch_size

    # Instantiate a model
    model = BaseCNN(config=config)

    # Checkpointers
    checkpointer = ModelCheckpoint(
        dirpath="checkpoints",  # Directory to save checkpoints
        filename=f"{config.model}_" + "{valid/loss:.2f}",  # Checkpoint filename format
        save_top_k=1,  # Save the best model
        monitor="valid/loss",  # Metric to monitor
        mode="min",  # Mode ('min' for loss, 'max' for accuracy)
        auto_insert_metric_name=False,  # To avoid issues when "/" in metric name
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
        callbacks=[checkpointer],
        enable_progress_bar=True,
    )

    # Train
    trainer.fit(
        model,
        datamodule,
    )

if __name__ == '__main__':
    main()