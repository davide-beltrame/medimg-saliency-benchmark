"""
Test all the models saved in /checkpoints/ on the test set.
"""
import json
import os
import argparse

import lightning as pl
from models import BaseCNN
from datamodule import Datamodule
from utils import BaseConfig, BootstrapTestCallback

# Create dir to store
os.makedirs("evaluation", exist_ok=True)

# read args
parser = argparse.ArgumentParser()
parser.add_argument("path_to_config", type=str, help="path/to/config.json")
args = parser.parse_args()

# Load default config
config = BaseConfig(args.path_to_config)

# Datamodule (with test logic implemented)
dm = Datamodule(config=config)

# For boostrapping
bootstrap = BootstrapTestCallback(
    n_bootstrap_samples=1000,
    confidence_level=0.95,
    seed=42
)

if __name__ == '__main__':
    # Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        precision="16-mixed",  # if your model was trained with mixed precision
        logger=False,
        enable_checkpointing=False,
        callbacks=[bootstrap]
    )

    for path_to_ckpt in os.listdir("./checkpoints"):
        
        # Skip non-checkpoint files
        if not path_to_ckpt.endswith('.ckpt'):
            print(f"Skipping non-checkpoint file: {path_to_ckpt}")
            continue

        # Info
        print(f"Testing: {path_to_ckpt}")

        # Get path
        path_to_ckpt = os.path.join("./checkpoints/", path_to_ckpt)

        # Load trained model
        model = BaseCNN.load_from_checkpoint(path_to_ckpt)

        # Test and get results
        # list of dictionaries, one per test dataloader 
        results = trainer.test(model, datamodule=dm)

        # Save results to json
        payload = {}
        payload["batch_size"] = config.batch_size
        payload["results"] = results
        path_to_results = os.path.join(
            "evaluation",
            os.path.basename(path_to_ckpt) + ".json"
        )
        with open(f"{path_to_results}", "w") as f:
            json.dump(payload, f, indent=4)

        # break