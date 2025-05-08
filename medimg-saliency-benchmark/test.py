"""
Test all the models saved in /checkpoints/ on the test set.
"""
import json
import os

import lightning as pl
from models import BaseCNN
from datamodule import Datamodule
from utils import BaseConfig

# Create dir to store
os.makedirs("evaluation", exist_ok=True)

# Load default config
config = BaseConfig("./config.json")

# Datamodule (with test logic implemented)
dm = Datamodule(config=config)

# Trainer
trainer = pl.Trainer(
    accelerator="auto",
    devices="auto",
    precision="16-mixed",  # if your model was trained with mixed precision
    logger=False,
    enable_checkpointing=False,
)

for path_to_ckpt in os.listdir("./checkpoints"):

    # Fix name
    path_to_ckpt = os.path.join("./checkpoints/", path_to_ckpt)

    # Load trained model
    model = BaseCNN.load_from_checkpoint(path_to_ckpt)

    # Test and get results
    # list of dictionaries, one per test dataloader 
    results = trainer.test(model, datamodule=dm)[0]

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
