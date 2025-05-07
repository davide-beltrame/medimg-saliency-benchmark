"""
Test all the trained models on the test set.
"""

import time
import json

import lightning as pl
from models import BaseCNN
from datamodule import Datamodule
from utils import BaseConfig

# Load default config
config = BaseConfig("./config.json")

# Datamodule (with test logic implemented)
dm = Datamodule(config=config)

# Load pretrianed model
model = BaseCNN.load_from_checkpoint("path/to/checkpoint.ckpt")

# Trainer
trainer = pl.Trainer(
    accelerator="auto",
    devices="auto",
    precision="16-mixed",  # if your model was trained with mixed precision
    logger=False,
    enable_checkpointing=False,
)

# Test and get results
# list of dictionaries, one per test dataloader 
results = trainer.test(model, datamodule=dm)[0]

# Save results to json
payload = {}
payload["config"] = config.__dict__
payload["results"] = results
with open(f"{int(time.time())}.json", "w") as f:
    json.dump(payload, f, indent=4)
