import subprocess
import json

# Get original config
with open("config.json", "r") as f:
    config = json.load(f)

# Make sure to log
config["wandb"] = True

# Generate new configs and run test
for model in ["rn", "vgg", "in", "an"]:
    for gap in [True, False]:
        for pretrained in [True, False]:
            
            # Log
            print(f"Testing model={model}, gap={gap}, pretrained={pretrained}.")

            # No pretrained weights for LeNet
            if pretrained and model == "ln":
                continue

            # Create new local config
            local_config = config.copy()

            # Update gridserch params
            local_config["model"] = model
            local_config["gap"] = gap

            # Save temporary config
            with open("local_config.json", "w") as f:
                json.dump(local_config, f)

            # Run the job
            try:
                subprocess.run(["python", "train.py", "local_config.json"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Training failed for model={model}, gap={gap}")
                print(f"Error: {e}")

print("\nGrid search done!")