import subprocess
import json

# Get original config
with open("config.json", "r") as f:
    config = json.load(f)

# Make sure to log
config["wandb"] = True

# Generate new configs and run test
for model in ["in", "vgg", "rn", "an"]:
    for linear in [True, False]:
        for pretrained in [True, False]:
            
            if (model == "in" or model == "rn"):
                if not linear:
                    continue    # alread GAP + FC, no need to test adaptation
            # Log
            print(f"Testing model={model}, linear={linear}, pretrained={pretrained}.")

            # Create new local config
            local_config = config.copy()

            # Update gridserch params
            local_config["model"] = model
            local_config["linear"] = linear
            local_config["pretrained"] = pretrained
            local_config["run_name"] = f"{model}_{linear}_{pretrained}"

            # Save temporary config
            with open("local_config.json", "w") as f:
                json.dump(local_config, f)

            # Run the job
            try:
                subprocess.run(
                    ["python", "train.py", "local_config.json"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Training failed for model={model}, linear={linear}, pretrained={pretrained}.")
                print(f"Error: {e}")

print("\nGrid search done!")