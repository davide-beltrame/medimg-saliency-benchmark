import os
import glob
import json
import argparse
import torch
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from torchvision import transforms

import utils
from models import BaseCNN 
from models import AlexNetBinary, VGG16Binary, ResNet50Binary, InceptionNetBinary
import saliency
from datamodule import Dataset as PnDataset 

CHECKPOINT_DIR = "./checkpoints"
ANNOTATIONS_METADATA_PATH = "data/annotations/metadata.json"
ANNOTATED_MASKS_DIR = "data/annotations/annotated"
ORIGINAL_IMAGES_DIR_FOR_SALIENCY = "data/test" # Or wherever the original images for saliency evaluation are

MODEL_INPUT_SIZE = (224, 224) 

SALIENCY_BINARIZATION_THRESHOLD = 0.5

INITIAL_PRE_CLOSING_KERNEL_SIZE = 3
SOLIDITY_THRESHOLD = 0.6            
OUTLINE_FILL_CLOSING_KERNEL_SIZE = 7 
OUTLINE_EROSION_KERNEL_SIZE = 7      
FILLED_REGION_HOLE_CLOSING_KERNEL_SIZE = 5 
MIN_CONTOUR_AREA_FILTER = 20  
CONSENSUS_POST_FILTER_TYPE = 'open' # Filter applied to individual processed masks before consensus
CONSENSUS_POST_FILTER_KERNEL_SIZE = 3
CONSENSUS_METHOD = 'intersection'

# --- Helper Functions ---
def get_device():
    """Gets the appropriate torch device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_image_tensor(image_path, device):
    """Loads an image and converts it to a tensor for model input."""
    try:
        img = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(MODEL_INPUT_SIZE),
            transforms.ToTensor() # Scales to [0,1]
        ])
        img_tensor = transform(img).unsqueeze(0) # Add batch dimension
        return img_tensor.to(device)
    except FileNotFoundError:
        print(f"Warning: Image not found at {image_path}")
        return None

def find_checkpoint(model_short_key):
    """
    Finds a checkpoint file for a given model short key (e.g., "an", "vgg", "rn", "in").
    If multiple checkpoints match (e.g., different training parameters or epochs),
    it currently picks the first one found by glob, sorted alphabetically.
    You might want to add logic to pick the "best" one if scores are in filenames.

    Args:
        model_short_key (str): The short key for the model (e.g., "an", "vgg", "rn", "in").

    Returns:
        str or None: Path to the checkpoint file, or None if not found.
    """
    if not model_short_key:
        print("Warning: model_short_key is empty in find_checkpoint.")
        return None
        
    # Construct the pattern, e.g., "an_*.ckpt"
    # This assumes filenames like "an_True_True_0.05.ckpt"
    ckpt_pattern = os.path.join(CHECKPOINT_DIR, f"{model_short_key}_*.ckpt")
    ckpts = sorted(glob.glob(ckpt_pattern)) # Sort for consistency

    if ckpts:        
        selected_ckpt = ckpts[0] # Pick the first one (e.g., lowest loss if sorted by loss, or just first alphabetically)
        print(f"Found checkpoint for '{model_short_key}': {selected_ckpt}")
        return selected_ckpt
    else:
        if model_short_key == "vgg":
            ckpt_pattern_alt = os.path.join(CHECKPOINT_DIR, "vgg*.ckpt")
            ckpts_alt = sorted(glob.glob(ckpt_pattern_alt))
            if ckpts_alt:
                print(f"Found checkpoint for '{model_short_key}' using alternative pattern: {ckpts_alt[0]}")
                return ckpts_alt[0]

        print(f"Warning: No checkpoint found for model key '{model_short_key}' with pattern '{ckpt_pattern}' in {CHECKPOINT_DIR}")
        return None

def binarize_saliency_map(saliency_map_np, threshold=SALIENCY_BINARIZATION_THRESHOLD):
    """Binarizes a saliency map."""
    if saliency_map_np is None:
        return None
    return (saliency_map_np >= threshold).astype(np.uint8)

def generate_random_map(size=MODEL_INPUT_SIZE, grid_size=10):
    """Generates a random saliency map as per table description."""
    random_map_small = np.zeros((grid_size, grid_size), dtype=np.float32)
    # Pick one random pixel in the small grid to activate
    rand_x, rand_y = np.random.randint(0, grid_size, 2)
    random_map_small[rand_y, rand_x] = 1.0 
    # Upsample to full size
    random_map_full = cv2.resize(random_map_small, size, interpolation=cv2.INTER_NEAREST)
    return random_map_full # Already 0 or 1, effectively binarized

def get_consensus_masks_for_evaluation(annotations_metadata_list, annotated_masks_dir):
    """
    Generates final consensus masks for all images that have annotations.
    Returns a dictionary: {image_filename: consensus_mask_np}
    Only includes images where the final consensus mask is non-empty.
    """
    consensus_masks_dict = {}
    unique_image_names = sorted(list(set(record['image_name'] for record in annotations_metadata_list)))
    
    print(f"\nGenerating consensus masks for {len(unique_image_names)} unique images...")
    processed_count = 0
    for image_name in unique_image_names:
        raw_masks_tuples = utils.get_masks_for_image_from_metadata(
            image_name,
            annotations_metadata_list,
            annotated_masks_dir,
            target_size=MODEL_INPUT_SIZE 
        )

        base_processed_masks = []
        for raw_mask, annotator_name in raw_masks_tuples:
            processed_mask_step1 = utils.process_circled_annotation(
                raw_mask,
                initial_closing_kernel_size=INITIAL_PRE_CLOSING_KERNEL_SIZE,
                solidity_threshold=SOLIDITY_THRESHOLD,
                outline_fill_closing_kernel_size=OUTLINE_FILL_CLOSING_KERNEL_SIZE,
                outline_erosion_kernel_size=OUTLINE_EROSION_KERNEL_SIZE,
                filled_region_hole_closing_kernel_size=FILLED_REGION_HOLE_CLOSING_KERNEL_SIZE,
                min_contour_area=MIN_CONTOUR_AREA_FILTER
            )
            if processed_mask_step1 is None: # Ensure it's an array for create_consensus_mask
                processed_mask_step1 = np.zeros(MODEL_INPUT_SIZE, dtype=np.uint8)
            base_processed_masks.append(processed_mask_step1)
        
        if not base_processed_masks: # Should not happen if get_masks_for_image returns empty arrays for failures
            continue

        # Create_consensus_mask now handles the post_filter internally and empty checks
        final_consensus = utils.create_consensus_mask(
            base_processed_masks,
            filter_type=CONSENSUS_POST_FILTER_TYPE,
            filter_kernel_size=CONSENSUS_POST_FILTER_KERNEL_SIZE,
            consensus_method=CONSENSUS_METHOD
        )

        if final_consensus is not None and final_consensus.sum() > 0:
            consensus_masks_dict[image_name] = final_consensus
            processed_count +=1
        # else:
            # print(f"  Skipping {image_name}: empty consensus mask.")
    print(f"Generated {processed_count} non-empty consensus masks for evaluation.")
    return consensus_masks_dict

# --- Main Evaluation Logic ---
def main():
    device = get_device()
    print(f"Using device: {device}")

    # 1. Load and Filter Annotations Metadata (excluding test annotators)
    try:
        with open(ANNOTATIONS_METADATA_PATH, 'r') as f:
            annotations_metadata_raw = json.load(f)
        df_metadata_raw = pd.DataFrame(annotations_metadata_raw)
        df_metadata_filtered = df_metadata_raw[~df_metadata_raw['annotator_name'].str.contains('test', case=False, na=False)].copy()
        annotations_metadata_list_filtered = df_metadata_filtered.to_dict(orient='records')
        if not annotations_metadata_list_filtered:
            print("No annotations found after filtering test users. Exiting.")
            return
        print(f"Loaded and filtered annotations: {len(annotations_metadata_list_filtered)} records remaining.")
    except Exception as e:
        print(f"Error loading or filtering metadata: {e}")
        return

    # 2. Get Expert Consensus Masks for images to be evaluated
    # This dictionary will contain {image_filename: consensus_mask_np}
    # Only images with non-empty consensus masks will be included.
    expert_consensus_masks = get_consensus_masks_for_evaluation(
        annotations_metadata_list_filtered,
        ANNOTATED_MASKS_DIR
    )

    if not expert_consensus_masks:
        print("No non-empty expert consensus masks could be generated. Exiting.")
        return

    evaluation_images = list(expert_consensus_masks.keys())
    print(f"\nStarting saliency evaluation for {len(evaluation_images)} images with non-empty consensus masks.")

    # 3. Define Models and Saliency Methods
    # Model keys should roughly match parts of checkpoint filenames
    model_configs = {
        "AlexNet": {"class": AlexNetBinary, "key_name": "an", "ckpt_path": None},
        "VGG": {"class": VGG16Binary, "key_name": "vgg", "ckpt_path": None},
        "ResNet": {"class": ResNet50Binary, "key_name": "rn", "ckpt_path": None}, # Assuming ResNet50
        "InceptionNet": {"class": InceptionNetBinary, "key_name": "in", "ckpt_path": None},
    }
    
    # Find checkpoints
    for model_name, config in model_configs.items():
        config["ckpt_path"] = find_checkpoint(config["key_name"])
        if not config["ckpt_path"]:
            print(f"Could not find checkpoint for {model_name}, it will be skipped.")

    saliency_methods = ["CAM", "GradCAM", "RISE", "Random"]
    results_data = [] # To store dicts for DataFrame

    # 4. Perform Evaluation
    for model_display_name, config in model_configs.items():
        if not config["ckpt_path"]:
            for sm_name in saliency_methods:
                results_data.append({"Model": model_display_name, "SaliencyMethod": sm_name, "AvgIoU": np.nan})
            continue

        print(f"\nEvaluating Model: {model_display_name} using checkpoint: {config['ckpt_path']}")
        try:
            # Load the PyTorch Lightning model wrapper
            pl_model_wrapper = BaseCNN.load_from_checkpoint(config["ckpt_path"], map_location=device)
            # Get the actual underlying model (e.g., AlexNetBinary instance)
            actual_model = pl_model_wrapper.model 
            actual_model.to(device).eval()
        except Exception as e:
            print(f"  Error loading model {model_display_name}: {e}. Skipping.")
            for sm_name in saliency_methods:
                results_data.append({"Model": model_display_name, "SaliencyMethod": sm_name, "AvgIoU": np.nan})
            continue

        # Initialize saliency tools for this model
        saliency_tools = {}
        try:
            if hasattr(saliency, 'CAM') and (isinstance(actual_model, AlexNetBinary) or isinstance(actual_model, VGG16Binary) or isinstance(actual_model, ResNet50Binary) or isinstance(actual_model, InceptionNetBinary)): # Check if model is compatible
                 saliency_tools["CAM"] = saliency.CAM(actual_model)
            else: print(f"  CAM not supported or saliency.CAM not found for {model_display_name}")
            if hasattr(saliency, 'GradCAM'): saliency_tools["GradCAM"] = saliency.GradCAM(actual_model)
            else: print(f"  saliency.GradCAM not found for {model_display_name}")
            if hasattr(saliency, 'RISE'): saliency_tools["RISE"] = saliency.RISE(actual_model, num_masks=500, scale_factor=16) # Adjusted RISE params for speed
            else: print(f"  saliency.RISE not found for {model_display_name}")
        except Exception as e:
            print(f"  Error initializing saliency tools for {model_display_name}: {e}")
            # Continue, some tools might still work or we use NaN

        for sm_name in saliency_methods:
            ious_for_current_pair = []
            print(f"  Processing Saliency Method: {sm_name}")

            for image_idx, image_filename in enumerate(evaluation_images):
                # print(f"    Image {image_idx+1}/{len(evaluation_images)}: {image_filename}") # Verbose
                
                # Ensure original image path is correct.
                # Assuming image_filename is like 'personX_bacteria_Y.jpeg' and it's in ORIGINAL_IMAGES_DIR_FOR_SALIENCY
                # For example, if ORIGINAL_IMAGES_DIR_FOR_SALIENCY = "data/test/NORMAL" or "data/test/PNEUMONIA"
                # We need to find the full path.
                # For simplicity, let's assume evaluation_images are just filenames and we search for them.
                # This part might need adjustment based on your exact image storage for the 50 test images.
                
                # Try to find the image in common pneumonia/normal test subdirs
                possible_paths = [
                    os.path.join(ORIGINAL_IMAGES_DIR_FOR_SALIENCY, "PNEUMONIA", image_filename),
                    os.path.join(ORIGINAL_IMAGES_DIR_FOR_SALIENCY, "NORMAL", image_filename),
                    os.path.join(ORIGINAL_IMAGES_DIR_FOR_SALIENCY, image_filename) # If it's directly in the folder
                ]
                image_path_for_saliency = None
                for p_path in possible_paths:
                    if os.path.exists(p_path):
                        image_path_for_saliency = p_path
                        break
                
                if not image_path_for_saliency:
                    # print(f"    Skipping image {image_filename}: Original file not found in common test locations.")
                    continue

                input_tensor = load_image_tensor(image_path_for_saliency, device)
                if input_tensor is None:
                    continue

                expert_mask_np = expert_consensus_masks[image_filename] # Already (224,224) and binary

                saliency_map_np = None
                if sm_name == "Random":
                    saliency_map_np = generate_random_map(size=MODEL_INPUT_SIZE)
                elif sm_name in saliency_tools:
                    try:
                        saliency_map_np = saliency_tools[sm_name](input_tensor)
                    except Exception as e:
                        print(f"    Error generating {sm_name} for {image_filename} with {model_display_name}: {e}")
                        saliency_map_np = None # Ensure it's None if error
                
                if saliency_map_np is not None:
                    binarized_saliency_map = binarize_saliency_map(saliency_map_np, threshold=SALIENCY_BINARIZATION_THRESHOLD)
                    if binarized_saliency_map is not None:
                        iou = utils.calculate_iou(binarized_saliency_map, expert_mask_np)
                        ious_for_current_pair.append(iou)
                    # else: print(f"    Binarized saliency map is None for {image_filename}, {sm_name}")
                # else: print(f"    Saliency map is None for {image_filename}, {sm_name}")


            avg_iou = np.mean(ious_for_current_pair) if ious_for_current_pair else np.nan
            results_data.append({"Model": model_display_name, "SaliencyMethod": sm_name, "AvgIoU": avg_iou})
            print(f"    {model_display_name} - {sm_name}: Avg IoU = {avg_iou:.4f} (over {len(ious_for_current_pair)} images)")

    # 5. Create and Print Results Table
    results_df = pd.DataFrame(results_data)
    pivot_table = results_df.pivot(index="Model", columns="SaliencyMethod", values="AvgIoU")
    # Ensure correct column order
    pivot_table = pivot_table.reindex(columns=saliency_methods, fill_value=np.nan) 
    
    print("\n--- Model-Experts Agreement (IoU) ---")
    print(pivot_table.to_string(float_format="%.4f"))

    # Optionally save to CSV
    pivot_table.to_csv("saliency_iou_results.csv")
    print("\nResults saved to saliency_iou_results.csv")

if __name__ == "__main__":
    # Example: You might want to add argparse later to specify checkpoint dir, etc.
    main()
