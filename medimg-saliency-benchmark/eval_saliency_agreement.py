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

CONSENSUS_METHOD = "full"
RUN_NAME = "test"

CHECKPOINT_DIR = "./checkpoints"
ANNOTATIONS_METADATA_PATH = "data/annotations/metadata.json"
ANNOTATED_MASKS_DIR = "data/annotations/annotated"
ORIGINAL_IMAGES_DIR_FOR_SALIENCY = "data/test" # Or wherever the original images for saliency evaluation are

MODEL_INPUT_SIZE = (224, 224) 

SALIENCY_BINARIZATION_THRESHOLD = 0.74

INITIAL_PRE_CLOSING_KERNEL_SIZE = 3
SOLIDITY_THRESHOLD = 0.6            
OUTLINE_FILL_CLOSING_KERNEL_SIZE = 7 
OUTLINE_EROSION_KERNEL_SIZE = 7      
FILLED_REGION_HOLE_CLOSING_KERNEL_SIZE = 5 
MIN_CONTOUR_AREA_FILTER = 20  
CONSENSUS_POST_FILTER_TYPE = 'open' # Filter applied to individual processed masks before consensus
CONSENSUS_POST_FILTER_KERNEL_SIZE = 3
CONSENSUS_METHOD = 'intersection'

def get_device():
    """Gets the appropriate torch device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_consensus_masks_for_evaluation(annotations_metadata_list, annotated_masks_dir):
    """
    Generates final consensus masks for all images that have annotations.
    Returns a dictionary: {image_filename: consensus_mask_np}
    Only includes images where the final consensus mask is non-empty.
    """
    consensus_masks_dict = {}
    unique_image_names = sorted(list(set(record['image_name'] for record in annotations_metadata_list)))
    
    print(f"\nRunning {CONSENSUS_METHOD} consensus with threshold {SALIENCY_BINARIZATION_THRESHOLD} from {RUN_NAME} mode.")
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
    print(f"Generated {processed_count} non-empty consensus masks for evaluation.")
    return consensus_masks_dict

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
        config["ckpt_path"] = utils.find_checkpoint(config["key_name"])
        if not config["ckpt_path"]:
            print(f"Could not find checkpoint for {model_name}, it will be skipped.")

    saliency_methods = ["CAM", "GradCAM", "Random"]
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

                input_tensor = utils.load_image_tensor(image_path_for_saliency, device)
                if input_tensor is None:
                    continue

                expert_mask_np = expert_consensus_masks[image_filename] # Already (224,224) and binary

                saliency_map_np = None
                if sm_name == "Random":
                    saliency_map_np = utils.generate_random_map(size=MODEL_INPUT_SIZE)
                elif sm_name in saliency_tools:
                    try:
                        saliency_map_np = saliency_tools[sm_name](input_tensor)
                    except Exception as e:
                        print(f"    Error generating {sm_name} for {image_filename} with {model_display_name}: {e}")
                        saliency_map_np = None # Ensure it's None if error
                
                if saliency_map_np is not None:
                    binarized_saliency_map = utils.binarize_saliency_map(saliency_map_np, method="fixed") # New
                    if binarized_saliency_map is not None:
                        iou = utils.calculate_iou(binarized_saliency_map, expert_mask_np)
                        ious_for_current_pair.append(iou)

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

    csv_name = f"evaluation/saliency_iou_results_{CONSENSUS_METHOD}_{RUN_NAME}_{SALIENCY_BINARIZATION_THRESHOLD}"
    pivot_table.to_csv(csv_name)
    print(f"\nResults saved to {csv_name}")

if __name__ == "__main__":
    main()