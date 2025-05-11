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

CONSENSUS_TYPE = "full" # this is only used for the output file name
RUN_NAME = "test" # this is only used for the output file name

CHECKPOINT_DIR = "./checkpoints"
ANNOTATIONS_METADATA_PATH = "data/annotations/metadata.json"
ANNOTATED_MASKS_DIR = "data/annotations/annotated"
ORIGINAL_IMAGES_DIR_FOR_SALIENCY = "data/test"

MODEL_INPUT_SIZE = (224, 224) 

SALIENCY_BINARIZATION_THRESHOLD = 0.74 # empirically found to be the best threshold for full consensus

# morphological filter parameters tuned empirically for full consensus, you can read about them in utils.py
INITIAL_PRE_CLOSING_KERNEL_SIZE = 3 
SOLIDITY_THRESHOLD = 0.6            
OUTLINE_FILL_CLOSING_KERNEL_SIZE = 7 
OUTLINE_EROSION_KERNEL_SIZE = 7      
FILLED_REGION_HOLE_CLOSING_KERNEL_SIZE = 5 
MIN_CONTOUR_AREA_FILTER = 20  
CONSENSUS_POST_FILTER_TYPE = 'open' 
CONSENSUS_POST_FILTER_KERNEL_SIZE = 3
CONSENSUS_METHOD = 'intersection'

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
        raw_masks_tuples = utils.get_masks_for_image_from_metadata(image_name,annotations_metadata_list, annotated_masks_dir, target_size=MODEL_INPUT_SIZE)

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
    device = utils.get_device()
    print(f"Using device: {device}")

    # 1. Load and Filter Annotations Metadata (excluding test annotators)
    with open(ANNOTATIONS_METADATA_PATH, 'r') as f:
        annotations_metadata_raw = json.load(f)
    df_metadata_raw = pd.DataFrame(annotations_metadata_raw)
    df_metadata_filtered = df_metadata_raw[~df_metadata_raw['annotator_name'].str.contains('test', case=False, na=False)].copy()
    annotations_metadata_list_filtered = df_metadata_filtered.to_dict(orient='records')
    print(f"Loaded and filtered annotations: {len(annotations_metadata_list_filtered)} records remaining.")

    # 2. Get Expert Consensus Masks for images to be evaluated
    # This dictionary will contain {image_filename: consensus_mask_np}
    # Only images with non-empty consensus masks will be included.
    expert_consensus_masks = get_consensus_masks_for_evaluation(
        annotations_metadata_list_filtered,
        ANNOTATED_MASKS_DIR
    )

    evaluation_images = list(expert_consensus_masks.keys())
    print(f"\nStarting saliency evaluation for {len(evaluation_images)} images with non-empty consensus masks.")

    saliency_methods = ["CAM", "GradCAM", "Random"] # RISE is not included because it's way too slow, but it works
    all_results_data = [] # To store dicts for DataFrame: {'model': 'an', 'linear': True, ... 'CAM': 0.1, ...}

    # Get all checkpoint files from the CHECKPOINT_DIR
    all_checkpoint_paths = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.ckpt")))
    print(f"\nFound {len(all_checkpoint_paths)} checkpoint files to process.")

    # 3. Perform Evaluation for each checkpoint
    for ckpt_idx, ckpt_path in enumerate(all_checkpoint_paths):
        print(f"\nProcessing checkpoint {ckpt_idx + 1}/{len(all_checkpoint_paths)}: {os.path.basename(ckpt_path)}")
        
        parsed_info = utils.parse_checkpoint_filename(os.path.basename(ckpt_path))

        current_config_results = {
            'model': parsed_info['model'],
            'linear': parsed_info['linear'],
            'pretrained': parsed_info['pretrained']
        }

        pl_model_wrapper = BaseCNN.load_from_checkpoint(ckpt_path, map_location=device)
        actual_model = pl_model_wrapper.model 
        actual_model.to(device).eval()

        for sm_name in saliency_methods:
            ious_for_current_saliency_method = []
            
            saliency_tool = None
            try:
                if sm_name == "CAM" and hasattr(saliency, 'CAM'):
                    saliency_tool = saliency.CAM(actual_model)
                elif sm_name == "GradCAM" and hasattr(saliency, 'GradCAM'):
                    saliency_tool = saliency.GradCAM(actual_model)
                elif sm_name == "RISE" and hasattr(saliency, 'RISE'):
                    saliency_tool = saliency.RISE(actual_model, num_masks=500, scale_factor=16)
            except Exception as e:
                print(f"  Warning: Error initializing {sm_name} for {ckpt_path}: {e}")

            for image_filename in evaluation_images: # this is the only loop that runs for each image
                possible_paths = [ # it checks if the image is present in the PNEUMONIA or NORMAL folder
                    os.path.join(ORIGINAL_IMAGES_DIR_FOR_SALIENCY, "PNEUMONIA", image_filename),
                    os.path.join(ORIGINAL_IMAGES_DIR_FOR_SALIENCY, "NORMAL", image_filename),
                ]
                image_path_for_saliency = None
                for p_path in possible_paths:
                    if os.path.exists(p_path):
                        image_path_for_saliency = p_path
                        break
                if not image_path_for_saliency: continue
                input_tensor = utils.load_image_tensor(image_path_for_saliency, device)
                if input_tensor is None: continue
                expert_mask_np = expert_consensus_masks[image_filename]

                saliency_map_np = None
                if sm_name == "Random":
                    saliency_map_np = utils.generate_random_map(size=MODEL_INPUT_SIZE)
                elif saliency_tool is not None:
                    saliency_map_np = saliency_tool(input_tensor)
                
                if saliency_map_np is not None:
                    binarized_saliency_map = utils.binarize_saliency_map(saliency_map_np, method="fixed", threshold_value=SALIENCY_BINARIZATION_THRESHOLD)
                    if binarized_saliency_map is not None:
                        iou = utils.calculate_iou(binarized_saliency_map, expert_mask_np)
                        ious_for_current_saliency_method.append(iou)
            
            if saliency_tool and hasattr(saliency_tool, 'remove_hook'):
                saliency_tool.remove_hook()
                
            # Ensure we have a valid value for the saliency method (even if it's 0)
            avg_iou = np.mean(ious_for_current_saliency_method) if ious_for_current_saliency_method else 0.0
            current_config_results[sm_name] = avg_iou
        
        all_results_data.append(current_config_results)
        print(f"  Finished {os.path.basename(ckpt_path)}. Avg IoUs: CAM={current_config_results.get('CAM', float('nan')):.4f}, GradCAM={current_config_results.get('GradCAM', float('nan')):.4f}, Random={current_config_results.get('Random', float('nan')):.4f}")

    # 4. Create and Save Results Table (Granular)
    results_df = pd.DataFrame(all_results_data)

    # Define column order for the output CSV
    output_columns = ['model', 'linear', 'pretrained'] + \
                    [sm for sm in saliency_methods if sm in results_df.columns]
    # Add any saliency methods that might have all NaNs but should still be columns
    for sm_col in saliency_methods:
        if sm_col not in results_df.columns:
            results_df[sm_col] = np.nan # Add column with NaNs if it wasn't processed
    results_df = results_df[output_columns] # Reorder/select columns

    print("\n--- Granular Model-Experts Agreement (IoU) ---")
    print(results_df.to_string(index=False, float_format="%.4f"))

    csv_path_0 = f"saliency_iou_results_{CONSENSUS_TYPE}_{RUN_NAME}_{SALIENCY_BINARIZATION_THRESHOLD}.csv"
    csv_output_path = os.path.join("evaluation", csv_path_0)
    results_df.to_csv(csv_output_path, index=False, float_format="%.4f")
    print(f"\nGranular results saved to {csv_output_path}")

if __name__ == "__main__":
    main()