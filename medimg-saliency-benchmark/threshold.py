import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import importlib
from torchvision import transforms
import cv2
import torch
import torch.nn as nn

BASE_ANNOTATIONS_DIR = "data/annotations"
METADATA_PATH = os.path.join(BASE_ANNOTATIONS_DIR, "metadata.json")
ANNOTATED_MASKS_DIR = os.path.join(BASE_ANNOTATIONS_DIR, "annotated")
ORIGINAL_IMAGES_DIR = "data/test"

TARGET_MASK_SIZE = (224, 224)

CIRCLE_CLOSING_KERNEL_SIZE = 7 
CIRCLE_EROSION_KERNEL_SIZE = 7 
MIN_CONTOUR_AREA_FILTER = 20   

INITIAL_PRE_CLOSING_KERNEL_SIZE = 3  # For initial small gap closing (0 to disable)
SOLIDITY_THRESHOLD = 0.6             # Solidity < threshold means it's an OUTLINE. Range (0.0 to 1.0)
                                     # Lower for thinner/more broken outlines, Higher if outlines are quite solid.
OUTLINE_FILL_CLOSING_KERNEL_SIZE = 7 # To close gaps in detected outlines before filling (0 to disable)
OUTLINE_EROSION_KERNEL_SIZE = 7      # To "remove" line thickness from filled OUTLINES (0 to disable)
FILLED_REGION_HOLE_CLOSING_KERNEL_SIZE = 5 # To close internal holes in detected FILLED regions (0 to disable)
MIN_CONTOUR_AREA_FILTER = 20         # Minimum pixel area to consider a contour significant

CONSENSUS_POST_FILTER_TYPE = 'open' # Filter applied to individual processed masks before consensus
CONSENSUS_POST_FILTER_KERNEL_SIZE = 3
CONSENSUS_METHOD = 'intersection'

with open(METADATA_PATH, 'r') as f:
        annotations_metadata_raw = json.load(f)

df_metadata_raw = pd.DataFrame(annotations_metadata_raw)
df_metadata = df_metadata_raw[~df_metadata_raw['annotator_name'].str.contains('test', case=False, na=False)].copy()
# Create a filtered list of dictionaries for functions that expect list format
annotations_metadata = df_metadata.to_dict(orient='records')

import utils 
from models import BaseCNN, AlexNetBinary, VGG16Binary, ResNet50Binary, InceptionNetBinary
import saliency

# RISE parameters (if chosen)
RISE_NUM_MASKS_VIS = 200 # Fewer masks for faster visualization in notebook
RISE_SCALE_FACTOR_VIS = 16
MODEL_SHORT_KEY_FOR_SALIENCY = "vgg" # Example: AlexNet

# Path to original images (ensure this aligns with where your *test* set images are,
# as annotations are typically on test/evaluation set images)
# This should be the directory containing PNEUMONIA/NORMAL subfolders or images directly
VIS_ORIGINAL_IMAGES_DIR = "data/test" # Or "data/annotations/original/" if those are the ones to use

# Get device
device = utils.get_device() if 'utils' in sys.modules and hasattr(utils, 'get_device') else torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for saliency visualization: {device}")

# Helper to load a specific model checkpoint
# Assumes CHECKPOINT_DIR is defined (e.g. "./checkpoints")
if 'CHECKPOINT_DIR' not in globals(): CHECKPOINT_DIR = "./checkpoints"

loaded_saliency_model = None
saliency_model_name_display = "Unknown"

if MODEL_SHORT_KEY_FOR_SALIENCY:
    checkpoint_path_vis = utils.find_checkpoint(MODEL_SHORT_KEY_FOR_SALIENCY) # Use the revised find_checkpoint
    if checkpoint_path_vis:
        try:
            print(f"Loading model for saliency from: {checkpoint_path_vis}")
            pl_model_wrapper_vis = BaseCNN.load_from_checkpoint(checkpoint_path_vis, map_location=device)
            loaded_saliency_model = pl_model_wrapper_vis.model
            loaded_saliency_model.to(device).eval()
            saliency_model_name_display = f"{MODEL_SHORT_KEY_FOR_SALIENCY.upper()} model"
            print(f"Successfully loaded {saliency_model_name_display}")
        except Exception as e:
            print(f"Error loading model {MODEL_SHORT_KEY_FOR_SALIENCY} from {checkpoint_path_vis}: {e}")
            loaded_saliency_model = None


MODEL_KEY_FOR_THRESHOLD_PLOT = "vgg"  # Options: "an", "vgg", "rn", "in"
SALIENCY_METHOD_FOR_THRESHOLD_PLOT = "CAM" # Options: "CAM", "GradCAM", "RISE"
# (0.90 - 0.02) / 0.02 + 1 = 0.88 / 0.02 + 1 = 44 + 1 = 45 points
THRESHOLDS_TO_TEST = np.linspace(0.02, 0.90, 45) 
MODEL_INPUT_SIZE = (224, 224) 
#MODEL_SHORT_KEY_FOR_SALIENCY = "vgg" # Example: AlexNet
#SALIENCY_METHOD_TO_VISUALIZE = "CAM" # Options: "CAM", "GradCAM", "RISE", "Random"

OUTPUT_CSV_DIR_THRESH_PLOT = "evaluation" # Relative to notebook location
# Ensure VIS_ORIGINAL_IMAGES_DIR is defined (e.g., from Cell 6 or setup)
# Ensure device is defined

# --- Load Selected Model ---
# This section assumes CHECKPOINT_DIR, BaseCNN, utils.find_checkpoint, device are available
loaded_model_for_thresh_plot = None
model_name_for_thresh_plot_title = MODEL_KEY_FOR_THRESHOLD_PLOT.upper()

if 'CHECKPOINT_DIR' not in globals(): CHECKPOINT_DIR = "./checkpoints" # Define if not already
if 'device' not in globals(): device = torch.device("cpu") # Fallback device

checkpoint_path_thresh = utils.find_checkpoint(MODEL_KEY_FOR_THRESHOLD_PLOT)
if checkpoint_path_thresh:
    try:
        print(f"Loading model {MODEL_KEY_FOR_THRESHOLD_PLOT} from {checkpoint_path_thresh} for threshold analysis...")
        pl_model_wrapper_thresh = BaseCNN.load_from_checkpoint(checkpoint_path_thresh, map_location=device)
        loaded_model_for_thresh_plot = pl_model_wrapper_thresh.model
        loaded_model_for_thresh_plot.to(device).eval()
        print(f"Successfully loaded {model_name_for_thresh_plot_title} model.")
    except Exception as e:
        print(f"Error loading model {MODEL_KEY_FOR_THRESHOLD_PLOT}: {e}")
        loaded_model_for_thresh_plot = None 
else:
    print(f"No checkpoint found for {MODEL_KEY_FOR_THRESHOLD_PLOT}. Skipping threshold analysis.")
    loaded_model_for_thresh_plot = None 

# --- Main Loop for Threshold Analysis ---
print(f"\n--- IoU vs. Binarization Threshold for {model_name_for_thresh_plot_title} using {SALIENCY_METHOD_FOR_THRESHOLD_PLOT} ---")

# 1. Pre-calculate all necessary expert masks (Full Consensus, Giovanni, No Giovanni)
#    The cache will store masks for ALL unique images that have ANY annotations.
expert_masks_cache_thresh = {} 

images_for_thresh_analysis_all = sorted(list(set(record['image_name'] for record in annotations_metadata)))

print(f"Pre-calculating all expert mask types for {len(images_for_thresh_analysis_all)} images...")
for image_name in images_for_thresh_analysis_all:
    raw_masks_tuples = utils.get_masks_for_image_from_metadata(
        image_name, annotations_metadata, ANNOTATED_MASKS_DIR, target_size=MODEL_INPUT_SIZE
    )
    if not raw_masks_tuples: continue 

    current_image_cache = {}
    processed_masks_list_full = []
    for rm, an in raw_masks_tuples:
        processed_mask = utils.process_circled_annotation(rm, INITIAL_PRE_CLOSING_KERNEL_SIZE, SOLIDITY_THRESHOLD, OUTLINE_FILL_CLOSING_KERNEL_SIZE, OUTLINE_EROSION_KERNEL_SIZE, FILLED_REGION_HOLE_CLOSING_KERNEL_SIZE, MIN_CONTOUR_AREA_FILTER)
        # Ensure processed_mask is an array, even if None was returned
        processed_masks_list_full.append(processed_mask if processed_mask is not None else np.zeros(MODEL_INPUT_SIZE, dtype=np.uint8))
    
    # Corrected line for 'full' consensus caching:
    consensus_full_result = utils.create_consensus_mask(processed_masks_list_full, CONSENSUS_POST_FILTER_TYPE, CONSENSUS_POST_FILTER_KERNEL_SIZE, CONSENSUS_METHOD)
    current_image_cache['full'] = consensus_full_result if consensus_full_result is not None else np.zeros(MODEL_INPUT_SIZE, dtype=np.uint8)


    giovanni_mask = np.zeros(MODEL_INPUT_SIZE, dtype=np.uint8)
    for rm, an in raw_masks_tuples: 
        if "giovanni" in an.lower():
            processed_gio = utils.process_circled_annotation(rm, INITIAL_PRE_CLOSING_KERNEL_SIZE, SOLIDITY_THRESHOLD, OUTLINE_FILL_CLOSING_KERNEL_SIZE, OUTLINE_EROSION_KERNEL_SIZE, FILLED_REGION_HOLE_CLOSING_KERNEL_SIZE, MIN_CONTOUR_AREA_FILTER)
            if processed_gio is not None and processed_gio.sum() > 0: 
                giovanni_mask = processed_gio
            break 
    current_image_cache['gio'] = giovanni_mask
    
    base_processed_no_gio = []
    annotators_other_than_giovanni_present = False
    for rm, an in raw_masks_tuples: 
        if "giovanni" not in an.lower():
            annotators_other_than_giovanni_present = True
            processed_mask_no_gio = utils.process_circled_annotation(rm, INITIAL_PRE_CLOSING_KERNEL_SIZE, SOLIDITY_THRESHOLD, OUTLINE_FILL_CLOSING_KERNEL_SIZE, OUTLINE_EROSION_KERNEL_SIZE, FILLED_REGION_HOLE_CLOSING_KERNEL_SIZE, MIN_CONTOUR_AREA_FILTER)
            base_processed_no_gio.append(processed_mask_no_gio if processed_mask_no_gio is not None else np.zeros(MODEL_INPUT_SIZE, dtype=np.uint8))

    consensus_no_gio = np.zeros(MODEL_INPUT_SIZE, dtype=np.uint8)
    if annotators_other_than_giovanni_present and base_processed_no_gio: 
        temp_consensus_no_gio = utils.create_consensus_mask(base_processed_no_gio, CONSENSUS_POST_FILTER_TYPE, CONSENSUS_POST_FILTER_KERNEL_SIZE, CONSENSUS_METHOD)
        if temp_consensus_no_gio is not None: 
            consensus_no_gio = temp_consensus_no_gio
    current_image_cache['no_gio'] = consensus_no_gio
    
    expert_masks_cache_thresh[image_name] = current_image_cache # Cache all three mask types for every image that had annotations

print(f"Cached all expert mask types for {len(expert_masks_cache_thresh)} images.")

# 2. Initialize Saliency Tool
saliency_tool_thresh = None
if SALIENCY_METHOD_FOR_THRESHOLD_PLOT == "CAM":
    if isinstance(loaded_model_for_thresh_plot, (AlexNetBinary, VGG16Binary, ResNet50Binary, InceptionNetBinary)):
        # Check if VGG16Binary with non-linear classifier needs special handling
        check_cam_compatibility(loaded_model_for_thresh_plot)
        try:
            saliency_tool_thresh = saliency.CAM(loaded_model_for_thresh_plot)
            print(f"Successfully initialized {SALIENCY_METHOD_FOR_THRESHOLD_PLOT} for {type(loaded_model_for_thresh_plot).__name__}")
        except Exception as e:
            print(f"Error initializing CAM: {e}")
            print("Falling back to GradCAM which works with all model architectures")
            SALIENCY_METHOD_FOR_THRESHOLD_PLOT = "GradCAM"
            saliency_tool_thresh = saliency.GradCAM(loaded_model_for_thresh_plot)
elif SALIENCY_METHOD_FOR_THRESHOLD_PLOT == "GradCAM":
    saliency_tool_thresh = saliency.GradCAM(loaded_model_for_thresh_plot)
elif SALIENCY_METHOD_FOR_THRESHOLD_PLOT == "RISE":
    rise_masks = 200
    rise_scale = 16
    if hasattr(saliency, 'RISE'): saliency_tool_thresh = saliency.RISE(loaded_model_for_thresh_plot, num_masks=rise_masks, scale_factor=rise_scale)

if not saliency_tool_thresh:
    print(f"Could not initialize saliency tool {SALIENCY_METHOD_FOR_THRESHOLD_PLOT}. Aborting threshold analysis.")
else:
    results_for_csv = [] 

    for threshold_val in THRESHOLDS_TO_TEST: 
        print(f"  Testing threshold: {threshold_val:.2f}")
        current_thresh_ious_full = []
        current_thresh_ious_gio = []
        current_thresh_ious_no_gio = []

        # Iterate over ALL images for which we cached masks
        for image_name, masks in expert_masks_cache_thresh.items():
            original_image_path_thresh = None
            current_vis_original_images_dir_thresh = VIS_ORIGINAL_IMAGES_DIR if 'VIS_ORIGINAL_IMAGES_DIR' in locals() else "data/test"
            possible_paths_thresh = [
                os.path.join(current_vis_original_images_dir_thresh, "PNEUMONIA", image_name),
                os.path.join(current_vis_original_images_dir_thresh, "NORMAL", image_name),
                os.path.join(current_vis_original_images_dir_thresh, image_name) 
            ]
            for p_path_thresh in possible_paths_thresh:
                if os.path.exists(p_path_thresh): original_image_path_thresh = p_path_thresh; break
            
            if not original_image_path_thresh: continue
            input_tensor_thresh = utils.load_image_tensor(original_image_path_thresh, device)
            if input_tensor_thresh is None: continue

            try:
                saliency_map_np = saliency_tool_thresh(input_tensor_thresh)
            except Exception as e:
                continue 
            
            if saliency_map_np is None: continue

            bin_saliency = utils.binarize_saliency_map(saliency_map_np, threshold_value=threshold_val) 
            if bin_saliency is None: continue

            # Calculate IoU if the respective expert mask is non-empty
            if masks['full'].sum() > 0:
                current_thresh_ious_full.append(utils.calculate_iou(bin_saliency, masks['full']))
            if masks['gio'].sum() > 0:
                current_thresh_ious_gio.append(utils.calculate_iou(bin_saliency, masks['gio']))
            if masks['no_gio'].sum() > 0:
                current_thresh_ious_no_gio.append(utils.calculate_iou(bin_saliency, masks['no_gio']))

        avg_iou_full = np.nanmean(current_thresh_ious_full) if current_thresh_ious_full else np.nan
        avg_iou_gio = np.nanmean(current_thresh_ious_gio) if current_thresh_ious_gio else np.nan
        avg_iou_no_gio = np.nanmean(current_thresh_ious_no_gio) if current_thresh_ious_no_gio else np.nan
        
        results_for_csv.append({
            'threshold': threshold_val,
            'iou_vs_full_consensus': avg_iou_full,
            'iou_vs_giovanni_only': avg_iou_gio,
            'iou_vs_consensus_no_giovanni': avg_iou_no_gio,
            'n_full_consensus': len(current_thresh_ious_full), 
            'n_giovanni': len(current_thresh_ious_gio),       
            'n_no_giovanni': len(current_thresh_ious_no_gio)  
        })
        print(f"    Avg IoUs for threshold {threshold_val:.2f}: Full={avg_iou_full:.4f} (n={len(current_thresh_ious_full)}), Gio={avg_iou_gio:.4f} (n={len(current_thresh_ious_gio)}), NoGio={avg_iou_no_gio:.4f} (n={len(current_thresh_ious_no_gio)})")

    df_thresh_results = pd.DataFrame(results_for_csv)

    plt.figure(figsize=(12, 7)) 
    plt.plot(df_thresh_results['threshold'], df_thresh_results['iou_vs_full_consensus'], marker='o', label=f'IoU vs Full Consensus (n={df_thresh_results["n_full_consensus"].iloc[-1] if not df_thresh_results.empty and "n_full_consensus" in df_thresh_results.columns else 0})')
    plt.plot(df_thresh_results['threshold'], df_thresh_results['iou_vs_giovanni_only'], marker='s', label=f'IoU vs Giovanni Only (n={df_thresh_results["n_giovanni"].iloc[-1] if not df_thresh_results.empty and "n_giovanni" in df_thresh_results.columns else 0})')
    plt.plot(df_thresh_results['threshold'], df_thresh_results['iou_vs_consensus_no_giovanni'], marker='^', label=f'IoU vs Consensus (No Gio) (n={df_thresh_results["n_no_giovanni"].iloc[-1] if not df_thresh_results.empty and "n_no_giovanni" in df_thresh_results.columns else 0})')
    
    plt.xlabel("Saliency Binarization Threshold")
    plt.ylabel("Average IoU")
    plt.title(f"IoU vs. Binarization Threshold ({SALIENCY_METHOD_FOR_THRESHOLD_PLOT} on {model_name_for_thresh_plot_title})")
    plt.legend()
    plt.grid(True)
    # Ensure all tested thresholds are marked, round for display if many points
    if len(THRESHOLDS_TO_TEST) <= 15:
            plt.xticks(THRESHOLDS_TO_TEST)
    else: # If too many points, let matplotlib decide or set a specific number of ticks
            plt.xticks(np.linspace(min(THRESHOLDS_TO_TEST), max(THRESHOLDS_TO_TEST), 10))

    plt.ylim(bottom=0) 
    plt.show()

    os.makedirs(OUTPUT_CSV_DIR_THRESH_PLOT, exist_ok=True)
    csv_filename = f"iou_vs_threshold_{MODEL_KEY_FOR_THRESHOLD_PLOT}_{SALIENCY_METHOD_FOR_THRESHOLD_PLOT}.csv"
    csv_path = os.path.join(OUTPUT_CSV_DIR_THRESH_PLOT, csv_filename)
    df_thresh_results.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Threshold analysis results saved to: {csv_path}")

def check_cam_compatibility(model):
    """Check if the model is compatible with the CAM method and fix if needed.
    
    For VGG16Binary with linear=False, the classifier is Sequential and we need to fix
    the saliency.CAM implementation to access the weights correctly.
    """
    if isinstance(model, VGG16Binary) and not model.linear:
        # If classifier is Sequential with last layer being Linear, 
        # we need to access that specifically
        if isinstance(model.classifier, nn.Sequential):
            # Get the last layer of the sequential classifier
            last_layer = model.classifier[-1]
            if isinstance(last_layer, nn.Linear):
                # Monkey patch the model to expose the weight directly 
                model.classifier.weight = last_layer.weight
                return True
    return False

if __name__ == "__main__":
    main()