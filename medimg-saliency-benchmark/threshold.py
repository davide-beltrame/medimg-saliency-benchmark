import os
import sys
import glob
import json
import argparse
import torch
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# --- Setup Project Path ---
# Assuming threshold.py is in the project root (medimg-saliency-benchmark/)
# and custom modules are in medimg-saliency-benchmark/medimg-saliency-benchmark/
module_path = os.path.abspath(os.path.join('.', 'medimg-saliency-benchmark'))
if module_path not in sys.path:
    sys.path.append(module_path)

try:
    import utils # Your utils.py
    from models import BaseCNN, AlexNetBinary, VGG16Binary, ResNet50Binary, InceptionNetBinary
    import saliency # Your saliency.py
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure 'threshold.py' is in the project root and the "
          "'medimg-saliency-benchmark/medimg-saliency-benchmark/' directory is in sys.path or accessible.")
    sys.exit(1)

# --- Script Configuration (Defaults & Constants from Notebook Context) ---
CHECKPOINT_DIR = "./checkpoints"
ANNOTATIONS_METADATA_PATH = "data/annotations/metadata.json"
ANNOTATED_MASKS_DIR = "data/annotations/annotated"
ORIGINAL_IMAGES_DIR = "data/test" # Directory for original images for saliency

MODEL_INPUT_SIZE = (224, 224)

# Annotation Processing Parameters (from notebook context)
INITIAL_PRE_CLOSING_KERNEL_SIZE = 3
SOLIDITY_THRESHOLD = 0.6
OUTLINE_FILL_CLOSING_KERNEL_SIZE = 7
OUTLINE_EROSION_KERNEL_SIZE = 7
FILLED_REGION_HOLE_CLOSING_KERNEL_SIZE = 5
MIN_CONTOUR_AREA_FILTER = 20
CONSENSUS_POST_FILTER_TYPE = 'open'
CONSENSUS_POST_FILTER_KERNEL_SIZE = 3
CONSENSUS_METHOD = 'intersection'

OUTPUT_CSV_DIR = "evaluation"
OUTPUT_PLOT_DIR = "plots" # For saving the threshold plot

# --- Helper to get device ---
def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def main(args):
    device = get_device()
    print(f"Using device: {device}")

    # --- Configuration from Args ---
    model_key_for_plot = args.model_key
    saliency_method_for_plot = args.saliency_method
    threshold_step = args.threshold_step
    threshold_start = args.threshold_start
    threshold_end = args.threshold_end
    
    # Generate thresholds based on args
    # np.arange is exclusive for the end point, so add step to include it if desired
    num_steps = int(round((threshold_end - threshold_start) / threshold_step)) + 1
    thresholds_to_test = np.round(np.linspace(threshold_start, threshold_end, num_steps), 2)


    print(f"Model for analysis: {model_key_for_plot.upper()}")
    print(f"Saliency method: {saliency_method_for_plot}")
    print(f"Testing thresholds from {threshold_start} to {threshold_end} with step {threshold_step}")
    print(f"Thresholds: {thresholds_to_test}")


    # 1. Load and Filter Annotations Metadata
    try:
        with open(ANNOTATIONS_METADATA_PATH, 'r') as f:
            annotations_metadata_raw = json.load(f)
        df_metadata_raw = pd.DataFrame(annotations_metadata_raw)
        df_metadata_filtered = df_metadata_raw[~df_metadata_raw['annotator_name'].str.contains('test', case=False, na=False)].copy()
        annotations_metadata = df_metadata_filtered.to_dict(orient='records')
        if not annotations_metadata:
            print("No annotations found after filtering. Exiting.")
            return
        print(f"Loaded and filtered annotations: {len(annotations_metadata)} records remaining.")
    except Exception as e:
        print(f"Error loading or filtering metadata: {e}")
        return

    # 2. Load Selected Model
    loaded_model_for_plot = None
    model_name_for_plot_title = model_key_for_plot.upper()
    checkpoint_path_thresh = utils.find_checkpoint(model_key_for_plot) # Assumes CHECKPOINT_DIR is global in utils or here

    if checkpoint_path_thresh:
        try:
            print(f"Loading model {model_key_for_plot} from {checkpoint_path_thresh}...")
            pl_model_wrapper_thresh = BaseCNN.load_from_checkpoint(checkpoint_path_thresh, map_location=device)
            loaded_model_for_plot = pl_model_wrapper_thresh.model
            loaded_model_for_plot.to(device).eval()
            print(f"Successfully loaded {model_name_for_plot_title} model.")
        except Exception as e:
            print(f"Error loading model {model_key_for_plot}: {e}")
            return # Exit if model can't be loaded
    else:
        print(f"No checkpoint found for {model_key_for_plot}. Exiting.")
        return

    # 3. Pre-calculate all necessary expert masks
    expert_masks_cache = {}
    images_for_analysis_all = sorted(list(set(record['image_name'] for record in annotations_metadata)))
    print(f"Pre-calculating expert masks for {len(images_for_analysis_all)} images...")

    for image_name in images_for_analysis_all:
        raw_masks_tuples = utils.get_masks_for_image_from_metadata(
            image_name, annotations_metadata, ANNOTATED_MASKS_DIR, target_size=MODEL_INPUT_SIZE
        )
        if not raw_masks_tuples: continue

        current_image_cache = {}
        # Full Consensus
        processed_masks_list_full = [
            (utils.process_circled_annotation(rm, INITIAL_PRE_CLOSING_KERNEL_SIZE, SOLIDITY_THRESHOLD, OUTLINE_FILL_CLOSING_KERNEL_SIZE, OUTLINE_EROSION_KERNEL_SIZE, FILLED_REGION_HOLE_CLOSING_KERNEL_SIZE, MIN_CONTOUR_AREA_FILTER)
             if rm is not None else np.zeros(MODEL_INPUT_SIZE, dtype=np.uint8))
            for rm, an in raw_masks_tuples
        ]
        consensus_full_result = utils.create_consensus_mask(processed_masks_list_full, CONSENSUS_POST_FILTER_TYPE, CONSENSUS_POST_FILTER_KERNEL_SIZE, CONSENSUS_METHOD)
        current_image_cache['full'] = consensus_full_result if consensus_full_result is not None else np.zeros(MODEL_INPUT_SIZE, dtype=np.uint8)

        # Giovanni's mask
        giovanni_mask = np.zeros(MODEL_INPUT_SIZE, dtype=np.uint8)
        for rm, an in raw_masks_tuples:
            if "giovanni" in an.lower():
                processed_gio = utils.process_circled_annotation(rm, INITIAL_PRE_CLOSING_KERNEL_SIZE, SOLIDITY_THRESHOLD, OUTLINE_FILL_CLOSING_KERNEL_SIZE, OUTLINE_EROSION_KERNEL_SIZE, FILLED_REGION_HOLE_CLOSING_KERNEL_SIZE, MIN_CONTOUR_AREA_FILTER)
                if processed_gio is not None and processed_gio.sum() > 0: giovanni_mask = processed_gio
                break
        current_image_cache['gio'] = giovanni_mask
        
        # No Giovanni Consensus
        base_processed_no_gio = []
        annotators_no_gio_present = False
        for rm, an in raw_masks_tuples:
            if "giovanni" not in an.lower():
                annotators_no_gio_present = True
                processed_mask_no_gio = utils.process_circled_annotation(rm, INITIAL_PRE_CLOSING_KERNEL_SIZE, SOLIDITY_THRESHOLD, OUTLINE_FILL_CLOSING_KERNEL_SIZE, OUTLINE_EROSION_KERNEL_SIZE, FILLED_REGION_HOLE_CLOSING_KERNEL_SIZE, MIN_CONTOUR_AREA_FILTER)
                base_processed_no_gio.append(processed_mask_no_gio if processed_mask_no_gio is not None else np.zeros(MODEL_INPUT_SIZE, dtype=np.uint8))
        
        consensus_no_gio_result = np.zeros(MODEL_INPUT_SIZE, dtype=np.uint8)
        if annotators_no_gio_present and base_processed_no_gio:
            temp_no_gio = utils.create_consensus_mask(base_processed_no_gio, CONSENSUS_POST_FILTER_TYPE, CONSENSUS_POST_FILTER_KERNEL_SIZE, CONSENSUS_METHOD)
            if temp_no_gio is not None: consensus_no_gio_result = temp_no_gio
        current_image_cache['no_gio'] = consensus_no_gio_result
        
        expert_masks_cache[image_name] = current_image_cache
    print(f"Cached expert masks for {len(expert_masks_cache)} images.")

    # 4. Initialize Saliency Tool
    saliency_tool = None
    if saliency_method_for_plot == "CAM":
        if hasattr(saliency, 'CAM') and isinstance(loaded_model_for_plot, (AlexNetBinary, VGG16Binary, ResNet50Binary, InceptionNetBinary)):
            saliency_tool = saliency.CAM(loaded_model_for_plot)
    elif saliency_method_for_plot == "GradCAM":
        if hasattr(saliency, 'GradCAM'): saliency_tool = saliency.GradCAM(loaded_model_for_plot)
    elif saliency_method_for_plot == "RISE": # Assuming RISE params are fixed for this script or added to args
        if hasattr(saliency, 'RISE'): saliency_tool = saliency.RISE(loaded_model_for_plot, num_masks=200, scale_factor=16) # Example params

    if not saliency_tool:
        print(f"Could not initialize saliency tool {saliency_method_for_plot}. Aborting.")
        if hasattr(saliency_tool, 'remove_hook'): saliency_tool.remove_hook() # Clean up if partially initialized
        return

    # 5. Loop through thresholds and calculate IoUs
    results_for_csv = []
    print(f"\nCalculating IoUs for various thresholds...")
    for threshold_val in thresholds_to_test:
        print(f"  Testing threshold: {threshold_val:.2f}")
        ious_full, ious_gio, ious_no_gio = [], [], []

        for image_name, masks in expert_masks_cache.items():
            original_image_path = None
            possible_paths = [
                os.path.join(ORIGINAL_IMAGES_DIR, "PNEUMONIA", image_name),
                os.path.join(ORIGINAL_IMAGES_DIR, "NORMAL", image_name),
                os.path.join(ORIGINAL_IMAGES_DIR, image_name)
            ]
            for p_path in possible_paths:
                if os.path.exists(p_path): original_image_path = p_path; break
            
            if not original_image_path: continue
            input_tensor = utils.load_image_tensor(original_image_path, device)
            if input_tensor is None: continue

            try:
                saliency_map_np = saliency_tool(input_tensor)
            except Exception: continue # Skip if saliency generation fails
            if saliency_map_np is None: continue

            bin_saliency = utils.binarize_saliency_map(saliency_map_np, threshold_value=threshold_val)
            if bin_saliency is None: continue

            if masks['full'].sum() > 0: ious_full.append(utils.calculate_iou(bin_saliency, masks['full']))
            if masks['gio'].sum() > 0: ious_gio.append(utils.calculate_iou(bin_saliency, masks['gio']))
            if masks['no_gio'].sum() > 0: ious_no_gio.append(utils.calculate_iou(bin_saliency, masks['no_gio']))

        results_for_csv.append({
            'threshold': threshold_val,
            'iou_vs_full_consensus': np.nanmean(ious_full) if ious_full else np.nan,
            'iou_vs_giovanni_only': np.nanmean(ious_gio) if ious_gio else np.nan,
            'iou_vs_consensus_no_giovanni': np.nanmean(ious_no_gio) if ious_no_gio else np.nan,
            'n_full_consensus': len(ious_full),
            'n_giovanni': len(ious_gio),
            'n_no_giovanni': len(ious_no_gio)
        })
    
    if hasattr(saliency_tool, 'remove_hook'): saliency_tool.remove_hook() # Cleanup hook

    # 6. Create DataFrame, Plot, and Save CSV
    df_thresh_results = pd.DataFrame(results_for_csv)

    plt.figure(figsize=(12, 7))
    plt.plot(df_thresh_results['threshold'], df_thresh_results['iou_vs_full_consensus'], marker='o', label=f'IoU vs Full Consensus (n~{df_thresh_results["n_full_consensus"].mean():.0f})')
    plt.plot(df_thresh_results['threshold'], df_thresh_results['iou_vs_giovanni_only'], marker='s', label=f'IoU vs Giovanni Only (n~{df_thresh_results["n_giovanni"].mean():.0f})')
    plt.plot(df_thresh_results['threshold'], df_thresh_results['iou_vs_consensus_no_giovanni'], marker='^', label=f'IoU vs Consensus (No Gio) (n~{df_thresh_results["n_no_giovanni"].mean():.0f})')
    
    plt.xlabel("Saliency Binarization Threshold")
    plt.ylabel("Average IoU")
    plt.title(f"IoU vs. Binarization Threshold ({saliency_method_for_plot} on {model_name_for_plot_title})")
    plt.legend()
    plt.grid(True)
    plt.xticks(np.round(np.linspace(min(thresholds_to_test), max(thresholds_to_test), 10),2))
    plt.ylim(bottom=0)
    
    # Save plot
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    plot_filename = f"iou_vs_threshold_plot_{model_key_for_plot}_{saliency_method_for_plot}.png"
    plot_path = os.path.join(OUTPUT_PLOT_DIR, plot_filename)
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    print(f"Plot saved to: {plot_path}")
    plt.show()

    # Save CSV
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    csv_filename = f"iou_vs_threshold_{model_key_for_plot}_{saliency_method_for_plot}.csv"
    csv_path = os.path.join(OUTPUT_CSV_DIR, csv_filename)
    df_thresh_results.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Threshold analysis results saved to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze IoU vs. Saliency Binarization Threshold.")
    parser.add_argument("model_key", type=str, choices=["an", "vgg", "rn", "in"], help="Model key (e.g., 'vgg', 'an').")
    parser.add_argument("saliency_method", type=str, choices=["CAM", "GradCAM", "RISE"], help="Saliency method to use.")
    parser.add_argument("--threshold_start", type=float, default=0.02, help="Start of threshold range.")
    parser.add_argument("--threshold_end", type=float, default=0.90, help="End of threshold range.")
    parser.add_argument("--threshold_step", type=float, default=0.02, help="Step for threshold range.")
    
    cli_args = parser.parse_args()
    main(cli_args)
