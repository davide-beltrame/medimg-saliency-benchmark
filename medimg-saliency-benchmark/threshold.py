import argparse
import os
import json
# import glob # No longer needed for checkpoint finding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Assuming utils.py, models.py, saliency.py are in the same directory
# or accessible via PYTHONPATH.
import utils
from models import BaseCNN
import saliency

# --- Configuration Section ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
ANNOTATIONS_METADATA_PATH = os.path.join(PROJECT_ROOT, "data/annotations/metadata.json")
ANNOTATED_MASKS_DIR = os.path.join(PROJECT_ROOT, "data/annotations/annotated")
ORIGINAL_IMAGES_DIR_FOR_SALIENCY = os.path.join(PROJECT_ROOT, "data/test")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

MODEL_INPUT_SIZE = (224, 224)
#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
#_#  USER ACTION REQUIRED:                                                #_#
#_#  Verify the exact name for Giovanni Casini in your metadata.json      #_#
#_#  (case-insensitive). Update the GIOVANNI_NAME_PATTERN below.          #_#
#_#  To find it, you can load metadata.json into a pandas DataFrame       #_#
#_#  and print df['annotator_name'].unique()                              #_#
#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
GIOVANNI_NAME_PATTERN = "giovanni casini" # <<< VERIFY AND EDIT THIS
# --- End Configuration Section ---

def get_expert_consensus_masks_for_specific_annotators(
    group_name,
    filtered_annotations_metadata,
    all_image_names_in_dataset # Currently unused, but good for context
):
    """
    Generates consensus masks based on a pre-filtered list of annotations.
    Returns a dictionary: {image_filename: consensus_mask_np}.
    Only includes images where the final consensus mask is non-empty.
    """
    print(f"  Processing consensus for group: {group_name}")
    expert_consensus_masks = {}

    unique_image_names_for_this_consensus = sorted(list(set(record['image_name'] for record in filtered_annotations_metadata)))

    if not unique_image_names_for_this_consensus:
        print(f"    No images found with annotations from group '{group_name}' after filtering.")
        return {}

    # print(f"    Found {len(filtered_annotations_metadata)} annotations for {len(unique_image_names_for_this_consensus)} unique images in group '{group_name}'.")

    for image_name in unique_image_names_for_this_consensus:
        raw_masks_tuples_for_image = utils.get_masks_for_image_from_metadata(
            image_name,
            filtered_annotations_metadata,
            ANNOTATED_MASKS_DIR,
            target_size=MODEL_INPUT_SIZE
        )

        if not raw_masks_tuples_for_image:
            continue

        individual_masks_for_image = [mask_tuple[0] for mask_tuple in raw_masks_tuples_for_image]

        base_processed_masks = []
        for idx, raw_mask in enumerate(individual_masks_for_image):
            processed_mask_step1 = utils.process_circled_annotation(
                raw_mask,
                initial_closing_kernel_size=utils.INITIAL_PRE_CLOSING_KERNEL_SIZE,
                solidity_threshold=utils.SOLIDITY_THRESHOLD,
                outline_fill_closing_kernel_size=utils.OUTLINE_FILL_CLOSING_KERNEL_SIZE,
                outline_erosion_kernel_size=utils.OUTLINE_EROSION_KERNEL_SIZE,
                filled_region_hole_closing_kernel_size=utils.FILLED_REGION_HOLE_CLOSING_KERNEL_SIZE,
                min_contour_area=utils.MIN_CONTOUR_AREA_FILTER
            )
            if processed_mask_step1 is None:
                processed_mask_step1 = np.zeros(MODEL_INPUT_SIZE, dtype=np.uint8)
            base_processed_masks.append(processed_mask_step1)

        if not base_processed_masks:
            continue

        final_consensus = utils.create_consensus_mask(
            base_processed_masks,
            filter_type=utils.CONSENSUS_POST_FILTER_TYPE,
            filter_kernel_size=utils.CONSENSUS_POST_FILTER_KERNEL_SIZE,
            consensus_method=utils.CONSENSUS_METHOD
        )

        if final_consensus is not None and final_consensus.sum() > 0:
            expert_consensus_masks[image_name] = final_consensus

    print(f"    Generated {len(expert_consensus_masks)} non-empty consensus masks for group '{group_name}'.")
    return expert_consensus_masks


def main():
    parser = argparse.ArgumentParser(description="Evaluate IoU vs. Binarization Threshold for different expert consensus groups.")
    parser.add_argument("model_key", type=str, help="Model short key ('an', 'vgg', 'rn', 'in').")
    parser.add_argument("saliency_method", type=str, help="Saliency method (e.g., 'CAM', 'GradCAM').")
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    device = utils.get_device()
    print(f"Using device: {device}")

    # 1. Load Model
    target_ckpt_path = None
    model_key_lower = args.model_key.lower()

    print(f"\nSelecting checkpoint for model key: '{model_key_lower}'")

    if model_key_lower == "an":  # AlexNet
        target_ckpt_path = os.path.join(CHECKPOINT_DIR, "an_True_True_0.05.ckpt") 
    elif model_key_lower == "vgg":  # VGG16
        target_ckpt_path = os.path.join(CHECKPOINT_DIR, "vgg_True_True_0.03.ckpt") 
    elif model_key_lower == "rn":  # ResNet (e.g., ResNet50)
        target_ckpt_path = os.path.join(CHECKPOINT_DIR, "rn_True_True_0.05.ckpt")
    elif model_key_lower == "in":  # InceptionNetV1 (GoogLeNet)
        target_ckpt_path = os.path.join(CHECKPOINT_DIR, "in_True_True_0.01.ckpt")
    else:
        print(f"Error: Model key '{args.model_key}' is not recognized for hardcoded checkpoint selection.")
        print(f"Supported keys: 'an' (AlexNet), 'vgg' (VGG), 'rn' (ResNet), 'in' (InceptionNet).")
        print(f"Please add an entry for this key in the script or use a supported key.")
        return

    if "YOUR_" in target_ckpt_path: # Basic check if placeholder is still there
        print(f"Error: Placeholder checkpoint filename found for model '{args.model_key}'.")
        print(f"Please edit threshold.py and replace '{os.path.basename(target_ckpt_path)}' with your actual checkpoint file.")
        return

    if not os.path.exists(target_ckpt_path):
        print(f"Error: Specified checkpoint file does not exist: {target_ckpt_path}")
        print(f"Please ensure the filename and path are correct for model '{args.model_key}'.")
        return

    print(f"Attempting to load specified checkpoint: {os.path.basename(target_ckpt_path)}")
    
    model_checkpoint = BaseCNN.load_from_checkpoint(target_ckpt_path, map_location=device)
    model_to_explain = model_checkpoint.model 
    model_to_explain.to(device).eval()
    
    print(f"Successfully loaded model: {type(model_to_explain).__name__} from {os.path.basename(target_ckpt_path)}")

    # 2. Initialize Saliency Tool
    saliency_tool = None
    saliency_method_upper = args.saliency_method.upper()

    if saliency_method_upper == "CAM":
        if hasattr(model_to_explain, 'linear') and not model_to_explain.linear:
            print(f"Warning: CAM selected, and the loaded model '{type(model_to_explain).__name__}' has its 'linear' attribute set to False.")
            print(f"         CAM may not function correctly or may produce poor results with this model configuration.")
        elif not hasattr(model_to_explain, 'linear') and model_key_lower not in ['rn', 'in']:
            print(f"Warning: CAM selected for model '{type(model_to_explain).__name__}'. This model type does not have an explicit 'linear' attribute for quick compatibility check.")
            print(f"         Ensure its architecture (e.g., ending with GAP and a Linear layer) is suitable for CAM.")
        
        try:
            saliency_tool = saliency.CAM(model_to_explain)
        except Exception as e:
            print(f"Error initializing CAM: {e}")
            print(f"Ensure the loaded model ('{type(model_to_explain).__name__}') and its checkpoint ('{os.path.basename(target_ckpt_path)}') are compatible with CAM.")
            return
    elif saliency_method_upper == "GRADCAM":
        saliency_tool = saliency.GradCAM(model_to_explain)
    else:
        print(f"Error: Saliency method '{args.saliency_method}' not supported by this script.")
        return
    print(f"Initialized saliency method: {saliency_method_upper}")

    # 3. Load and Prepare Annotation Metadata
    if not os.path.exists(ANNOTATIONS_METADATA_PATH):
        print(f"Error: Annotations metadata file not found at {ANNOTATIONS_METADATA_PATH}")
        return
        
    with open(ANNOTATIONS_METADATA_PATH, 'r') as f:
        annotations_metadata_raw = json.load(f)
    
    df_metadata_all = pd.DataFrame(annotations_metadata_raw)
    df_metadata_no_test = df_metadata_all[
        ~df_metadata_all['annotator_name'].str.contains('test', case=False, na=False)
    ].copy()
    annotations_non_test_list = df_metadata_no_test.to_dict(orient='records')
    
    print(f"\nTotal non-test annotations loaded: {len(annotations_non_test_list)}")
    
    # --- For Debugging GIOVANNI_NAME_PATTERN ---
    # Uncomment the following lines to print all unique lowercased annotator names
    # from your non-test metadata. This will help you find the correct pattern for Giovanni.
    # ----
    # if annotations_non_test_list:
    #     unique_names_in_data = sorted(list(set(r.get('annotator_name', '').lower() for r in annotations_non_test_list if r.get('annotator_name'))))
    #     print(f"DEBUG: Unique lowercased annotator names in non-test data: {unique_names_in_data}")
    # else:
    #     print("DEBUG: No non-test annotations found to extract unique names.")
    # ----

    annotations_full_expert = annotations_non_test_list

    initial_gio_annotations = [
        r for r in annotations_non_test_list if GIOVANNI_NAME_PATTERN in r.get('annotator_name', '').lower()
    ]
    if not initial_gio_annotations:
        print(f"CRITICAL: No annotator names in the (non-test) metadata matched the pattern '{GIOVANNI_NAME_PATTERN}'. 'Giovanni Only' curve will be empty or missing. Please verify the pattern against actual annotator names in metadata.json.")
    else:
        print(f"Found {len(initial_gio_annotations)} raw annotations potentially matching Giovanni ('{GIOVANNI_NAME_PATTERN}').")
    annotations_giovanni_only = initial_gio_annotations
    
    annotations_no_giovanni = [
        r for r in annotations_non_test_list if GIOVANNI_NAME_PATTERN not in r.get('annotator_name', '').lower()
    ]

    # 4. Get the three types of expert consensus masks
    print("\nGenerating consensus masks...")
    consensus_masks_full = get_expert_consensus_masks_for_specific_annotators("Full Expert", annotations_full_expert, [])
    
    consensus_masks_gio = {}
    if annotations_giovanni_only: 
        consensus_masks_gio = get_expert_consensus_masks_for_specific_annotators("Giovanni Only", annotations_giovanni_only, [])
    else:
        print("  Skipping Giovanni Only consensus mask generation as no raw annotations were found for the pattern.")
    
    consensus_masks_no_gio = get_expert_consensus_masks_for_specific_annotators("No Giovanni", annotations_no_giovanni, [])
    
    all_evaluation_image_names = sorted(list(set(r['image_name'] for r in annotations_non_test_list)))
    print(f"\nWill attempt to generate saliency maps for up to {len(all_evaluation_image_names)} unique images.")

    # 5. Iterate through thresholds and calculate IoUs
    thresholds_arr = np.round(np.arange(0.02, 0.91, 0.02), 2)
    
    iou_results_full = {t: [] for t in thresholds_arr}
    iou_results_gio = {t: [] for t in thresholds_arr}
    iou_results_no_gio = {t: [] for t in thresholds_arr}
    
    images_contributing_full = set()
    images_contributing_gio = set()
    images_contributing_no_gio = set()

    print(f"Starting IoU calculation across {len(thresholds_arr)} thresholds for {len(all_evaluation_image_names)} images...")
    for img_idx, image_filename in enumerate(all_evaluation_image_names):
        image_path_pneumonia = os.path.join(ORIGINAL_IMAGES_DIR_FOR_SALIENCY, "PNEUMONIA", image_filename)
        image_path_normal = os.path.join(ORIGINAL_IMAGES_DIR_FOR_SALIENCY, "NORMAL", image_filename)
        
        actual_image_path = None
        if os.path.exists(image_path_pneumonia):
            actual_image_path = image_path_pneumonia
        elif os.path.exists(image_path_normal):
            actual_image_path = image_path_normal
        else:
            continue

        input_tensor = utils.load_image_tensor(actual_image_path, device)
        if input_tensor is None:
            continue

        try:
            saliency_map_np = saliency_tool(input_tensor) 
        except Exception as e:
            # print(f"  Error generating saliency map for {image_filename} with {saliency_method_upper}: {e}. Skipping.")
            continue
            
        if saliency_map_np is None:
            continue
        if saliency_map_np.shape != MODEL_INPUT_SIZE: 
             saliency_map_np = utils.cv2.resize(saliency_map_np, MODEL_INPUT_SIZE, interpolation=utils.cv2.INTER_LINEAR)

        # --- Full Expert Consensus ---
        if image_filename in consensus_masks_full:
            expert_mask_full = consensus_masks_full[image_filename]
            images_contributing_full.add(image_filename)
            for t in thresholds_arr:
                binarized_saliency = utils.binarize_saliency_map(saliency_map_np, threshold_value=t)
                iou = utils.calculate_iou(binarized_saliency, expert_mask_full)
                iou_results_full[t].append(iou)

        # --- Giovanni Only Consensus ---
        if image_filename in consensus_masks_gio: 
            expert_mask_gio = consensus_masks_gio[image_filename]
            images_contributing_gio.add(image_filename)
            for t in thresholds_arr:
                binarized_saliency = utils.binarize_saliency_map(saliency_map_np, threshold_value=t)
                iou = utils.calculate_iou(binarized_saliency, expert_mask_gio)
                iou_results_gio[t].append(iou)
        
        # --- No Giovanni Consensus ---
        if image_filename in consensus_masks_no_gio:
            expert_mask_no_gio = consensus_masks_no_gio[image_filename]
            images_contributing_no_gio.add(image_filename)
            for t in thresholds_arr:
                binarized_saliency = utils.binarize_saliency_map(saliency_map_np, threshold_value=t)
                iou = utils.calculate_iou(binarized_saliency, expert_mask_no_gio)
                iou_results_no_gio[t].append(iou)
        
    if hasattr(saliency_tool, 'remove_hook') and callable(saliency_tool.remove_hook):
         saliency_tool.remove_hook()
    if hasattr(saliency_tool, 'remove_hooks') and callable(saliency_tool.remove_hooks): 
         saliency_tool.remove_hooks()

    # 6. Aggregate results and create DataFrame
    avg_iou_full = [np.mean(iou_results_full[t]) if iou_results_full[t] else np.nan for t in thresholds_arr]
    avg_iou_gio = [np.mean(iou_results_gio[t]) if iou_results_gio[t] else np.nan for t in thresholds_arr]
    avg_iou_no_gio = [np.mean(iou_results_no_gio[t]) if iou_results_no_gio[t] else np.nan for t in thresholds_arr]

    num_images_full = len(images_contributing_full)
    num_images_gio = len(images_contributing_gio)
    num_images_no_gio = len(images_contributing_no_gio)

    print(f"\nNumber of unique images contributing to IoU calculation (n):")
    print(f"  IoU vs Full Consensus: n={num_images_full}")
    print(f"  IoU vs Giovanni Only: n={num_images_gio}")
    print(f"  IoU vs Consensus (No Gio): n={num_images_no_gio}")

    df_columns = {'Saliency Binarization Threshold': thresholds_arr}
    if num_images_full > 0 : df_columns[f'IoU vs Full Consensus (n={num_images_full})'] = avg_iou_full
    if num_images_gio > 0 : df_columns[f'IoU vs Giovanni Only (n={num_images_gio})'] = avg_iou_gio
    if num_images_no_gio > 0 : df_columns[f'IoU vs Consensus (No Gio) (n={num_images_no_gio})'] = avg_iou_no_gio
    results_df = pd.DataFrame(df_columns)

    # 7. Save CSV
    csv_filename = f"iou_vs_threshold_{args.model_key.lower()}_{saliency_method_upper.lower()}.csv"
    csv_path = os.path.join(PLOTS_DIR, csv_filename)
    results_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nResults saved to {csv_path}")

    # 8. Analyzing Best Thresholds
    print("\n--- Analyzing Best Thresholds from Calculated Data ---")
    col_full = f'IoU vs Full Consensus (n={num_images_full})'
    if num_images_full > 0 and col_full in results_df.columns and not results_df[col_full].isnull().all():
        best_idx_full = results_df[col_full].idxmax()
        best_thresh_full = results_df['Saliency Binarization Threshold'].loc[best_idx_full]
        max_iou_full = results_df[col_full].loc[best_idx_full]
        print(f"For 'Full Expert Consensus': Best Threshold = {best_thresh_full:.4f}, Max Avg. IoU = {max_iou_full:.4f}")
    
    col_gio = f'IoU vs Giovanni Only (n={num_images_gio})'
    if num_images_gio > 0 and col_gio in results_df.columns and not results_df[col_gio].isnull().all():
        best_idx_gio = results_df[col_gio].idxmax()
        best_thresh_gio = results_df['Saliency Binarization Threshold'].loc[best_idx_gio]
        max_iou_gio = results_df[col_gio].loc[best_idx_gio]
        print(f"For 'Giovanni Only Consensus': Best Threshold = {best_thresh_gio:.4f}, Max Avg. IoU = {max_iou_gio:.4f}")

    col_no_gio = f'IoU vs Consensus (No Gio) (n={num_images_no_gio})'
    if num_images_no_gio > 0 and col_no_gio in results_df.columns and not results_df[col_no_gio].isnull().all():
        best_idx_no_gio = results_df[col_no_gio].idxmax()
        best_thresh_no_gio = results_df['Saliency Binarization Threshold'].loc[best_idx_no_gio]
        max_iou_no_gio = results_df[col_no_gio].loc[best_idx_no_gio]
        print(f"For 'Consensus (No Giovanni)': Best Threshold = {best_thresh_no_gio:.4f}, Max Avg. IoU = {max_iou_no_gio:.4f}")

    # 9. Generate and Save Plot
    plt.style.use('seaborn-v0_8-whitegrid') 
    plt.figure(figsize=(10, 6))
    
    plotted_anything = False
    if num_images_full > 0 and col_full in results_df.columns and not results_df[col_full].isnull().all(): # check for all NaNs
        plt.plot(results_df['Saliency Binarization Threshold'], results_df[col_full], marker='o', linestyle='-', label=col_full)
        plotted_anything = True
    if num_images_gio > 0 and col_gio in results_df.columns and not results_df[col_gio].isnull().all(): # check for all NaNs
        plt.plot(results_df['Saliency Binarization Threshold'], results_df[col_gio], marker='s', linestyle='-', label=col_gio)
        plotted_anything = True
    if num_images_no_gio > 0 and col_no_gio in results_df.columns and not results_df[col_no_gio].isnull().all(): # check for all NaNs
        plt.plot(results_df['Saliency Binarization Threshold'], results_df[col_no_gio], marker='^', linestyle='-', label=col_no_gio)
        plotted_anything = True

    plt.title(f'IoU vs. Binarization Threshold ({saliency_method_upper} on {args.model_key.upper()})')
    plt.xlabel('Saliency Binarization Threshold')
    plt.ylabel('Average IoU')
    
    plt.grid(True)
    if plotted_anything:
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No data to plot for any group.", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        
    plt.tight_layout()

    plot_filename = f"iou_vs_threshold_{args.model_key.lower()}_{saliency_method_upper.lower()}.png"
    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()