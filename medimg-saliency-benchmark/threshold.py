import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import utils
from models import BaseCNN
import saliency

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
ANNOTATIONS_METADATA_PATH = os.path.join(PROJECT_ROOT, "data/annotations/clean_metadata.csv")
ANNOTATED_MASKS_DIR = os.path.join(PROJECT_ROOT, "data/annotations/annotated")
ORIGINAL_IMAGES_DIR_FOR_SALIENCY = os.path.join(PROJECT_ROOT, "data/test")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

MODEL_INPUT_SIZE = (224, 224)

MODEL_CONFIGS = {
    "an": "an_True_True_0.05.ckpt",    
    "vgg": "vgg_True_True_0.03.ckpt",  
    "rn": "rn_True_True_0.05.ckpt",   
    "in": "in_True_True_0.01.ckpt"      
}

def get_expert_consensus_masks_for_specific_annotators(
    group_name,
    filtered_annotations_metadata
):
    """
    Generates consensus masks based on a pre-filtered list of annotations.
    Returns a dictionary: {image_filename: consensus_mask_np}.
    """
    print(f"  Processing consensus for group: {group_name}")
    expert_consensus_masks = {}
    unique_image_names_for_this_consensus = sorted(list(set(record['image_name'] for record in filtered_annotations_metadata)))

    if not unique_image_names_for_this_consensus:
        print(f"    No images found with annotations from group '{group_name}' after filtering.")
        return {}

    for image_name in unique_image_names_for_this_consensus:
        raw_masks_tuples_for_image = utils.get_masks_for_image_from_metadata(
            image_name, filtered_annotations_metadata, ANNOTATED_MASKS_DIR, target_size=MODEL_INPUT_SIZE
        )
        if not raw_masks_tuples_for_image:
            continue

        individual_masks_for_image = [mt[0] for mt in raw_masks_tuples_for_image]
        
        # Corrected part: Reverted from list comprehension with 'or' to explicit loop
        base_processed_masks = []
        for raw_mask in individual_masks_for_image:
            processed_mask_step1 = utils.process_circled_annotation(
                raw_mask,
                initial_closing_kernel_size=utils.INITIAL_PRE_CLOSING_KERNEL_SIZE,
                solidity_threshold=utils.SOLIDITY_THRESHOLD,
                outline_fill_closing_kernel_size=utils.OUTLINE_FILL_CLOSING_KERNEL_SIZE,
                outline_erosion_kernel_size=utils.OUTLINE_EROSION_KERNEL_SIZE,
                filled_region_hole_closing_kernel_size=utils.FILLED_REGION_HOLE_CLOSING_KERNEL_SIZE,
                min_contour_area=utils.MIN_CONTOUR_AREA_FILTER
            )
            # utils.process_circled_annotation should return an all-zeros array if processing fails or input is None/empty.
            # Adding a None check here just in case its behavior changes or for extra safety.
            if processed_mask_step1 is None:
                processed_mask_step1 = np.zeros(MODEL_INPUT_SIZE, dtype=np.uint8)
            base_processed_masks.append(processed_mask_step1)

        if not base_processed_masks: # If all individual masks processed to empty or were initially empty
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
    parser = argparse.ArgumentParser(description="Evaluate IoU vs. Binarization Threshold for multiple models.")
    parser.add_argument("saliency_method", type=str, help="Saliency method (e.g., 'CAM', 'GradCAM').")
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    device = utils.get_device()
    print(f"Using device: {device}")
    saliency_method_upper = args.saliency_method.upper()

    # 1. Load and Prepare Annotation Metadata (once for all models)
    annotations_metadata_list_filtered = pd.read_csv(
        os.path.join(
            os.path.dirname(ANNOTATIONS_METADATA_PATH),
            "clean_metadata.csv"
        )
    ).to_dict(orient='records')
    
    df_metadata_all = pd.DataFrame(annotations_metadata_list_filtered)
    df_metadata_no_test = df_metadata_all[~df_metadata_all['annotator_name'].str.contains('test', case=False, na=False)].copy()
    annotations_non_test_list = df_metadata_no_test.to_dict(orient='records')
    print(f"\nTotal non-test annotations loaded: {len(annotations_non_test_list)}")

    annotations_full_expert = annotations_non_test_list

    # 2. Get Expert Consensus Masks (once for all models)
    print("\nGenerating expert consensus masks (once for all models)...")
    consensus_masks_full = get_expert_consensus_masks_for_specific_annotators("Full Expert", annotations_full_expert)

    all_evaluation_image_names = sorted(list(set(r['image_name'] for r in annotations_non_test_list)))
    print(f"\nWill attempt evaluation for up to {len(all_evaluation_image_names)} unique images from non-test annotations.")

    thresholds_arr = np.round(np.arange(0.02, 0.91, 0.02), 2)
    
    # DataFrame to store all results
    all_results_data = {"Saliency Binarization Threshold": thresholds_arr}
    # To store 'n' for legend and best threshold analysis
    n_values_for_legend = {} 
    best_threshold_details = []

    # 3. Loop through each model configuration
    for model_key, ckpt_filename_part in MODEL_CONFIGS.items():
        print(f"\n--- Processing Model: {model_key.upper()} ---")
        target_ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_filename_part)

        if not os.path.exists(target_ckpt_path):
            print(f"Error: Checkpoint file '{ckpt_filename_part}' for model '{model_key}' not found at {target_ckpt_path}. Skipping this model.")
            continue
        
        print(f"  Attempting to load checkpoint: {ckpt_filename_part}")
        model_checkpoint = BaseCNN.load_from_checkpoint(target_ckpt_path, map_location=device)
        model_to_explain = model_checkpoint.model
        model_to_explain.to(device).eval()
        print(f"  Successfully loaded model: {type(model_to_explain).__name__}")

        # Initialize Saliency Tool for the current model
        saliency_tool = None
        if saliency_method_upper == "CAM":
            if hasattr(model_to_explain, 'linear') and not model_to_explain.linear:
                print(f"  Warning: CAM selected, loaded model '{type(model_to_explain).__name__}' has linear=False. CAM may not work as expected.")
            try:
                saliency_tool = saliency.CAM(model_to_explain)
            except Exception as e:
                print(f"  Error initializing CAM for {model_key.upper()}: {e}. Skipping saliency for this model.")
                continue
        elif saliency_method_upper == "GRADCAM":
            saliency_tool = saliency.GradCAM(model_to_explain)
        elif saliency_method_upper == "RISE":
            saliency_tool = saliency.RISE(model_to_explain)
        else:
            print(f"  Error: Saliency method '{saliency_method_upper}' not supported. Skipping saliency for this model.")
            continue
        print(f"  Initialized saliency method: {saliency_method_upper} for {model_key.upper()}")

        # Initialize results storage for this model
        iou_results_full_model = {t: [] for t in thresholds_arr}
        images_contributing_full_model = set()

        print(f"  Starting IoU calculation for {model_key.upper()} across {len(all_evaluation_image_names)} images...")
        for image_filename in all_evaluation_image_names:
            image_path_pneumonia = os.path.join(ORIGINAL_IMAGES_DIR_FOR_SALIENCY, "PNEUMONIA", image_filename)
            image_path_normal = os.path.join(ORIGINAL_IMAGES_DIR_FOR_SALIENCY, "NORMAL", image_filename)
            actual_image_path = None
            if os.path.exists(image_path_pneumonia): actual_image_path = image_path_pneumonia
            elif os.path.exists(image_path_normal): actual_image_path = image_path_normal
            else: continue

            input_tensor = utils.load_image_tensor(actual_image_path, device)
            if input_tensor is None: continue

            try:
                saliency_map_np = saliency_tool(input_tensor)
            except Exception as e: continue
            
            if saliency_map_np is None: continue
            if saliency_map_np.shape != MODEL_INPUT_SIZE:
                saliency_map_np = utils.cv2.resize(saliency_map_np, MODEL_INPUT_SIZE, interpolation=utils.cv2.INTER_LINEAR)

            if image_filename in consensus_masks_full:
                images_contributing_full_model.add(image_filename)
                for t in thresholds_arr:
                    bin_map = utils.binarize_saliency_map(saliency_map_np, threshold_value=t)
                    iou_results_full_model[t].append(utils.calculate_iou(bin_map, consensus_masks_full[image_filename]))
        
        if hasattr(saliency_tool, 'remove_hook') and callable(saliency_tool.remove_hook): saliency_tool.remove_hook()
        if hasattr(saliency_tool, 'remove_hooks') and callable(saliency_tool.remove_hooks): saliency_tool.remove_hooks()

        # Aggregate and store results for this model
        avg_iou_full_model = [np.mean(iou_results_full_model[t]) if iou_results_full_model[t] else np.nan for t in thresholds_arr]
        
        n_full = len(images_contributing_full_model)
        n_values_for_legend[(model_key, 'Full')] = n_full

        print(f"  Finished IoU for {model_key.upper()}: Full (n={n_full})")

        if n_full > 0:
            col_name_full = f"{model_key.upper()}_Full (n={n_full})"
            all_results_data[col_name_full] = avg_iou_full_model
            if not pd.Series(avg_iou_full_model).isnull().all():
                best_idx = pd.Series(avg_iou_full_model).idxmax()
                best_thresh = thresholds_arr[best_idx]
                max_iou = avg_iou_full_model[best_idx]
                best_threshold_details.append(f"  {model_key.upper()} Full: Best Thr={best_thresh:.4f}, Max IoU={max_iou:.4f} (n={n_full})")

    # 4. Create final DataFrame and Save CSV
    results_df = pd.DataFrame(all_results_data)
    csv_filename = f"all_models_iou_vs_threshold_{saliency_method_upper.lower()}.csv"
    csv_path = os.path.join(PLOTS_DIR, csv_filename)
    results_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nAll model results saved to {csv_path}")

    print("\n--- Analyzing Best Thresholds from Calculated Data ---")

    for detail in best_threshold_details:
        print(detail)

    # create dataframe with the best thresholds and the max iou for each model

    best_thresholds_df = pd.DataFrame(best_threshold_details)
    best_thresholds_df.to_csv(os.path.join(PLOTS_DIR, f"best_thresholds_{saliency_method_upper.lower()}.csv"), index=False)
    print(f"\nBest thresholds saved to {os.path.join(PLOTS_DIR, f'best_thresholds_{saliency_method_upper.lower()}.csv')}")


    # 5. Generate and Save Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8)) # Wider for more lines
    
    # Define a list of distinct colors and linestyles
    # Cycle through colors for models
    model_colors = plt.cm.get_cmap('tab10', len(MODEL_CONFIGS)) 

    plotted_anything = False
    for i, model_key in enumerate(MODEL_CONFIGS.keys()):
        n_full = n_values_for_legend.get((model_key, 'Full'), 0)
        col_full = f"{model_key.upper()}_Full (n={n_full})"
        
        if n_full > 0 and col_full in results_df.columns and not results_df[col_full].isnull().all():
            plt.plot(results_df['Saliency Binarization Threshold'], results_df[col_full], 
                     marker='o', markersize=4, linestyle='-', label=f"{model_key.upper()} Full", color=model_colors(i))
            plotted_anything = True

    plt.title(f'All Models IoU vs. Binarization Threshold ({saliency_method_upper})')
    plt.xlabel('Saliency Binarization Threshold')
    plt.ylabel('Average IoU')
    plt.grid(True)
    if plotted_anything:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) # Legend outside plot
    else:
        plt.text(0.5, 0.5, "No data to plot.", ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    plot_filename = f"all_models_iou_vs_threshold_{saliency_method_upper.lower()}.png"
    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()