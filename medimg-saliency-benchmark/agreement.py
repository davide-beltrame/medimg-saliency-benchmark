import os
import glob

import numpy as np
import pandas as pd
import utils
from models import BaseCNN 
import saliency

CONSENSUS_TYPE = "full" # this is only used for the output file name
RUN_NAME = "test" # this is only used for the output file name

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
ANNOTATIONS_METADATA_PATH = os.path.join(os.path.dirname(__file__),"data/annotations/metadata.json")
ANNOTATED_MASKS_DIR = os.path.join(os.path.dirname(__file__),"data/annotations/annotated")
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


def main():

    # Get device
    device = utils.get_device()
    print(f"Using device: {device}")

    # Load filtered annotations 
    annotations_metadata_list_filtered = pd.read_csv(
        os.path.join(
            os.path.dirname(ANNOTATIONS_METADATA_PATH),
            "clean_metadata.csv"
        )
    ).to_dict(orient='records')

    # 2. Get Expert Consensus Masks for images to be evaluated
    # This dictionary will contain {image_filename: consensus_mask_np}
    # Only images with non-empty consensus masks will be included.
    expert_consensus_masks = utils.get_consensus_masks_for_evaluation(
        annotations_metadata_list_filtered,
        ANNOTATED_MASKS_DIR
    )

    evaluation_images = list(expert_consensus_masks.keys())
    print(f"\nStarting saliency evaluation for {len(evaluation_images)} images with non-empty consensus masks.")

     # RISE is not included because it's way too slow, but it works
    saliency_methods = ["CAM", "GradCAM", "Random"]

    # To store dicts for DataFrame: {'model': 'an', 'linear': True, ... 'CAM': 0.1, ...}
    all_results_data = [] 

    # Get all checkpoint files from the CHECKPOINT_DIR
    all_checkpoint_paths = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.ckpt")))
    print(f"\nFound {len(all_checkpoint_paths)} checkpoint files to process.")

    # Loop 1: Model Checkpoint
    for ckpt_idx, ckpt_path in enumerate(all_checkpoint_paths):
        print(f"\nProcessing checkpoint {ckpt_idx + 1}/{len(all_checkpoint_paths)}: {os.path.basename(ckpt_path)}")
        
        parsed_info = utils.parse_checkpoint_filename(os.path.basename(ckpt_path))

        current_config_results = {
            'model': parsed_info['model'],
            'linear': parsed_info['linear'],
            'pretrained': parsed_info['pretrained']
        }

        # Load the model checkpoint
        model = BaseCNN.load_from_checkpoint(ckpt_path, map_location=device)
        model.model.to(device).eval()

        # Loop 2: Saliency Methods
        for sm_name in saliency_methods:

            # Skip models no appropriate for CAM
            if sm_name == "CAM" and not current_config_results["linear"]:
                continue
            
            # Keep track of results
            ious_for_current_saliency_method = []
            try:
                if sm_name == "CAM" and hasattr(saliency, 'CAM'):
                    saliency_tool = saliency.CAM(model.model)
                elif sm_name == "GradCAM" and hasattr(saliency, 'GradCAM'):
                    saliency_tool = saliency.GradCAM(model.model)
                elif sm_name == "RISE" and hasattr(saliency, 'RISE'):
                    saliency_tool = saliency.RISE(model.model, num_masks=4000, scale_factor=20)
            except Exception as e:
                print(f"  Warning: Error initializing {sm_name} for {ckpt_path}: {e}")

            # Loop 3: Test Images
            for image_filename in evaluation_images:

                image_path_for_saliency = os.path.join(ORIGINAL_IMAGES_DIR_FOR_SALIENCY, "PNEUMONIA", image_filename)
                assert os.path.exists(image_path_for_saliency)

                input_tensor = utils.load_image_tensor(image_path_for_saliency, device)
                
                if input_tensor is None: 
                    continue
                
                # From dict of annotatoins
                expert_mask_np = expert_consensus_masks[image_filename]

                saliency_map_np = None
                if sm_name == "Random":
                    saliency_map_np = utils.generate_random_map(size=MODEL_INPUT_SIZE)
                else:
                    saliency_map_np = saliency_tool(input_tensor)
                assert saliency_map_np is not None

                binarized_saliency_map = utils.binarize_saliency_map(
                    saliency_map_np,
                    method="fixed",
                    threshold_value=SALIENCY_BINARIZATION_THRESHOLD
                )
                iou = utils.calculate_iou(binarized_saliency_map, expert_mask_np)
                ious_for_current_saliency_method.append(iou)

            # Best practice            
            if saliency_tool and hasattr(saliency_tool, 'remove_hook'):
                saliency_tool.remove_hook()
                
            # Ensure we have a valid value for the saliency method (even if it's 0)
            avg_iou = np.mean(ious_for_current_saliency_method) if ious_for_current_saliency_method else float("nan")
            current_config_results[sm_name] = avg_iou
        
        all_results_data.append(current_config_results)
        print(
            f"Finished {os.path.basename(ckpt_path)}."
            f"Avg IoUs: CAM={current_config_results.get('CAM', float('nan')):.4f},"
            f"GradCAM={current_config_results.get('GradCAM', float('nan')):.4f},"
            f"Random={current_config_results.get('Random', float('nan')):.4f}"
        )

    # Convert to df
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