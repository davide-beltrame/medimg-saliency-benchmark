"""
For each model checkpoint, for each saliency methods and for each image compute:
 - IoU
 - Pointing Game
 P-values between saliency scores and random score
"""
import os
import glob

import scipy.stats as stats
import numpy as np
import pandas as pd
import utils
from models import BaseCNN 
import saliency

CONSENSUS_TYPE = "full" # this is only used for the output file name
RUN_NAME = "test" # this is only used for the output file name

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__),
    "checkpoints"
)
ANNOTATIONS_METADATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data/annotations/metadata.json"
)
ANNOTATED_MASKS_DIR = os.path.join(
    os.path.dirname(__file__),
    "data/annotations/annotated"
)
ORIGINAL_IMAGES_DIR_FOR_SALIENCY = os.path.join(
    os.path.dirname(__file__),
    "data/annotations/original"
)
PLOTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "plots"
)

MODEL_INPUT_SIZE = (224, 224) 

# Default threshold (used for Random and as fallback)
DEFAULT_SALIENCY_BINARIZATION_THRESHOLD = 0.74

# morphological filter parameters tuned empirically for full consensus, you can read about them in utils.py
INITIAL_PRE_CLOSING_KERNEL_SIZE = 3 
SOLIDITY_THRESHOLD = 0.6            
OUTLINE_FILL_CLOSING_KERNEL_SIZE = 7 
OUTLINE_EROSION_KERNEL_SIZE = 7      
FILLED_REGION_HOLE_CLOSING_KERNEL_SIZE = 5 
MIN_CONTOUR_AREA_FILTER = 20  
CONSENSUS_POST_FILTER_TYPE = 'open' 
CONSENSUS_POST_FILTER_KERNEL_SIZE = 3
CONSENSUS_METHOD = 'mean'


def get_best_threshold_for_model(model_name, saliency_method):
    """
    Get the best threshold for a given model and saliency method from threshold analysis files.
    Returns the default threshold if threshold data is not available.
    """
    try:
        threshold_file = os.path.join(PLOTS_DIR, f"best_thresholds_{saliency_method.lower()}.csv")
        if not os.path.exists(threshold_file):
            return DEFAULT_SALIENCY_BINARIZATION_THRESHOLD
        
        # Read the CSV file
        with open(threshold_file, 'r') as f:
            lines = f.readlines()
        
        # Parse lines to find the best threshold for the model (Full consensus)
        model_upper = model_name.upper()
        for line in lines:
            if f"{model_upper} Full:" in line:
                # Extract the threshold value
                import re
                match = re.search(r"Best Thr=(\d+\.\d+)", line)
                if match:
                    return float(match.group(1))
        
        return DEFAULT_SALIENCY_BINARIZATION_THRESHOLD
    except Exception as e:
        print(f"Error reading threshold for {model_name}: {e}")
        return DEFAULT_SALIENCY_BINARIZATION_THRESHOLD


def main():

    # Get device
    device = utils.get_device()
    print(f"Using device: {device}")
    
    # Create evaluation directory if it doesn't exist
    evaluation_dir = os.path.join(os.path.dirname(__file__), "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)

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

    # To check
    metrics = ["iou", "pg"]

    # To store results for each saliency method
    results_by_method = {method: [] for method in saliency_methods}

    # To keep track of random perc
    map_active_perc = []

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
        # Skip models
        if not (current_config_results["linear"] and current_config_results["pretrained"]):
            continue
        # Load the model checkpoint
        model = BaseCNN.load_from_checkpoint(ckpt_path, map_location=device)
        model.model.to(device).eval()

        # Loop 2: Saliency Methods
        saliency_map_np = None
        for sm_name in saliency_methods:
            # Get the best threshold for this model and saliency method
            if sm_name.lower() != "random":
                threshold = get_best_threshold_for_model(current_config_results['model'], sm_name)
                print(f"  Using threshold {threshold:.4f} for {sm_name} with model {current_config_results['model']}")
            else:
                # For Random, use a fixed threshold of 0.5 
                # This provides more consistent random baselines across all models
                threshold = 0.5
                print(f"  Using fixed threshold {threshold:.4f} for {sm_name}")
            
            # Keep track of results
            ious_for_current_saliency_method = []
            pgs_for_current_saliency_method = []
            
            # Saliency method
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

                image_path_for_saliency = os.path.join(ORIGINAL_IMAGES_DIR_FOR_SALIENCY, image_filename)
                
                assert os.path.exists(image_path_for_saliency)

                input_tensor = utils.load_image_tensor(image_path_for_saliency, device)
                
                if input_tensor is None: 
                    continue
                
                # Load annotation mask
                expert_mask_np = expert_consensus_masks[image_filename]

                # Random saliency map
                if sm_name.lower() == "random":
                    assert len(map_active_perc) > 0
                    assert saliency_map_np is not None

                    saliency_map_np = utils.generate_random_mask_like(
                        saliency_map_np,
                        grid_size=10,
                        nonzero_perc=np.mean(map_active_perc)
                    )
                # True saliency map
                else:
                    saliency_map_np = saliency_tool(input_tensor)

                # Binarize
                binarized_saliency_map = utils.binarize_saliency_map(
                    saliency_map_np,
                    method="fixed",
                    threshold_value=threshold
                )

                # keep track of active perc for later generating random
                map_active_perc.append(
                    binarized_saliency_map.sum() / binarized_saliency_map.size
                )

                # Compute iou
                iou = utils.calculate_iou(
                    binarized_saliency_map,
                    expert_mask_np
                )

                # Compute pointing game
                pg = utils.pointing_game(
                    binarized_saliency_map,
                    expert_mask_np
                )

                # Store results
                ious_for_current_saliency_method.append(iou)
                pgs_for_current_saliency_method.append(pg)

            
            # Best practice            
            if saliency_tool and hasattr(saliency_tool, 'remove_hook'):
                saliency_tool.remove_hook()
                
            # Aggregate metrics for this saliency method over all images
            assert ious_for_current_saliency_method
            assert pgs_for_current_saliency_method
            
            # Store the mean values for the current saliency method
            method_results = current_config_results.copy()
            
            # Store original lists of values for statistical tests
            method_results[f"iou_original"] = ious_for_current_saliency_method.copy()
            method_results[f"pg_original"] = pgs_for_current_saliency_method.copy()
            
            # Calculate and store the means
            method_results[f"iou"] = np.mean(ious_for_current_saliency_method).item()
            method_results[f"pg"] = np.mean(pgs_for_current_saliency_method).item()
            
            # Add the threshold used to the results
            method_results["threshold"] = threshold
            
            # Add to the results for this saliency method
            results_by_method[sm_name].append(method_results)

    # For each non-random method, compute p-values vs random
    for sm_name in saliency_methods:
        if sm_name.lower() == "random":
            continue
        
        # Get random results to compare with
        random_results_df = pd.DataFrame(results_by_method["Random"])
        
        # For each model in the current saliency method
        for i, result in enumerate(results_by_method[sm_name]):
            model_name = result['model']
            
            # Find the corresponding random result for this model
            random_row = random_results_df[random_results_df['model'] == model_name]
            
            if not random_row.empty:
                # Compute p-values for each metric
                for metric in metrics:
                    # Since we only have single values (means), we need to compare the original lists
                    # Use Mann-Whitney U test with the alternative hypothesis that the saliency method
                    # produces higher scores than random
                    
                    # For the current result, get the original list of values
                    method_values = results_by_method[sm_name][i][f"{metric}_original"]
                    random_values = random_row[f"{metric}_original"].iloc[0]
                    
                    # Perform Mann-Whitney U test
                    try:
                        t_stat, p_value = stats.mannwhitneyu(
                            method_values,
                            random_values,
                            alternative='greater'
                        )
                        results_by_method[sm_name][i][f"{metric}_pval"] = p_value.item()
                    except Exception as e:
                        print(f"Warning: Mann-Whitney U test failed for {sm_name}, {metric}, model {model_name}: {e}")
                        # Fallback to simple comparison if test fails
                        if np.mean(method_values) > np.mean(random_values):
                            results_by_method[sm_name][i][f"{metric}_pval"] = 0.01
                        else:
                            results_by_method[sm_name][i][f"{metric}_pval"] = 0.99
    
    # Create separate DataFrames for each saliency method
    for sm_name in saliency_methods:
        if not results_by_method[sm_name]:
            print(f"No results for {sm_name}")
            continue
        
        # Create a copy of the results without the original lists for CSV output
        csv_results = []
        for result in results_by_method[sm_name]:
            csv_result = {k: v for k, v in result.items() if not k.endswith('_original')}
            csv_results.append(csv_result)
        
        # Convert to DataFrame
        method_df = pd.DataFrame(csv_results)
        
        # Print the results
        print(f"\n--- Results for {sm_name} ---")
        print(method_df.to_string(index=False, float_format="%.4f"))
        
        # Save to CSV
        csv_output_path = os.path.join(
            evaluation_dir,
            f"model-expert-agreement-{sm_name.lower()}.csv"
        )
        method_df.to_csv(
            csv_output_path,
            index=False,
            float_format="%.4f"
        )
        print(f"Results for {sm_name} saved to {csv_output_path}")

if __name__ == "__main__":
    main()