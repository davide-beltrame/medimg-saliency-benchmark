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

    # To check
    metrics = ["iou", "pg"]

    # To store dicts for DataFrame: {'model': 'an', 'linear': True, ... 'CAM': 0.1, ...}
    all_results_data = [] 

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

        # Load the model checkpoint
        model = BaseCNN.load_from_checkpoint(ckpt_path, map_location=device)
        model.model.to(device).eval()

        # Loop 2: Saliency Methods
        saliency_map_np = None
        for sm_name in saliency_methods:

            # Skip models no appropriate for CAM
            if not (current_config_results["linear"] and current_config_results["pretrained"]):
                continue
            # if sm_name == "CAM" and not current_config_results["linear"]:
            #     continue
            
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
                    threshold_value=SALIENCY_BINARIZATION_THRESHOLD
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
            
            # Store the list
            current_config_results[f"{sm_name}_iou"] = ious_for_current_saliency_method.copy()
            current_config_results[f"{sm_name}_pg"] = pgs_for_current_saliency_method.copy()

        
        # Compute p-values of all methods vs random
        for sm_name in saliency_methods:
            
            # Cannot do random vs random
            if sm_name.lower() == "random":
                continue
            
            # p-values
            for metric in metrics:
                colname = f"{sm_name}_{metric}"
                if colname not in current_config_results:
                    continue
                # Perform statistical test to get p-value
                # H_0: iou is not greater than random
                # --> p < 0.05 --> iou is greater than random iou
                t_stat, p_value = stats.mannwhitneyu(
                    current_config_results[f"{sm_name}_{metric}"],
                    current_config_results[f"Random_{metric}"],
                    alternative='greater'
                )
                current_config_results[f"{sm_name}_{metric}_pval"] = p_value.item()

        # Only keep the mean after pvals
        for sm_name in saliency_methods:
            for metric in metrics:
                colname = f"{sm_name}_{metric}"
                if colname not in current_config_results:
                    continue
                current_config_results[f"{sm_name}_{metric}"] = np.mean(
                    current_config_results[f"{sm_name}_{metric}"]
                ).item()

        # Save results
        all_results_data.append(current_config_results)

    # Convert to df
    results_df = pd.DataFrame(all_results_data)

    # Define column order for the output CSV
    output_columns = ['model', 'linear', 'pretrained']
    output_columns += sorted(
        [f"{sm}_{metric}" for sm, metric in zip(saliency_methods, metrics)]
    )
    output_columns += sorted(
        [f"{sm}_{metric}_pval" for sm, metric in zip(saliency_methods, metrics)]
    )
    
    results_df = results_df[output_columns]
    
    # Print & save results
    print(results_df.to_string(index=False, float_format="%.4f"))
    csv_output_path = os.path.join(
        "evaluation",
        "model-expert-agreement.csv"
    )
    results_df.to_csv(
        csv_output_path,
        index=False,
        float_format="%.4f"
    )
    print(f"\nResults saved to {csv_output_path}")

if __name__ == "__main__":
    main()