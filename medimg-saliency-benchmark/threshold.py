import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image

import utils
from models import BaseCNN, VGG16Binary
import saliency

# Constants
BASE_ANNOTATIONS_DIR = "data/annotations"
METADATA_PATH = os.path.join(BASE_ANNOTATIONS_DIR, "metadata.json")
ANNOTATED_MASKS_DIR = os.path.join(BASE_ANNOTATIONS_DIR, "annotated")
ORIGINAL_IMAGES_DIR = "data/test"
MODEL_INPUT_SIZE = (224, 224)

# Processing parameters
INITIAL_PRE_CLOSING_KERNEL_SIZE = 3
SOLIDITY_THRESHOLD = 0.6
OUTLINE_FILL_CLOSING_KERNEL_SIZE = 7
OUTLINE_EROSION_KERNEL_SIZE = 7
FILLED_REGION_HOLE_CLOSING_KERNEL_SIZE = 5
MIN_CONTOUR_AREA_FILTER = 20
CONSENSUS_POST_FILTER_TYPE = 'open'
CONSENSUS_POST_FILTER_KERNEL_SIZE = 3
CONSENSUS_METHOD = 'intersection'

# Analysis parameters
MODEL_KEY = "vgg"
THRESHOLDS = np.linspace(0.02, 0.90, 45)

# Load annotations metadata
with open(METADATA_PATH, 'r') as f:
    annotations_metadata_raw = json.load(f)

df_metadata = pd.DataFrame(annotations_metadata_raw)
df_metadata = df_metadata[~df_metadata['annotator_name'].str.contains('test', case=False, na=False)]
annotations_metadata = df_metadata.to_dict(orient='records')

# Get device and load model
device = utils.get_device()
checkpoint_path = utils.find_checkpoint(MODEL_KEY)
pl_model_wrapper = BaseCNN.load_from_checkpoint(checkpoint_path, map_location=device)
model = pl_model_wrapper.model
model.to(device).eval()

# Patch VGG model if needed for CAM
if isinstance(model, VGG16Binary) and not model.linear:
    if isinstance(model.classifier, nn.Sequential):
        model.classifier.weight = model.classifier[-1].weight

# Initialize saliency method
saliency_method = saliency.CAM(model)

# Calculate expert masks
expert_masks_cache = {}
unique_images = sorted(list(set(record['image_name'] for record in annotations_metadata)))

print(f"Pre-calculating expert masks for {len(unique_images)} images...")
for image_name in unique_images:
    raw_masks_tuples = utils.get_masks_for_image_from_metadata(
        image_name, annotations_metadata, ANNOTATED_MASKS_DIR, target_size=MODEL_INPUT_SIZE
    )
    if not raw_masks_tuples:
        continue
    
    current_image_cache = {}
    
    # Full consensus
    processed_masks_list_full = []
    for rm, an in raw_masks_tuples:
        processed_mask = utils.process_circled_annotation(
            rm, INITIAL_PRE_CLOSING_KERNEL_SIZE, SOLIDITY_THRESHOLD, 
            OUTLINE_FILL_CLOSING_KERNEL_SIZE, OUTLINE_EROSION_KERNEL_SIZE, 
            FILLED_REGION_HOLE_CLOSING_KERNEL_SIZE, MIN_CONTOUR_AREA_FILTER
        )
        processed_masks_list_full.append(
            processed_mask if processed_mask is not None else np.zeros(MODEL_INPUT_SIZE, dtype=np.uint8)
        )
    
    consensus_full = utils.create_consensus_mask(
        processed_masks_list_full, CONSENSUS_POST_FILTER_TYPE, 
        CONSENSUS_POST_FILTER_KERNEL_SIZE, CONSENSUS_METHOD
    )
    current_image_cache['full'] = consensus_full if consensus_full is not None else np.zeros(MODEL_INPUT_SIZE, dtype=np.uint8)
    
    # Giovanni only
    giovanni_mask = np.zeros(MODEL_INPUT_SIZE, dtype=np.uint8)
    for rm, an in raw_masks_tuples:
        if "giovanni" in an.lower():
            processed_gio = utils.process_circled_annotation(
                rm, INITIAL_PRE_CLOSING_KERNEL_SIZE, SOLIDITY_THRESHOLD, 
                OUTLINE_FILL_CLOSING_KERNEL_SIZE, OUTLINE_EROSION_KERNEL_SIZE, 
                FILLED_REGION_HOLE_CLOSING_KERNEL_SIZE, MIN_CONTOUR_AREA_FILTER
            )
            if processed_gio is not None and processed_gio.sum() > 0:
                giovanni_mask = processed_gio
            break
    current_image_cache['gio'] = giovanni_mask
    
    # Consensus without Giovanni
    base_processed_no_gio = []
    for rm, an in raw_masks_tuples:
        if "giovanni" not in an.lower():
            processed_mask_no_gio = utils.process_circled_annotation(
                rm, INITIAL_PRE_CLOSING_KERNEL_SIZE, SOLIDITY_THRESHOLD, 
                OUTLINE_FILL_CLOSING_KERNEL_SIZE, OUTLINE_EROSION_KERNEL_SIZE, 
                FILLED_REGION_HOLE_CLOSING_KERNEL_SIZE, MIN_CONTOUR_AREA_FILTER
            )
            base_processed_no_gio.append(
                processed_mask_no_gio if processed_mask_no_gio is not None else np.zeros(MODEL_INPUT_SIZE, dtype=np.uint8)
            )
    
    consensus_no_gio = np.zeros(MODEL_INPUT_SIZE, dtype=np.uint8)
    if base_processed_no_gio:
        temp_consensus_no_gio = utils.create_consensus_mask(
            base_processed_no_gio, CONSENSUS_POST_FILTER_TYPE, 
            CONSENSUS_POST_FILTER_KERNEL_SIZE, CONSENSUS_METHOD
        )
        if temp_consensus_no_gio is not None:
            consensus_no_gio = temp_consensus_no_gio
    current_image_cache['no_gio'] = consensus_no_gio
    
    expert_masks_cache[image_name] = current_image_cache

print(f"Cached expert masks for {len(expert_masks_cache)} images.")

# Calculate IoU for different thresholds
results = []
for threshold in THRESHOLDS:
    print(f"Testing threshold: {threshold:.2f}")
    ious_full, ious_gio, ious_no_gio = [], [], []
    
    for image_name, masks in expert_masks_cache.items():
        # Find image path
        image_path = None
        for path in [
            os.path.join(ORIGINAL_IMAGES_DIR, "PNEUMONIA", image_name),
            os.path.join(ORIGINAL_IMAGES_DIR, "NORMAL", image_name),
            os.path.join(ORIGINAL_IMAGES_DIR, image_name)
        ]:
            if os.path.exists(path):
                image_path = path
                break
        
        if not image_path:
            continue
            
        # Generate saliency map
        input_tensor = utils.load_image_tensor(image_path, device)
        saliency_map = saliency_method(input_tensor)
        bin_saliency = utils.binarize_saliency_map(saliency_map, threshold_value=threshold)
        
        # Calculate IoU for each mask type
        if masks['full'].sum() > 0:
            ious_full.append(utils.calculate_iou(bin_saliency, masks['full']))
        if masks['gio'].sum() > 0:
            ious_gio.append(utils.calculate_iou(bin_saliency, masks['gio']))
        if masks['no_gio'].sum() > 0:
            ious_no_gio.append(utils.calculate_iou(bin_saliency, masks['no_gio']))
    
    # Calculate average IoU
    avg_full = np.nanmean(ious_full) if ious_full else np.nan
    avg_gio = np.nanmean(ious_gio) if ious_gio else np.nan
    avg_no_gio = np.nanmean(ious_no_gio) if ious_no_gio else np.nan
    
    results.append({
        'threshold': threshold,
        'iou_vs_full_consensus': avg_full,
        'iou_vs_giovanni_only': avg_gio,
        'iou_vs_consensus_no_giovanni': avg_no_gio,
        'n_full_consensus': len(ious_full),
        'n_giovanni': len(ious_gio),
        'n_no_giovanni': len(ious_no_gio)
    })

# Plot results
df_results = pd.DataFrame(results)
plt.figure(figsize=(12, 7))
plt.plot(df_results['threshold'], df_results['iou_vs_full_consensus'], marker='o', 
         label=f'IoU vs Full Consensus (n={df_results["n_full_consensus"].iloc[-1]})')
plt.plot(df_results['threshold'], df_results['iou_vs_giovanni_only'], marker='s', 
         label=f'IoU vs Giovanni Only (n={df_results["n_giovanni"].iloc[-1]})')
plt.plot(df_results['threshold'], df_results['iou_vs_consensus_no_giovanni'], marker='^', 
         label=f'IoU vs Consensus (No Gio) (n={df_results["n_no_giovanni"].iloc[-1]})')

plt.xlabel("Saliency Binarization Threshold")
plt.ylabel("Average IoU")
plt.title(f"IoU vs. Binarization Threshold (CAM on {MODEL_KEY.upper()})")
plt.legend()
plt.grid(True)
plt.xticks(np.linspace(min(THRESHOLDS), max(THRESHOLDS), 10))
plt.ylim(bottom=0)

# Save results
os.makedirs("plots", exist_ok=True)
plt.savefig(f"plots/iou_vs_threshold_{MODEL_KEY}_CAM.png")
plt.show()

os.makedirs("evaluation", exist_ok=True)
df_results.to_csv(f"evaluation/iou_vs_threshold_{MODEL_KEY}_CAM.csv", index=False, float_format='%.4f')