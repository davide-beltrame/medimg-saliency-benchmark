import json
import torch
import numpy as np
from lightning.pytorch.callbacks import Callback
from torchmetrics.classification import (
    BinaryAccuracy, 
    BinaryPrecision, 
    BinaryRecall, 
    BinaryF1Score, 
    BinaryAUROC,
    BinarySpecificity
)
import cv2 
from PIL import Image, ImageDraw
import os 
import glob
from torchvision import transforms

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

class BaseConfig:
    """
    Reads a config and creates a class with attributes corresponding to the keys.
    """
    def __init__(self, path_to_config):
        
        # Read config dict
        with open(path_to_config, "r") as f:
            config_dict = json.load(f)
        
        # Set attributes
        for k,v in config_dict.items():
            self.__setattr__(k, v)


class BootstrapTestCallback(Callback):
    """
    Callback to perform bootstrap sampling on test predictions to estimate
    confidence intervals for test metrics.
    
    This callback collects predictions and targets during testing
    and calculates bootstrapped statistics at the end of the test epoch.
    """
    
    def __init__(self, n_bootstrap_samples=1000, confidence_level=0.95, seed=42):
        """
        Args:
            n_bootstrap_samples: Number of bootstrap samples to generate
            confidence_level: Confidence level for interval estimation (default: 0.95)
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.n_bootstrap_samples = n_bootstrap_samples
        self.confidence_level = confidence_level
        self.seed = seed
        self.all_preds = []  # Store probabilities directly instead of logits
        self.all_targets = []
        
        # Initialize torchmetrics for bootstrapping
        self.init_metrics()
        
    def init_metrics(self):
        """Initialize all metrics for a bootstrap sample"""
        self.metrics = {
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "f1": BinaryF1Score(),
            "auroc": BinaryAUROC(),
            "specificity": BinarySpecificity()
        }
        
    def reset_metrics(self):
        """Reset all metrics"""
        for metric in self.metrics.values():
            metric.reset()
        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect predictions and targets after each test batch"""
        X, y = batch
        with torch.no_grad():
            # Get logits and convert to probabilities
            logits = pl_module(X).view(-1)
            preds = torch.nn.functional.sigmoid(logits).view(-1)
            
            # Store predictions and targets
            self.all_preds.extend(preds.cpu().numpy())  # Store probabilities
            self.all_targets.extend(y.cpu().numpy())
    
    def on_test_epoch_end(self, trainer, pl_module):
        """Calculate bootstrapped statistics at the end of testing"""
        # Convert to numpy arrays for easier processing
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)

        # Print distribution statistics to help diagnose
        print("\nTest set statistics:")
        print(f"Number of samples: {len(targets)}")
        print(f"Class distribution: {np.bincount(targets.astype(int))}")
        print(f"Prediction mean: {np.mean(preds):.4f}, std: {np.std(preds):.4f}")
        print(f"Prediction range: [{np.min(preds):.4f}, {np.max(preds):.4f}]")

        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Initialize dictionaries to store bootstrap results
        bootstrap_results = {
            k:np.zeros(self.n_bootstrap_samples) 
            for k,_ in self.metrics.items()
        }

        # Sample size equals original dataset size
        n_samples = len(preds)
        
        # Generate bootstrap samples and calculate metrics
        for i in range(self.n_bootstrap_samples):
            # Reset metrics for this bootstrap iteration
            self.reset_metrics()
            
            # Generate bootstrap sample indices
            indices = np.random.choice(n_samples, n_samples, replace=True)
            
            # Get bootstrap predictions and targets
            bootstrap_preds = torch.tensor(preds[indices]).to(torch.float32)
            bootstrap_targets = torch.tensor(targets[indices]).long()
            
            # Calculate metrics using torchmetrics
            try:
                for name, metric in self.metrics.items():
                    bootstrap_results[name][i] = metric(bootstrap_preds, bootstrap_targets).item()
            except Exception as e:
                print(f"Metrics calculation failed for bootstrap sample {i}: {e}")
                # Keep the default 0 value for this iteration

        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        # Calculate width of confidence intervals for diagnostics
        ci_widths = {}
        
        # Generate log dictionary
        metrics_bootstrap = {}
        for metric_name in bootstrap_results.keys():
            mean_val = np.mean(bootstrap_results[metric_name])
            ci_lower = np.percentile(bootstrap_results[metric_name], lower_percentile)
            ci_upper = np.percentile(bootstrap_results[metric_name], upper_percentile)
            
            metrics_bootstrap[f"test/{metric_name}_mean"] = mean_val
            metrics_bootstrap[f"test/{metric_name}_ci_lower"] = ci_lower
            metrics_bootstrap[f"test/{metric_name}_ci_upper"] = ci_upper
            
            # Calculate width as percentage points for binary metrics
            ci_widths[metric_name] = (ci_upper - ci_lower) * 100
        
        # Log the bootstrap metrics
        pl_module.log_dict(metrics_bootstrap, on_step=False, on_epoch=True)
        
        # Print bootstrap results with additional diagnostics
        print("\n\n===== Bootstrap Test Results =====")
        for metric_name in bootstrap_results.keys():
            print(f"{metric_name.capitalize()}: {metrics_bootstrap[f'test/{metric_name}_mean']:.4f} "
                  f"({metrics_bootstrap[f'test/{metric_name}_ci_lower']:.4f}, "
                  f"{metrics_bootstrap[f'test/{metric_name}_ci_upper']:.4f})")
            print(f"  - CI width: {ci_widths[metric_name]:.2f} percentage points")
            print(f"  - Bootstrap samples std: {np.std(bootstrap_results[metric_name]):.4f}")
        
        # Analyze distribution of bootstrap results for the problematic metrics
        # Find metrics with wide CIs
        wide_ci_metrics = [m for m, w in ci_widths.items() if w > 10.0]
        if wide_ci_metrics:
            print("\n===== Analysis of Wide CI Metrics =====")
            for metric_name in wide_ci_metrics:
                values = bootstrap_results[metric_name]
                print(f"{metric_name} distribution:")
                print(f"  - min: {np.min(values):.4f}, max: {np.max(values):.4f}")
                print(f"  - 10th percentile: {np.percentile(values, 10):.4f}")
                print(f"  - 25th percentile: {np.percentile(values, 25):.4f}")
                print(f"  - median: {np.median(values):.4f}")
                print(f"  - 75th percentile: {np.percentile(values, 75):.4f}")
                print(f"  - 90th percentile: {np.percentile(values, 90):.4f}")
        
        print("================================\n")

        # Clear collected data for next potential test run
        self.all_preds.clear()
        self.all_targets.clear()


def load_mask(mask_path, target_size=(224, 224)):
    """
    Loads a PNG mask image, converts to grayscale, resizes, and binarizes.
    White pixels (255) are foreground, black (0) are background.
    """
    try:
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        mask = mask.resize(target_size, Image.NEAREST)
        mask_np = np.array(mask)
        # Binarize: threshold at 128 (common for L mode if not purely 0/255)
        # If you are sure your masks are strictly 0 and 255, this can be more direct.
        binary_mask = (mask_np > 128).astype(np.uint8) # Foreground is 1, background is 0
        return binary_mask
    except FileNotFoundError:
        print(f"Warning: Mask file not found at {mask_path}")
        return None
    except Exception as e:
        print(f"Warning: Error loading mask {mask_path}: {e}")
        return None
    

def process_circled_annotation(binary_mask_np,
                               initial_closing_kernel_size=3,
                               solidity_threshold=0.6,
                               outline_fill_closing_kernel_size=7, 
                               outline_erosion_kernel_size=7,    
                               filled_region_hole_closing_kernel_size=5, 
                               min_contour_area=20):
    """
    Processes a binary mask that might contain outlines or already filled regions.
    Iterates over all significant contours in the mask.
    - If a contour is an outline (based on solidity), it attempts to close gaps, fill it, 
      and then erode the boundary.
    - If a contour is a filled region, it attempts to close internal holes.
    Results from all processed contours are combined.

    Args:
        binary_mask_np (np.ndarray): Input binary mask (HxW, values 0 or 1).
        initial_closing_kernel_size (int): Kernel size for an initial morphological closing 
                                           to pre-process the mask (e.g., connect tiny breaks).
        solidity_threshold (float): Ratio of contour area to convex hull area. Contours with
                                    solidity below this are treated as outlines.
        outline_fill_closing_kernel_size (int): Kernel size for closing larger gaps in detected outlines
                                                before attempting to fill them.
        outline_erosion_kernel_size (int): Kernel size for erosion applied *only* to filled outlines
                                           to remove the drawn line's thickness.
        filled_region_hole_closing_kernel_size (int): Kernel size for closing internal holes in
                                                      regions already identified as filled.
        min_contour_area (int): Minimum area for a contour to be considered significant.

    Returns:
        np.ndarray: Processed binary mask, or an empty mask if processing fails.
    """
    if binary_mask_np is None:
        # Return a default empty mask of a standard size if input is None
        # Assuming target_size is (224,224) as used elsewhere, but this could be a parameter
        return np.zeros((224,224), dtype=np.uint8) 
    if binary_mask_np.sum() == 0: # If mask is already empty
        return binary_mask_np.astype(np.uint8)

    mask_uint8 = binary_mask_np.astype(np.uint8)

    # 1. Initial (small) closing to connect very minor breaks and smooth the input.
    # This helps in finding more coherent contours.
    if initial_closing_kernel_size > 0:
        temp_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (initial_closing_kernel_size, initial_closing_kernel_size))
        processed_mask_for_contours = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, temp_kernel)
    else:
        processed_mask_for_contours = mask_uint8.copy()

    # 2. Find ALL external contours on this initially processed mask.
    # cv2.RETR_EXTERNAL retrieves only the extreme outer contours.
    # cv2.CHAIN_APPROX_SIMPLE compresses segments, leaving only their end points.
    contours, hierarchy = cv2.findContours(processed_mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros_like(binary_mask_np, dtype=np.uint8) # Return empty if no contours

    # Create a final mask to accumulate results from all processed contours
    final_combined_mask = np.zeros_like(binary_mask_np, dtype=np.uint8)

    # 3. Iterate over ALL found contours
    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_contour_area:
            continue # Skip small, insignificant contours

        # Create a temporary mask for processing this single contour
        # This will hold the processed version of the current contour
        single_contour_processed_mask = np.zeros_like(processed_mask_for_contours, dtype=np.uint8)

        # Calculate Solidity of the current contour
        # Solidity = Area of Contour / Area of Convex Hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0.0
        
        # --- Decision based on solidity for the CURRENT contour ---
        if solidity < solidity_threshold:
            # --- BRANCH 1: Assumed to be an OUTLINE ---
            # The goal is to fill this outline and then erode by the line thickness.
            
            # a. Create a mask with just the current contour line.
            #    Then, apply a more aggressive closing to ensure the outline is connected for filling.
            current_outline_mask = np.zeros_like(processed_mask_for_contours, dtype=np.uint8)
            cv2.drawContours(current_outline_mask, [contour], -1, 1, thickness=1) # Draw the line of this contour

            if outline_fill_closing_kernel_size > 0:
                close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outline_fill_closing_kernel_size, outline_fill_closing_kernel_size))
                closed_outline_for_filling = cv2.morphologyEx(current_outline_mask, cv2.MORPH_CLOSE, close_kernel)
            else:
                closed_outline_for_filling = current_outline_mask # Use as is if no closing
            
            # b. Fill this (hopefully now closed) outline.
            #    Find contours on this specifically prepared mask and fill them.
            fill_contours_for_this_outline, _ = cv2.findContours(closed_outline_for_filling, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if fill_contours_for_this_outline:
                # Filter by area again, in case closing created tiny artifacts or multiple small fills
                # Use a relative threshold based on the original contour's area or a small absolute one
                min_fill_area = max(min_contour_area / 4.0, area / 10.0) 
                significant_fill_contours = [fc for fc in fill_contours_for_this_outline if cv2.contourArea(fc) > min_fill_area]
                if significant_fill_contours:
                     cv2.drawContours(single_contour_processed_mask, significant_fill_contours, -1, 1, thickness=cv2.FILLED)
            
            # c. Erode the filled shape to account for the original line thickness.
            if outline_erosion_kernel_size > 0 and single_contour_processed_mask.sum() > 0:
                erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outline_erosion_kernel_size, outline_erosion_kernel_size))
                single_contour_processed_mask = cv2.erode(single_contour_processed_mask, erode_kernel, iterations=1)
            
        else:
            # --- BRANCH 2: Assumed to be a FILLED REGION ---
            # The region is likely already filled. The main task is to ensure any internal holes are closed.
            # We use the 'contour' found from the 'processed_mask_for_contours'.
            cv2.drawContours(single_contour_processed_mask, [contour], -1, 1, thickness=cv2.FILLED)

            # b. Apply morphological closing to fill internal holes.
            if filled_region_hole_closing_kernel_size > 0 and single_contour_processed_mask.sum() > 0:
                hole_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filled_region_hole_closing_kernel_size, filled_region_hole_closing_kernel_size))
                single_contour_processed_mask = cv2.morphologyEx(single_contour_processed_mask, cv2.MORPH_CLOSE, hole_close_kernel)
        
        # Add the processed result of this contour to the final combined mask
        # This ensures that if multiple regions were drawn, all processed versions are included.
        if single_contour_processed_mask.sum() > 0:
            final_combined_mask = np.logical_or(final_combined_mask, single_contour_processed_mask).astype(np.uint8)
            
    return final_combined_mask
    

def get_masks_for_image_from_metadata(image_name_to_find, annotations_metadata, annotated_masks_dir, target_size=(224, 224)):
    """
    Finds and loads all individual expert masks for a given image_name using metadata.
    
    Args:
        image_name_to_find (str): The filename of the original image (e.g., "person171_bacteria_826.jpeg").
        annotations_metadata (list): A list of dictionaries, where each dict is an entry from metadata.json.
        annotated_masks_dir (str): Path to the directory containing the '.png' mask files (e.g., "data/annotations/annotated/").
        target_size (tuple): The (width, height) to resize masks to.

    Returns:
        list: A list of loaded binary masks (numpy arrays) for the given image.
              Each mask is a 2D numpy array (H, W) with values 0 or 1.
              Returns an empty list if no masks are found or if errors occur.
    """
    loaded_masks_with_annotators = [] # Stores tuples of (mask_array, annotator_name)
    mask_paths_found = []
    for record in annotations_metadata:
        if record.get("image_name") == image_name_to_find:
            annotation_filename = record.get("annotation_file")
            annotator_name = record.get("annotator_name", "Unknown Annotator") # Get annotator name
            if annotation_filename:
                mask_path = os.path.join(annotated_masks_dir, annotation_filename)
                mask_paths_found.append(mask_path)
                mask = load_mask(mask_path, target_size=target_size)
                if mask is not None:
                    loaded_masks_with_annotators.append((mask, annotator_name)) # Store tuple

    return loaded_masks_with_annotators


def apply_morphological_filter(mask, operation='open', kernel_size=3):
    """
    Applies morphological filtering to a binary mask.
    mask: input binary mask (numpy array HxW, values 0 or 1).
    operation: 'open' (erosion then dilation) or 'close' (dilation then erosion).
    kernel_size: size of the structuring element.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        print("Warning: Invalid mask for morphological filter, skipping.")
        return mask

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'open':
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    else:
        print(f"Warning: Unknown morphological operation '{operation}'. Returning original mask.")
        return mask
    

def create_consensus_mask(individual_masks, filter_type='open', filter_kernel_size=3, consensus_method='intersection'):
    """
    Creates a consensus mask from a list of individual binary masks.
    If consensus_method is 'intersection', an empty mask from any participant
    (after initial processing via process_circled_annotation) will result in an empty consensus.
    The 'filter_type' (e.g. 'open') is applied *after* this initial check for intersection.
    """
    default_empty_shape = (224, 224) # Fallback shape

    if not individual_masks:
        return np.zeros(default_empty_shape, dtype=np.uint8)

    reference_shape = None
    for m in individual_masks:
        if m is not None and isinstance(m, np.ndarray):
            reference_shape = m.shape
            break
    if reference_shape is None:
        reference_shape = default_empty_shape

    # Standardize masks: ensure all are ndarray of reference_shape, None becomes empty
    # This list will contain a mask for *every* expert, even if it's empty.
    standardized_masks_for_all_experts = []
    for m_idx, m in enumerate(individual_masks):
        if m is None:
            standardized_masks_for_all_experts.append(np.zeros(reference_shape, dtype=np.uint8))
        elif isinstance(m, np.ndarray) and m.shape == reference_shape:
            standardized_masks_for_all_experts.append(m.astype(np.uint8))
        elif isinstance(m, np.ndarray): # Shape mismatch
            # print(f"Warning: Mask {m_idx} shape mismatch {m.shape} vs {reference_shape}. Treating as empty.")
            standardized_masks_for_all_experts.append(np.zeros(reference_shape, dtype=np.uint8))
        else: # Not an ndarray
            standardized_masks_for_all_experts.append(np.zeros(reference_shape, dtype=np.uint8))
    
    if not standardized_masks_for_all_experts: # Should be caught by initial check
        return np.zeros(reference_shape, dtype=np.uint8)

    # --- Crucial logic for intersection: if any expert's standardized mask is empty, intersection is empty ---
    if consensus_method == 'intersection':
        for i, m_expert in enumerate(standardized_masks_for_all_experts):
            if m_expert.sum() == 0:
                # print(f"Debug: Expert mask {i} is empty. Intersection result will be empty.")
                return np.zeros(reference_shape, dtype=np.uint8)

    # Apply the morphological filter (e.g., 'open') to each (now guaranteed non-empty for intersection) standardized mask
    # For UNION, empty masks will just be OR'd with others.
    masks_after_internal_filter = []
    if filter_type and filter_kernel_size > 0:
        for m_expert in standardized_masks_for_all_experts:
            # Only apply filter if mask has content, or if it's union (where filter might still be desired on non-empty ones)
            if m_expert.sum() > 0 or consensus_method == 'union':
                filtered_m = apply_morphological_filter(m_expert, operation=filter_type, kernel_size=filter_kernel_size)
                masks_after_internal_filter.append(filtered_m if filtered_m is not None else np.zeros(reference_shape, dtype=np.uint8))
            else: # For intersection, this path shouldn't be hit if an empty mask was found above. For others, carry empty.
                masks_after_internal_filter.append(m_expert.copy())
    else: 
        masks_after_internal_filter = [m.copy() for m in standardized_masks_for_all_experts]

    # --- Second check for intersection: if filter_type made a mask empty ---
    # This is important because the 'open' operation can remove small regions entirely.
    if consensus_method == 'intersection':
        for i, m_filtered in enumerate(masks_after_internal_filter):
            if m_filtered.sum() == 0:
                # print(f"Debug: Mask {i} became empty after filter '{filter_type}'. Intersection result will be empty.")
                return np.zeros(reference_shape, dtype=np.uint8)
    
    if not masks_after_internal_filter: # Should not happen
         return np.zeros(reference_shape, dtype=np.uint8)
    
    # --- Perform Consensus ---
    if consensus_method == 'intersection':
        # At this point, for intersection, all masks in masks_after_internal_filter are non-empty.
        consensus_result = masks_after_internal_filter[0].copy()
        for i in range(1, len(masks_after_internal_filter)):
            consensus_result = np.logical_and(consensus_result, masks_after_internal_filter[i]).astype(np.uint8)
    elif consensus_method == 'union':
        # Start with an empty mask for union to correctly accumulate all positive pixels
        consensus_result = np.zeros(reference_shape, dtype=np.uint8)
        for m_filtered in masks_after_internal_filter:
            if m_filtered is not None: # Ensure m_filtered is not None before logical_or
                 consensus_result = np.logical_or(consensus_result, m_filtered).astype(np.uint8)
    else:
        raise ValueError(f"Unknown consensus_method: {consensus_method}")
        
    return consensus_result


def overlay_binary_mask(background_img_pil, mask_np, mask_color=(255, 0, 0), alpha=0.5):
    """
    Overlays a binary mask on a background PIL image.
    
    Args:
        background_img_pil (PIL.Image): The background image.
        mask_np (np.ndarray): The binary mask (HxW, values 0 or 1).
        mask_color (tuple): RGB color for the mask.
        alpha (float): Transparency of the mask (0.0 fully transparent, 1.0 fully opaque).
        
    Returns:
        PIL.Image: Image with mask overlaid.
    """
    if mask_np is None:
        return background_img_pil

    # Ensure background is RGBA for alpha blending
    background_img_pil = background_img_pil.convert("RGBA")
    overlay_img = Image.new("RGBA", background_img_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_img)

    # Scale mask to background image size if different
    if mask_np.shape[:2] != (background_img_pil.height, background_img_pil.width):
        mask_np_resized = cv2.resize(mask_np.astype(np.uint8), (background_img_pil.width, background_img_pil.height), interpolation=cv2.INTER_NEAREST)
    else:
        mask_np_resized = mask_np.astype(np.uint8)

    # Create a colored version of the mask
    for y in range(mask_np_resized.shape[0]):
        for x in range(mask_np_resized.shape[1]):
            if mask_np_resized[y, x] == 1: # If mask is active at this pixel
                draw.point((x, y), fill=(*mask_color, int(alpha * 255)))
                
    # Alpha composite the overlay onto the background
    combined_img = Image.alpha_composite(background_img_pil, overlay_img)
    return combined_img.convert("RGB") # Convert back to RGB if needed


def calculate_iou(mask1, mask2):
    """Calculates Intersection over Union (IoU) for two binary masks."""
    if mask1 is None or mask2 is None: return 0.0
    if mask1.shape != mask2.shape:
        raise ValueError("Masks must have the same shape for IoU calculation.")
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def pointing_game(saliency_map, expert_mask, threshold_saliency=None):
    """
    Performs the Pointing Game evaluation.
    Checks if the pixel with the maximum value in the saliency map
    falls within the expert-annotated region (binarized).

    saliency_map: 2D numpy array (float, 0-1).
    expert_mask: 2D numpy array (binary, 0 or 1).
    threshold_saliency: Optional. If provided, binarizes saliency_map first.
                        Usually, Pointing Game uses the raw max point.
    Returns: 1 if the max saliency point is in the expert mask, 0 otherwise.
    """
    if saliency_map is None or expert_mask is None: return 0.0
    if saliency_map.shape != expert_mask.shape:
        # Attempt to resize saliency_map to expert_mask shape if they differ
        # This can happen if GradCAM output size is slightly different
        saliency_map = cv2.resize(saliency_map, (expert_mask.shape[1], expert_mask.shape[0]), interpolation=cv2.INTER_LINEAR)

    if threshold_saliency is not None:
        saliency_map = (saliency_map >= threshold_saliency).astype(np.uint8)

    # Find the coordinates of the maximum value in the saliency map
    # If multiple maxima, np.unravel_index gives the first one.
    max_coords = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)

    # Check if this point is within the expert mask (where expert_mask == 1)
    if expert_mask[max_coords] == 1:
        return 1.0
    else:
        return 0.0


def find_checkpoint(model_short_key):
    """
    Finds a checkpoint file for a given model short key (e.g., "an", "vgg", "rn", "in").
    If multiple checkpoints match (e.g., different training parameters or epochs),
    it currently picks the first one found by glob, sorted alphabetically.
    You might want to add logic to pick the "best" one if scores are in filenames.

    Args:
        model_short_key (str): The short key for the model (e.g., "an", "vgg", "rn", "in").

    Returns:
        str or None: Path to the checkpoint file, or None if not found.
    """
    if not model_short_key:
        print("Warning: model_short_key is empty in find_checkpoint.")
        return None
        
    # Construct the pattern, e.g., "an_*.ckpt"
    # This assumes filenames like "an_True_True_0.05.ckpt"
    ckpt_pattern = os.path.join(CHECKPOINT_DIR, f"{model_short_key}_*.ckpt")
    ckpts = sorted(glob.glob(ckpt_pattern)) # Sort for consistency

    if ckpts:        
        selected_ckpt = ckpts[0] # Pick the first one (e.g., lowest loss if sorted by loss, or just first alphabetically)
        print(f"Found checkpoint for '{model_short_key}': {selected_ckpt}")
        return selected_ckpt
    else:
        if model_short_key == "vgg":
            ckpt_pattern_alt = os.path.join(CHECKPOINT_DIR, "vgg*.ckpt")
            ckpts_alt = sorted(glob.glob(ckpt_pattern_alt))
            if ckpts_alt:
                print(f"Found checkpoint for '{model_short_key}' using alternative pattern: {ckpts_alt[0]}")
                return ckpts_alt[0]

        print(f"Warning: No checkpoint found for model key '{model_short_key}' with pattern '{ckpt_pattern}' in {CHECKPOINT_DIR}")
        return None
    

def load_image_tensor(image_path, device):
    """Loads an image and converts it to a tensor for model input."""
    try:
        img = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(MODEL_INPUT_SIZE),
            transforms.ToTensor() # Scales to [0,1]
        ])
        img_tensor = transform(img).unsqueeze(0) # Add batch dimension
        return img_tensor.to(device)
    except FileNotFoundError:
        print(f"Warning: Image not found at {image_path}")
        return None
    
def load_image_np(image_path):
    """Loads an image and resize it."""
    try:
        img = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(MODEL_INPUT_SIZE)
        ])
        img_tensor = transform(img) # Add batch dimension
        return np.array(img_tensor)
    except FileNotFoundError:
        print(f"Warning: Image not found at {image_path}")
        return None

def binarize_saliency_map(saliency_map_np, method="fixed", threshold_value=SALIENCY_BINARIZATION_THRESHOLD):
    if saliency_map_np is None:
        return None

    saliency_map_to_process = saliency_map_np.copy()
    if saliency_map_to_process.max() == saliency_map_to_process.min(): # Avoid issues with flat maps
        return np.zeros_like(saliency_map_to_process, dtype=np.uint8)

    # Normalize to 0-255 for Otsu if it's in [0,1]
    if saliency_map_to_process.max() <= 1.0 and saliency_map_to_process.min() >=0.0:
         saliency_map_uint8 = np.uint8(255 * saliency_map_to_process)
    else: # If already potentially 0-255 or other range, ensure it's scaled if needed.
          # For safety, let's rescale based on its own min/max if not in [0,1]
         saliency_map_uint8 = np.uint8(255 * (saliency_map_to_process - saliency_map_to_process.min()) / (saliency_map_to_process.max() - saliency_map_to_process.min() + 1e-8))

    if method == "otsu":
        otsu_threshold, binarized_map = cv2.threshold(
            saliency_map_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return (binarized_map / 255).astype(np.uint8)
    elif method == "fixed":
        return (saliency_map_np >= threshold_value).astype(np.uint8)
    else:
        print(f"Warning: Unknown binarization method '{method}'. Using fixed threshold.")
        return (saliency_map_np >= threshold_value).astype(np.uint8)
    

def generate_random_map(size=MODEL_INPUT_SIZE, grid_size=10):
    """Generates a random saliency map as per table description."""
    random_map_small = np.zeros((grid_size, grid_size), dtype=np.float32)
    # Pick one random pixel in the small grid to activate
    rand_x, rand_y = np.random.randint(0, grid_size, 2)
    random_map_small[rand_y, rand_x] = 1.0 
    # Upsample to full size
    random_map_full = cv2.resize(random_map_small, size, interpolation=cv2.INTER_NEAREST)
    return random_map_full # Already 0 or 1, effectively binarized by using inter_nearest

def generate_random_mask_like(mask, grid_size, nonzero_perc):
    """Generates a random mask with shape as the one."""
    H, W = mask.shape
    assert H == W
    random_mask_small = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Pick one random pixel in the small grid to activate
    n = int(grid_size*grid_size*nonzero_perc)
    for _ in range(n):
        rand_x, rand_y = np.random.randint(0, grid_size, 2)
        random_mask_small[rand_y, rand_x] = 1.0 
    # Upsample to full size
    random_mask_full = cv2.resize(random_mask_small, (H, W))
    return random_mask_full.round()