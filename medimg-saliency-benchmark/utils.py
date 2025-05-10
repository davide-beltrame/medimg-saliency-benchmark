import json
import torch
import numpy as np
from lightning.pytorch.callbacks import Callback
from torchmetrics.classification import (
    BinaryAccuracy, 
    BinaryPrecision, 
    BinaryRecall, 
    BinaryF1Score, 
    AUROC
)
import cv2 
from PIL import Image, ImageDraw
import os 
import glob 

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
        self.all_preds = []
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
            "auroc": AUROC(task="binary")
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
            logits = pl_module(X)
            preds = torch.nn.functional.sigmoid(logits).view(-1)
            
            # Store predictions and targets
            self.all_preds.extend(preds.cpu().numpy())
            self.all_targets.extend(y.cpu().numpy())
    
    def on_test_epoch_end(self, trainer, pl_module):
        """Calculate bootstrapped statistics at the end of testing"""
        # Convert to numpy arrays for easier processing
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Initialize dictionaries to store bootstrap results
        bootstrap_results = {
            "accuracy": np.zeros(self.n_bootstrap_samples),
            "precision": np.zeros(self.n_bootstrap_samples),
            "recall": np.zeros(self.n_bootstrap_samples),
            "f1": np.zeros(self.n_bootstrap_samples),
            "auroc": np.zeros(self.n_bootstrap_samples)
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
            bootstrap_preds = torch.tensor(preds[indices])
            bootstrap_targets = torch.tensor(targets[indices]).long()
            
            # Calculate metrics using torchmetrics
            try:
                for name, metric in self.metrics.items():
                    # Different handling for AUROC which needs probabilities
                    if name == "auroc":
                        bootstrap_results[name][i] = metric(bootstrap_preds, bootstrap_targets).item()
                    else:
                        # Other metrics use binary predictions
                        binary_preds = (bootstrap_preds >= 0.5).int()
                        bootstrap_results[name][i] = metric(binary_preds, bootstrap_targets).item()
            except Exception as e:
                print(f"Metrics calculation failed for bootstrap sample {i}: {e}")
                # Keep the default 0 value for this iteration
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        # Generate log dictionary
        metrics_bootstrap = {}
        for metric_name in bootstrap_results.keys():
            metrics_bootstrap[f"test/{metric_name}_mean"] = np.mean(bootstrap_results[metric_name])
            metrics_bootstrap[f"test/{metric_name}_ci_lower"] = np.percentile(bootstrap_results[metric_name], lower_percentile)
            metrics_bootstrap[f"test/{metric_name}_ci_upper"] = np.percentile(bootstrap_results[metric_name], upper_percentile)
        
        # Log the bootstrap metrics
        pl_module.log_dict(metrics_bootstrap, on_step=False, on_epoch=True)
        
        # Print bootstrap results
        print("\n\n===== Bootstrap Test Results =====")
        for metric_name in bootstrap_results.keys():
            print(f"{metric_name.capitalize()}: {metrics_bootstrap[f'test/{metric_name}_mean']:.4f} "
                  f"({metrics_bootstrap[f'test/{metric_name}_ci_lower']:.4f}, "
                  f"{metrics_bootstrap[f'test/{metric_name}_ci_upper']:.4f})")
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
    loaded_masks = []
    mask_paths_found = [] # To store paths for debugging or info

    for record in annotations_metadata:
        if record.get("image_name") == image_name_to_find:
            annotation_filename = record.get("annotation_file")
            if annotation_filename:
                mask_path = os.path.join(annotated_masks_dir, annotation_filename)
                mask_paths_found.append(mask_path)
                mask = load_mask(mask_path, target_size=target_size)
                if mask is not None:
                    loaded_masks.append(mask)
            else:
                print(f"Warning: Record for {image_name_to_find} found but 'annotation_file' is missing or empty.")
    
    if not mask_paths_found:
        print(f"Info: No annotation records found for image_name: {image_name_to_find} in metadata.")
    elif not loaded_masks and mask_paths_found:
        print(f"Warning: Found records for {image_name_to_find} but failed to load any masks from paths: {mask_paths_found}")

    return loaded_masks


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
    individual_masks: A list of 2D numpy arrays (binary masks).
    filter_type: 'open', 'close', or None. Applied to each mask before consensus.
    filter_kernel_size: Kernel size for morphological filter.
    consensus_method: 'intersection' or 'union'.
    """
    if not individual_masks:
        return None

    processed_masks = []
    for mask in individual_masks:
        if filter_type:
            mask_filtered = apply_morphological_filter(mask, operation=filter_type, kernel_size=filter_kernel_size)
            processed_masks.append(mask_filtered)
        else:
            processed_masks.append(mask)
    
    if not processed_masks:
        return None

    if consensus_method == 'intersection':
        # Start with the first mask, then intersect with the rest
        consensus = processed_masks[0].copy()
        for i in range(1, len(processed_masks)):
            consensus = np.logical_and(consensus, processed_masks[i]).astype(np.uint8)
    elif consensus_method == 'union':
        consensus = processed_masks[0].copy()
        for i in range(1, len(processed_masks)):
            consensus = np.logical_or(consensus, processed_masks[i]).astype(np.uint8)
    else:
        raise ValueError(f"Unknown consensus_method: {consensus_method}")

    return consensus


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