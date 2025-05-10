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
from PIL import Image
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
        self.all_logits = []
        self.all_targets = []
        
        # Initialize torchmetrics for bootstrapping
        self.init_metrics()
        
    def init_metrics(self):
        """Initialize all metrics for a bootstrap sample"""
        self.metrics = {
            "accuracy": BinaryAccuracy(),       # (TP + TN) / (TP+FP + TN+FN)
            "precision": BinaryPrecision(),     # TP / (TP + FP)    -> how many positive are correctly identified
            "recall": BinaryRecall(),           # TP / (TP + FN)    -> how many positive I am missing
            "f1": BinaryF1Score(),              # (2 * precision * recall) / (precision + recall)
            "auroc": BinaryAUROC(),             
            "specificity": BinarySpecificity()  # TN / (TN + FP)    -> detect if prdicting al positives
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
            # preds = torch.nn.functional.sigmoid(logits).view(-1)
            
            # Store predictions and targets
            self.all_logits.extend(logits.cpu().numpy())
            self.all_targets.extend(y.cpu().numpy())
    
    def on_test_epoch_end(self, trainer, pl_module):
        """Calculate bootstrapped statistics at the end of testing"""
        # Convert to numpy arrays for easier processing
        logits = np.array(self.all_logits)
        targets = np.array(self.all_targets)

        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Initialize dictionaries to store bootstrap results
        bootstrap_results = {
            k:np.zeros(self.n_bootstrap_samples) 
            for k,_ in self.metrics.items()
        }

        # Sample size equals original dataset size
        n_samples = len(logits)
        
        # Generate bootstrap samples and calculate metrics
        for i in range(self.n_bootstrap_samples):
            # Reset metrics for this bootstrap iteration
            self.reset_metrics()
            
            # Generate bootstrap sample indices
            indices = np.random.choice(n_samples, n_samples, replace=True)
            
            # Get bootstrap predictions and targets
            bootstrap_logits = torch.tensor(logits[indices]).to(torch.float32)
            bootstrap_targets = torch.tensor(targets[indices]).long()
            
            # Calculate metrics using torchmetrics
            try:
                for name, metric in self.metrics.items():
                    # All metrics handle automatic sigmoid if logits
                    bootstrap_results[name][i] = metric(bootstrap_logits, bootstrap_targets).item()
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
        self.all_logits.clear()
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
    

def find_individual_masks(image_filename_stem, annotations_dir):
    """
    Finds all individual expert masks for a given image stem.

    (REQUIRES RENAMING MASKS) Assumes masks are named like: [image_stem]_expert[ID]_mask.png or [image_stem]_mask_expert[ID].png
    or simply [image_stem]_mask.png if each expert's annotations are in their own subfolder of annotations_dir.

    For simplicity, let's assume a common pattern: annotations_dir contains masks
    named `[image_filename_stem]_ANYTHING_mask.png` or `[image_filename_stem]_mask_ANYTHING.png`.
    A more robust solution would be to know the exact naming convention.

    If your naming is simply `[image_filename_stem]_mask.png` but in different expert subdirectories:
    e.g., annotations_dir/expert1/[image_filename_stem]_mask.png
          annotations_dir/expert2/[image_filename_stem]_mask.png
    This function would need to be adapted to walk through subdirectories.

    Current assumption: All relevant masks for an image are in `annotations_dir`
    and can be identified by `image_filename_stem` and `_mask.png` suffix, possibly with expert identifiers in between.
    Example: `person1_bacteria_1_expertA_mask.png`, `person1_bacteria_1_expertB_mask.png`
    """
    mask_paths = []
    # Adjusted glob pattern: accounts for variations like imagename_expertID_mask.png or imagename_sometag_mask.png
    # It will find files starting with image_filename_stem, containing "_mask" and ending with ".png"
    # To be more specific, you might use:
    # search_pattern = os.path.join(annotations_dir, f"{image_filename_stem}_*_mask.png")
    # For now, a bit more general if only one _mask.png exists per expert for that image_filename_stem
    search_pattern_exact = os.path.join(annotations_dir, f"{image_filename_stem}_mask.png") # if only one per image
    
    # This pattern will find imagename_expert1_mask.png, imagename_expert2_mask.png etc.
    search_pattern_glob = os.path.join(annotations_dir, f"{image_filename_stem}*mask.png")

    # Check if multiple expert folders exist within annotations_dir
    potential_expert_dirs = [d for d in os.listdir(annotations_dir) if os.path.isdir(os.path.join(annotations_dir, d))]
    if potential_expert_dirs:
        for expert_dir in potential_expert_dirs:
            # Assumes mask name is image_filename_stem + "_mask.png" inside expert folder
            mask_file = os.path.join(annotations_dir, expert_dir, f"{image_filename_stem}_mask.png")
            if os.path.exists(mask_file):
                mask_paths.append(mask_file)
    
    if not mask_paths: # If no expert subdirectories or masks not found there
        # Fallback to globbing directly in annotations_dir
        # This pattern is broad: image_filename_stem<anything_including_nothing>mask.png
        # e.g. image_stem_mask.png, image_stem_expert1_mask.png
        for path in glob.glob(search_pattern_glob):
             # Ensure it's truly for this stem and not a longer one, e.g. image_stem_extra_mask.png
            if os.path.basename(path).startswith(image_filename_stem) and "_mask.png" in os.path.basename(path):
                 mask_paths.append(path)
        
        # If only one mask, it might be `imagename_mask.png`
        if not mask_paths and os.path.exists(search_pattern_exact):
            mask_paths.append(search_pattern_exact)

    if not mask_paths:
        print(f"Warning: No masks found for image stem {image_filename_stem} in {annotations_dir} with pattern {search_pattern_glob} or in subdirs.")
    
    loaded_masks = []
    for path in mask_paths:
        mask = load_mask(path)
        if mask is not None:
            loaded_masks.append(mask)
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