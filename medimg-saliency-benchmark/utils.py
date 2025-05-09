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