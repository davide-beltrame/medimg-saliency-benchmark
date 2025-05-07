"""
Implements the saliency methods.
Saliency methods always take as input a model when initialized.
The forward pass takes as input an image and returns the saliency map.
"""

import numpy as np
import cv2

class CAM:
    """
    Class Activation Map (CAM) saliency map.
    Requires GAP layer + FC in the original model.
    """
    def __init__(self):
        pass

class GradCAM:
    """
    Grad-CAM saliency map.  
    Can be applied to any model.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            # use .detach() to avoid memory leaks
            # https://github.com/pytorch/pytorch/issues/12863
            self.gradients = grad_output[0].detach()    

        # Hook to capture activations
        # The hook will be called every time after forward() 
        # has computed an output. 
        # It should have the following signature:
        # hook(module, input, output) -> None or modified output
        def forward_hook(module, input, output):
            self.activations = output

        # Register hooks on the last conv layer
        target_layer = self.model.features[-1]  # Last convolution layer
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        input_tensor.requires_grad = True
        output = self.model(input_tensor)  # forward
        if class_idx is None:
            class_idx = (output > 0).float().item()
        loss = output[0, 0]
        self.model.zero_grad()
        loss.backward()

        # Grad-CAM computation
        gradients = self.gradients[0].detach().cpu().numpy()         # [C, H, W]
        activations = self.activations[0].detach().cpu().numpy()     # [C, H, W]
        weights = np.mean(gradients, axis=(1, 2))           # [C]
        cam = np.sum(weights[:, None, None] * activations, axis=0)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.size(3), input_tensor.size(2)))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam  # [H, W] in [0, 1]

class ScoreCAM:
    """
    Randomized input sampling for explanation (RISE).
    Can be applied to any model.
    
    Apply random binary mask on input, the predicted class probability is a weight for the mask.
    Aggregate multiple masks to obtain a saliency map
    (masks hiding important features will have low weights, hence will be revelaed in the aggregation.
    """
    def __init__(self):
        pass