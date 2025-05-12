"""
Implements the saliency methods.
Saliency methods always take as input a model when initialized.
The forward pass takes as input an image and returns the saliency map.
"""
import torch
import numpy as np
import cv2
from models import (
    AlexNetBinary,
    VGG16Binary,
    ResNet50Binary,
    InceptionNetBinary
)
from tqdm import tqdm
class CAM:
    """
    Class Activation Map (CAM) saliency map.
    Requires GAP layer + FC in the original model.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Hook to capture activations
        def forward_hook(module, input, output):
            # [B, C, H, W]
            self.activations = output.clone()

        # Register hooks on the last conv layer
        if isinstance(model, AlexNetBinary):
            target_layer = self.model.features[10]
            # Check if classifier is a single layer or sequential
            if isinstance(self.model.classifier, torch.nn.Linear):
                self.weights = self.model.classifier.weight
            else:
                self.weights = self.model.classifier[-1].weight
        elif isinstance(model, VGG16Binary):
            target_layer = self.model.features[28] 
            # Check if classifier is a single layer or sequential
            if isinstance(self.model.classifier, torch.nn.Linear):
                self.weights = self.model.classifier.weight
            else:
                self.weights = self.model.classifier[-1].weight
        elif isinstance(model, ResNet50Binary):
            target_layer = self.model.model.layer4[-1].conv3 
            self.weights = self.model.model.fc.weight
        elif isinstance(model, InceptionNetBinary):
            target_layer = self.model.model.inception5b
            self.weights = self.model.model.fc.weight
        else:
            raise NotImplementedError(f"Unrecognized model instance {type(model)}.")
        
        # Register the hooks to capture activations
        target_layer.register_forward_hook(forward_hook)

    def __call__(self, input_tensor):

        assert input_tensor.dim() == 4
        assert input_tensor.shape[0] == 1
        assert input_tensor.shape[-2] == input_tensor.shape[-1]

        # Forward the input
        # we don't care about the output, we only
        # need the activation which will be weighted
        # by the FC weights (since only one class, we
        # don't even need to choose the correct weight group)
        _ = self.model(input_tensor)
       
        # [C, H, W]
        activations = self.activations[0].detach().cpu().numpy()
        # Compute the weight of each channel from the 
        # parameters of the network
        # [C]
        weights = self.weights[0].detach().cpu().numpy()         
        # Expand the average to the correct shape for broadcasting
        # [C, H, W]
        weights = weights[:, None, None]
        # Take the weighted sum
        cam = np.sum(weights * activations, axis=0)
        # Apply relu
        # replace each element c in cam with max(c, 0)
        cam = np.maximum(cam, 0)
        # Upsample
        # default uses bilinear interpolation (cv2.INTER_LINEAR)
        cam = cv2.resize(
            src = cam,
            dsize=(input_tensor.size(3), input_tensor.size(2))
        )
        # Rescale to be in [0,1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        # [H, W] in [0, 1]
        return cam  

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
        # The hook will be called every time after backward() 
        # has computed a gradient. 
        def backward_hook(module, grad_input, grad_output):
            # use .detach() to avoid memory leaks
            # https://github.com/pytorch/pytorch/issues/12863
            # grad_output is a tuple of 1 tensor
            # [B, C, H, W]
            self.gradients = grad_output[0].detach().clone()    

        # Hook to capture activations
        # The hook will be called every time after forward() 
        # has computed an output. 
        # It should have the following signature:
        # hook(module, input, output) -> None or modified output
        def forward_hook(module, input, output):
            # [B, C, H, W]
            self.activations = output.clone()

        # Register hooks on the last conv layer
        if isinstance(model, AlexNetBinary) or isinstance(model, VGG16Binary):
            target_layer = self.model.features[-1]  # Last convolution layer
        elif isinstance(model, ResNet50Binary):
            target_layer = self.model.model.layer4[-1].conv3 
        elif isinstance(model, InceptionNetBinary):
            target_layer = self.model.model.inception5b
        else:
            raise NotImplementedError(f"Unrecognized model instance {type(model)}.")
        
        # Register the hooks to capture gradients & activations
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor):

        assert input_tensor.dim() == 4
        assert input_tensor.shape[0] == 1
        assert input_tensor.shape[-2] == input_tensor.shape[-1]

        # To store the grad wrt input 
        # (what we need to compute gradcam)
        input_tensor.requires_grad = True

        # Forward the input
        # (activation map is stored in self by the hook)
        output = self.model(input_tensor)

        # Compute gradient wrt to the output logit 
        # (gradient map is stored in self by the hook)
        loss = output[0, 0]
        self.model.zero_grad()
        loss.backward()

        # Grad-CAM computation
        # [C, H, W]
        gradients = self.gradients[0].detach().cpu().numpy()         
        # [C, H, W]
        activations = self.activations[0].detach().cpu().numpy()
        # Compute the weight of each channel as the average gradient
        # [C]
        weights = np.mean(gradients, axis=(1, 2))           
        # Expand the average to the correct shape for broadcasting
        # [C, H, W]
        weights = weights[:, None, None]
        # Take the weighted sum
        cam = np.sum(weights * activations, axis=0)
        # Apply relu
        # replace each element c in cam with max(c, 0)
        cam = np.maximum(cam, 0)
        # Upsample
        # default uses bilinear interpolation (cv2.INTER_LINEAR)
        cam = cv2.resize(
            src = cam,
            dsize=(input_tensor.size(3), input_tensor.size(2))
        )
        # Rescale to be in [0,1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        # [H, W] in [0, 1]
        return cam  

class RISE:
    """
    Randomized Input Sampling for Explanation (RISE).
    Apply random binary masks on input and use predicted class probabilities as weights.
    Aggregate multiple masks to obtain a saliency map.
    """
    def __init__(self, model, num_masks=8000, input_size=224, scale_factor=20, p1=0.3):
        self.model = model
        self.model.eval()
        self.num_masks = num_masks
        self.input_size = input_size
        self.scale_factor = scale_factor
        self.p1 = p1  # Probability of 1 in the binary mask
        self.masks = None
        
    def generate_masks(self):
        # Generate smaller masks and upsample them
        cell_size = self.scale_factor
        grid_size = self.input_size // cell_size
        
        # Initialize smaller masks with random binary values
        masks = np.random.binomial(1, self.p1, size=(self.num_masks, grid_size, grid_size))
        
        # Upsample all masks at once for efficiency
        upsampled_masks = np.zeros((self.num_masks, self.input_size, self.input_size))
        for i in range(self.num_masks):
            mask = masks[i].astype(np.float32)

            # Use bilinear interpolation for smoother masks
            upsampled_masks[i] = cv2.resize(mask, (self.input_size, self.input_size), 
                                           interpolation=cv2.INTER_LINEAR)
        
        self.masks = upsampled_masks
        return upsampled_masks
        
    def __call__(self, input_tensor, target_class=None):
        assert input_tensor.dim() == 4
        assert input_tensor.shape[0] == 1 
        
        # Get input dimensions
        _, C, H, W = input_tensor.shape
        device = input_tensor.device
        
        # Generate masks if not already generated
        if self.masks is None or self.masks.shape[1:] != (H, W):
            self.input_size = H 
            self.generate_masks()
        
        # Prepare the output tensor for the saliency map
        saliency_map = np.zeros((H, W))
        
        # Get original prediction if target class is not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                # Binary case
                target_class = 0
        
        # Apply masks and get weighted sum
        for i in tqdm(range(self.num_masks)):
            mask = self.masks[i]
            mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).view(1, 1, H, W)
            mask_tensor = mask_tensor.repeat(1, C, 1, 1)  # Repeat for all channels
            
            # Apply mask to input
            masked_input = input_tensor * mask_tensor
            
            # Forward pass
            with torch.no_grad():
                output = self.model(masked_input)
            
            # Get probability for the target class
            class_prob = torch.sigmoid(output)[0, 0].item()
            
            # Add contribution to saliency map
            saliency_map += mask * class_prob
        
        # Normalize the saliency map
        if np.max(saliency_map) > 0:
            saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))
        
        return saliency_map
