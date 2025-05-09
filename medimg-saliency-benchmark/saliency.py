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
    ResNet101Binary,
    InceptionNetBinary
)
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
            print("activations hooked: ", output.shape)
            # [B, C, H, W]
            self.activations = output.clone()

        # Register hooks on the last conv layer
        if isinstance(model, AlexNetBinary):
            target_layer = self.model.features[10]
            self.weights = self.model.classifier.weight
        elif isinstance(model, VGG16Binary):
            target_layer = self.model.features[28] 
            self.weights = self.model.classifier.weight
        elif isinstance(model, ResNet101Binary):
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
            print("gradients hooked: ", grad_output[0].shape)
            # [B, C, H, W]
            self.gradients = grad_output[0].detach().clone()    

        # Hook to capture activations
        # The hook will be called every time after forward() 
        # has computed an output. 
        # It should have the following signature:
        # hook(module, input, output) -> None or modified output
        def forward_hook(module, input, output):
            print("activations hooked: ", output.shape)
            # [B, C, H, W]
            self.activations = output.clone()

        # Register hooks on the last conv layer
        if isinstance(model, AlexNetBinary) or isinstance(model, VGG16Binary):
            target_layer = self.model.features[-1]  # Last convolution layer
        elif isinstance(model, ResNet101Binary):
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
    Apply random binary mask on input and compute the predicted class probability as a weight for the mask.
    Aggregate multiple masks to obtain a saliency map.
    """
    def __init__(self, model, num_masks=100, scale_factor = 20):
        self.model = model
        self.model.eval()
        self.num_masks = num_masks
        self.scale_factor = scale_factor

    def __call__(self, input_tensor):

        assert input_tensor.dim() == 4
        assert input_tensor.shape[0] == 1
        assert input_tensor.shape[-2] == input_tensor.shape[-1]
        assert input_tensor.dtype == torch.float32
        _, _, H, W = input_tensor.shape
        
        # Prepare the output tensor for the saliency map
        map = np.zeros((H, W))
        
        # Generate random binary masks and aggregate
        for _ in range(self.num_masks):
            
            # Create a random binary mask
            mask = np.random.binomial(
                n=1,
                p=0.5,
                size=(H // self.scale_factor, W // self.scale_factor)
            ) 
            
            # Cast to input type
            mask = mask.astype(np.float32)

            # Upsample to get smooth mask
            mask = cv2.resize(
                src = mask,
                dsize=(W, H)
            )
            
            # Apply the mask to the input image
            masked_input = input_tensor * mask

            # Forward pass through the model
            output = self.model(masked_input)
            
            # Get the prob to use as weight
            class_prob = torch.nn.functional.sigmoid(output)
            
            # Convert to scalar
            class_prob = class_prob[0, 0].item()

            # Add the masked input contribution to the 
            # saliency map, weighted by the class score
            map += mask * class_prob

        # Ensure all values are non-negative
        map = np.maximum(map, 0)  

        # Normalize the saliency map to [0, 1]
        map = (map - map.min()) / (map.max() - map.min() + 1e-8)
        
        return mask