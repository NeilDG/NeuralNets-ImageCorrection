"""
Created on Wed Jun 19 17:12:04 2019
@author: Utku Ozbulak - github.com/utkuozbulak
"""
from visualizers import vanilla_backprop
import numpy as np
from PIL import Image

# from guided_backprop import GuidedBackprop  # To use with guided backprop
# from integrated_gradients import IntegratedGradients  # To use with integrated grads

def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def save_gradient_images(gradient):
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    return gradient
    
def visualize_activation(model, target_layer, input_image, target_class, penalty_function = 1.0):
    # Vanilla backprop
    VBP = vanilla_backprop.VanillaBackprop(model, target_layer)
    # Generate gradients
    vanilla_grads = VBP.generate_gradients(input_image, target_class, penalty_function)
    print("Vanilla grad shape: ", np.shape(vanilla_grads))
    
    grad_times_image = vanilla_grads[0] * input_image.cpu().numpy()[0]
    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(grad_times_image)
    # Save grayscale gradients
    
    return grayscale_vanilla_grads