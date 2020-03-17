import torch

class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.target_layer = target_layer
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
            #print("Received gradient from ", module)
        #Register hook to the target layer
        for module_pos, module in self.model._modules.items():
            if module_pos == "fc_block":
                break
            
            if module_pos == self.target_layer:
                print("Registered hook to ", module_pos)
                module.register_backward_hook(hook_function)
                break

    def generate_gradients(self, input_image, target_class, penalty_function = 1.0):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        target = target_class * penalty_function
        # Backward pass
        model_output.backward(gradient=target)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.cpu().numpy()[0]
        return gradients_as_arr

