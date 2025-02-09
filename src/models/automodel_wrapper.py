import torch
import torch.nn as nn

class AutoModelWrapper(nn.Module):
    def __init__(self, auto_model):
        super().__init__()
        
        self.model = auto_model
        self.model.eval()

    def forward(self, inputs):
        # Get the features
        with torch.inference_mode():
            tensor_list = inputs['pixel_values']
            features = [self.model(**{'pixel_values':input}).last_hidden_state[:, 0, :] for input in tensor_list]
            features = torch.stack(features, dim=0)
            return features