import torch.nn as nn

class PhikonWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(**x)

        result = x.last_hidden_state[:, 0, :]
        print(result.shape) 
        return result