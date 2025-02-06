from torch import nn
import torch
from timm_1_0_14 import timm as timm1014
from timm_1_0_14.timm.layers.mlp import SwiGLUPacked

class VirchowWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm1014.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=nn.SiLU)
        self.model.eval()

    def get_model(self):
        return self.model

    def forward(self, inputs):


        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            model = self.model.to("cuda")
            
            
            outputs = [model(image.to("cuda").unsqueeze(0)) for image in inputs]

            embedding = [torch.cat([output[:, 0], output[:, 5:].mean(1)], dim=-1).to(torch.float16) for output in outputs]

            embedding = torch.stack(embedding, dim=0)
        
            return embedding