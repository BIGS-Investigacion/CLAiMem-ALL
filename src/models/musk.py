import torch
from musk import utils
from musk.modeling import MUSK
from timm.models import create_model
import torch.nn as nn

class MUSKWrapper(nn.Module):
    def __init__(self):
        super(MUSKWrapper, self).__init__()
        self.model = create_model("musk_large_patch16_384")
        utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", self.model, 'model|module', '')
        self.model.to(device="cuda", dtype=torch.float16)
        self.model.eval()
    
    def forward(self, img_tensor:torch.Tensor):
        with torch.inference_mode():
            image_embeddings = self.model(
                image=img_tensor.to("cuda", dtype=torch.float16),
                with_head=False,
                out_norm=False,
                ms_aug=True,
                return_global=True  
                )[0]
            return image_embeddings