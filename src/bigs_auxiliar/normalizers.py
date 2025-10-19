import numpy as np
import torch
import torchstain


class MacenkoNormalizer:
    """Wrapper para normalización Macenko"""
    def __init__(self):
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        self.fitted = False
    
    def fit(self, image):
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            image_tensor = image
        self.normalizer.fit(image_tensor)
        self.fitted = True
    
    def transform(self, image):
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        normalized_tensor, _, _ = self.normalizer.normalize(image_tensor)
        
        if normalized_tensor.shape[0] == 3:
            normalized_array = normalized_tensor.permute(1, 2, 0).numpy()
        elif normalized_tensor.shape[2] == 3:
            normalized_array = normalized_tensor.numpy()
        else:
            raise ValueError(f"Unexpected shape: {normalized_tensor.shape}")
        
        return normalized_array.astype(np.uint8)
