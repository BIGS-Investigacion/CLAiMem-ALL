import argparse
import os
import h5py
import openslide
import torch
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision.models import vgg16
from huggingface_hub import hf_hub_download
import torch

class ModifiedVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        original = vgg16(pretrained=False)
        self.features = original.features  # convolucionales sin cambios
        self.avgpool = original.avgpool
        
        # Este bloque reemplaza el classifier original
        self.classifier = nn.Sequential(
            nn.Linear(25088, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 2)  # salida binaria
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


from torchvision import transforms

# === Transformación basada en el JSON del modelo ===
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7238, 0.5716, 0.6779],
                         std=[0.112, 0.1459, 0.1089])
])


# === CARGAR MODELO ===

# Nombre del modelo en Hugging Face
model_name = "kaczmarj/breast-tumor-vgg16mod.tcga-brca"
#model_name = "kaczmarj/breast-tumor-resnet34.tcga-brca"


# Descarga el archivo del modelo
model_path = hf_hub_download(repo_id="kaczmarj/breast-tumor-vgg16mod.tcga-brca", filename="pytorch_model.bin")


model = ModifiedVGG16()
model.load_state_dict(torch.load(model_path))

# Enviar modelo a dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()

parser = argparse.ArgumentParser(description="Filter tumor patches from WSI.")
parser.add_argument('--database', type=str, required=True, help="Database name (e.g., 'cptac').")
parser.add_argument('--patch_size', type=int, required=True, help="Patch size (e.g., 256).")
parser.add_argument('--wsi_extension', type=str, required=True, help="WSI file extension (e.g., '.svs').")
parser.add_argument('--wsi_dir', type=str, required=True, help="Directory containing WSI files.")
parser.add_argument('--patches_dir', type=str, required=True, help="Directory containing patch files.")

args = parser.parse_args()

DATABASE = args.database
PATCH_SIZE = args.patch_size
WSI_EXTENSION = args.wsi_extension
WSI_DIR = args.wsi_dir
PATCHES_DIR = args.patches_dir

#RECORREMOS LOS PATCHES DE CADA WSI
for wsi in os.listdir(WSI_DIR):
    if wsi.endswith(WSI_EXTENSION):
        wsi_path = os.path.join(WSI_DIR, wsi)
        
        print(f"Procesando '{wsi}'...")

        h5_input_path = os.path.join(PATCHES_DIR, wsi.replace(WSI_EXTENSION, '.h5'))
        # === CARGAR COORDENADAS ===
        with h5py.File(h5_input_path, 'r') as f:
            coords = f['coords'][:]

            # === ABRIR WSI ===
            slide = openslide.OpenSlide(wsi_path)

            # === LISTA DE COORDENADAS FILTRADAS ===
            filtered_coords = []

            # === CLASIFICAR PARCHES ===
            for (x, y) in tqdm(coords, desc="Filtrando parches tumorales"):
                patch = slide.read_region((int(x), int(y)), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")

                input_tensor = transform(patch).unsqueeze(0).to(device)  # [1, 3, 224, 224]
                with torch.no_grad():
                    outputs = model(input_tensor)
                    pred = torch.argmax(outputs, dim=1).item()
                
                    if pred == 1:
                        filtered_coords.append([x, y])

        print(str(len(filtered_coords)) + " out of " + str(len(coords)) + " patches are tumor")

        
        # === GUARDAR NUEVO ARCHIVO H5 ===
        with h5py.File(h5_input_path, 'w') as f:
            f.create_dataset('coords', data=np.array(filtered_coords, dtype=int))