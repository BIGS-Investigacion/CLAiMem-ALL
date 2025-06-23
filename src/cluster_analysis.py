import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

from models.builder import get_encoder

# ---------- CONFIGURACIÓN ----------
DATASET_PATH = "/media/jorge/investigacion/datos/pam50"  # Reemplaza por la ruta a tus carpetas
BATCH_SIZE = 32
USE_UMAP = False  # Cambia a True si tienes instalado UMAP
RANDOM_SEED = 42
K_FEATURES = 50
class_names = ['Basal', 'HER2-enriched', 'Luminal A', 'Luminal B', 'Normal-like']   
#class_names = ['negative', 'positive']  # Reemplaza por tus nombres de clases
# -----------------------------------
# Modelo ResNet50 sin la capa final
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, transform = get_encoder("virchow")

#models.resnet50(pretrained=True)
#model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval().to(device)

# Transformaciones
'''transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])'''

# Dataset y dataloader
dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
labels = np.array([label for _, label in dataset.samples])

num_classes = len(class_names)

# Extraer características
# Extracción de características
features = []
with torch.no_grad():
    for imgs, _ in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        out = out.view(out.size(0), -1)
        features.append(out.cpu())
features = torch.cat(features, dim=0).numpy()

# Selección de atributos
selector = SelectKBest(score_func=f_classif, k=K_FEATURES)
features = selector.fit_transform(features, labels)

# Reducción de dimensionalidad
if USE_UMAP:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED)
    reduced = reducer.fit_transform(features)
else:
    reducer = TSNE(n_components=2, random_state=RANDOM_SEED)
    reduced = reducer.fit_transform(features)

# Visualización por etiqueta
fig, axes = plt.subplots(1, num_classes, figsize=(6 * num_classes, 5), squeeze=False)
import seaborn as sns
# Paleta de colores "magma" con una entrada por clase
palette = sns.color_palette("magma", num_classes)
for i in range(num_classes):
    ax = axes[0, i]
    idx = labels == i
    ax.scatter(reduced[idx, 0], reduced[idx, 1], color=palette[i], alpha=0.6)
    ax.set_title(f"Label {i}: " + class_names[i])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(-100, 100)
    ax.set_ylim(-80, 80)

plt.suptitle("")
plt.tight_layout()
plt.show()


# Figura única con todos los puntos
plt.figure(figsize=(8, 6))
plt.title("")
plt.scatter(reduced[:, 0], reduced[:, 1], c=[label for _, label in dataset.samples], cmap="magma")

plt.tight_layout()
plt.show()

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

TOP_N = 5  # Número de imágenes más cercanas por clase
output_dir = "/home/jorge/Escritorio/laura"  # Cambia esta ruta según tu necesidad
        
class_centroids = {}
closest_images = {}

# 1. Calcular centroide de cada clase
for i in range(num_classes):
    class_points = reduced[labels == i]
    centroid = class_points.mean(axis=0)
    class_centroids[i] = centroid

# 2. Calcular distancia de cada punto al centroide de su clase
for class_idx in range(num_classes):
    idxs_in_class = np.where(labels == class_idx)[0]
    class_reduced = reduced[idxs_in_class]
    centroid = class_centroids[class_idx]

    # Distancia euclidiana
    dists = np.linalg.norm(class_reduced - centroid, axis=1)
    top_n = idxs_in_class[np.argsort(dists)[:TOP_N]]  # obtener índices originales
    closest_images[class_idx] = top_n

# 3. Mostrar las imágenes más cercanas a cada centroide
for class_idx, indices in closest_images.items():
    fig, axs = plt.subplots(1, TOP_N, figsize=(15, 3))
    fig.suptitle(f"{class_names[class_idx]} – {TOP_N} imágenes más cercanas al centroide", fontsize=14)
    for ax, idx in zip(axs, indices):
        img_path, _ = dataset.samples[idx]
        img = Image.open(img_path)
        # Guardar la imagen en un nuevo directorio organizado por clase
        class_dir = os.path.join(output_dir, class_names[class_idx])
        os.makedirs(class_dir, exist_ok=True)
        img.save(os.path.join(class_dir, os.path.basename(img_path)))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(os.path.basename(img_path), fontsize=8)
    plt.tight_layout()
    plt.show()