import h5py
import numpy as np
import os
import csv

#TODO: Fix issue which removes everything but coords.

# Función para cargar datasets de un archivo HDF5
def load_h5_data(file_path):
    with h5py.File(file_path, "r") as f:
        data = {}
        for key in f.keys():
            data[key] = f[key][:]
    return data

# Rutas a los archivos
original_path = "/media/jorge/SP_PHD_U3/perfil_molecular/features_20x_tumor/cptac/features_/h5_files"
reduced_path = "/media/jorge/SP_PHD_U3/perfil_molecular/.patches_20x_tumor/cptac/patches"
# Obtener lista de archivos en original_path
original_files = [os.path.join(original_path, file) for file in os.listdir(original_path) if file.endswith(".h5")]

for file in original_files:
    # Cargar datos
    original_data = load_h5_data(os.path.join(original_path, file))
    reduced_data = load_h5_data(os.path.join(reduced_path, file))

    # Convertir coordenadas a tuplas para comparación
    original_coords = [tuple(coord) for coord in original_data["coords"]]
    reduced_coords_set = set(tuple(coord) for coord in reduced_data["coords"])

    # Encontrar índices en el original que coinciden con el reducido
    matching_indices = [i for i, coord in enumerate(original_coords) if coord in reduced_coords_set]

    # Filtrar datos
    original_data["coords"] = original_data["coords"][matching_indices]
    original_data["features"] = original_data["features"][matching_indices]
    
    # Guardar nuevo archivo HDF5 filtrado
    output_path = os.path.join(original_path, file)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("filtered_dataset", data=original_data)
