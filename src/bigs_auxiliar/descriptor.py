import csv
import h5py
import os
import torch
import pandas as pd


def generate_file_label_dict(csv_path):
    file_label_dict = {}
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            file_label_dict[row['file']] = row['label']
    return file_label_dict


def procesar_h5_features(directorio_h5, wsi_labels, output_csv, clave_features='features'):
    """
    directorio_h5: carpeta con archivos .h5
    wsi_labels: diccionario con etiquetas {'wsi_id': label}
    output_csv: ruta donde guardar el archivo CSV final
    clave_features: nombre del campo que contiene los vectores (por defecto 'features')
    """
    filas = []

    for archivo in os.listdir(directorio_h5):
        if not archivo.endswith(".h5"):
            continue

        ruta = os.path.join(directorio_h5, archivo)
        wsi_id = os.path.splitext(archivo)[0]
        label = wsi_labels.get(wsi_id)

        if label is None:
            print(f"Etiqueta no encontrada para {wsi_id}, se omite.")
            continue

        with h5py.File(ruta, 'r') as f:
            if clave_features not in f:
                print(f"No se encontró '{clave_features}' en {archivo}, se omite.")
                continue

            features = f[clave_features][:]
            for vec in features:
                fila = list(vec) + [label, wsi_id]
                filas.append(fila)

    # Nombres de columnas
    dim = len(filas[0]) - 2 if filas else 0
    columnas = [f'feature_{i}' for i in range(dim)] + ['label', 'wsi_id']

    df = pd.DataFrame(filas, columns=columnas)
    df.to_csv(output_csv, index=False)
    print(f"CSV guardado en {output_csv} con {len(df)} filas.")
