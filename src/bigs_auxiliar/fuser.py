import pandas as pd
import os
import glob

def merge_csv_files(csv_files: list[str], output_file:str)->None:
    
    if not csv_files:
        print("No se encontraron archivos CSV en la carpeta especificada.")
        return

    # Inicializar un DataFrame vacío
    merged_df = pd.DataFrame()

    for file in csv_files:
        # Leer cada archivo CSV
        df = pd.read_csv(file)
        
        # Asegurarse de que las dos primeras columnas existan
        if df.shape[1] < 2:
            print(f"El archivo {file} no tiene al menos dos columnas. Se omitirá.")
            continue
        # Renombrar la tercera columna con el nombre del archivo actual
        if df.shape[1] > 2:
            df.rename(columns={df.columns[2]: file.split("/")[-1]}, inplace=True)
        # Usar las dos primeras columnas como clave primaria
        df.set_index([df.columns[0], df.columns[1]], inplace=True)
        
        # Fusionar con el DataFrame acumulado
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = merged_df.combine_first(df)

    # Restablecer el índice y guardar el resultado en un archivo CSV
    merged_df.reset_index(inplace=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Archivos fusionados guardados en {output_file}")

database = "cptac"
# Configuración
inputs = [f"data/dataset_csv/{database}-er.csv", f"data/dataset_csv/{database}-erbb2.csv", f"data/dataset_csv/{database}-pr.csv", f"data/dataset_csv/{database}-subtype_pam50.csv"]  # Cambia esta ruta a tu carpeta de entrada
output_file = f"data/dataset_csv/{database}_global.csv"  # Cambia esta ruta al archivo de salida

merge_csv_files(inputs, output_file)