import pandas as pd
import argparse
from pathlib import Path

"""
Script de ejemplo para crear un archivo CSV de etiquetas a partir de nombres de archivos.

Supone que los nombres de archivo contienen información sobre la etiqueta, por ejemplo:
- TCGA_BRCA_BASAL_001.pt → BASAL
- patient_123_LUMINAL_A.pt → LUMINAL_A
- WSI_ER_positive_456.pt → ER_positive
"""

def extract_label_from_filename(filename, pattern='underscore', label_position=-2):
    """
    Extrae la etiqueta del nombre del archivo según un patrón.

    Args:
        filename: nombre del archivo
        pattern: 'underscore' (separado por _), 'dash' (separado por -), 'custom'
        label_position: posición de la etiqueta cuando se separa por el delimitador
                        -1 = última, -2 = penúltima, etc.
    """
    # Remover extensión
    name = Path(filename).stem

    if pattern == 'underscore':
        parts = name.split('_')
    elif pattern == 'dash':
        parts = name.split('-')
    else:
        raise ValueError(f"Patrón no reconocido: {pattern}")

    # Extraer etiqueta
    try:
        label = parts[label_position]
    except IndexError:
        label = 'UNKNOWN'

    return label


def create_labels_from_directory(input_dir, output_csv, pattern='underscore',
                                 label_position=-2, extension='.pt'):
    """
    Crea un archivo CSV de etiquetas escaneando un directorio.
    """
    input_path = Path(input_dir)

    # Listar todos los archivos con la extensión especificada
    files = list(input_path.glob(f'*{extension}'))

    if not files:
        print(f"⚠ No se encontraron archivos con extensión {extension} en {input_dir}")
        return

    # Crear DataFrame
    data = []
    for file_path in files:
        filename = file_path.name
        label = extract_label_from_filename(filename, pattern, label_position)
        data.append({'filename': filename, 'label': label})

    df = pd.DataFrame(data)

    # Guardar
    df.to_csv(output_csv, index=False)
    print(f"✓ Archivo de etiquetas creado: {output_csv}")
    print(f"  Total de archivos: {len(df)}")
    print(f"\nDistribución de etiquetas:")
    print(df['label'].value_counts())


def create_labels_from_excel(excel_file, sheet_name, filename_col, label_col, output_csv):
    """
    Crea un archivo CSV de etiquetas a partir de un Excel existente.
    """
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    # Seleccionar columnas relevantes
    df_labels = df[[filename_col, label_col]].copy()
    df_labels.columns = ['filename', 'label']

    # Guardar
    df_labels.to_csv(output_csv, index=False)
    print(f"✓ Archivo de etiquetas creado: {output_csv}")
    print(f"  Total de archivos: {len(df_labels)}")
    print(f"\nDistribución de etiquetas:")
    print(df_labels['label'].value_counts())


def main():
    parser = argparse.ArgumentParser(description='Crear archivo CSV de etiquetas para WSI')
    parser.add_argument('--mode', '-m', type=str, required=True,
                        choices=['directory', 'excel'],
                        help='Modo: escanear directorio o leer desde Excel')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Directorio de entrada (modo directory) o archivo Excel (modo excel)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Archivo CSV de salida')
    parser.add_argument('--extension', '-e', type=str, default='.pt',
                        help='Extensión de archivos (modo directory, default: .pt)')
    parser.add_argument('--pattern', '-p', type=str, default='underscore',
                        choices=['underscore', 'dash'],
                        help='Patrón de separación en nombres de archivo (default: underscore)')
    parser.add_argument('--label_position', '-pos', type=int, default=-2,
                        help='Posición de la etiqueta en el nombre (default: -2 penúltimo)')
    parser.add_argument('--sheet', '-s', type=str, default='Sheet1',
                        help='Nombre de la hoja Excel (modo excel)')
    parser.add_argument('--filename_col', '-fc', type=str, default='filename',
                        help='Nombre de la columna con nombres de archivo (modo excel)')
    parser.add_argument('--label_col', '-lc', type=str, default='label',
                        help='Nombre de la columna con etiquetas (modo excel)')

    args = parser.parse_args()

    if args.mode == 'directory':
        create_labels_from_directory(
            args.input,
            args.output,
            pattern=args.pattern,
            label_position=args.label_position,
            extension=args.extension
        )
    elif args.mode == 'excel':
        create_labels_from_excel(
            args.input,
            args.sheet,
            args.filename_col,
            args.label_col,
            args.output
        )


if __name__ == '__main__':
    main()
