import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict
import json

"""
Script para extraer los top-K features con mayor atención de archivos WSI (.pt)
y agregarlos por etiqueta, generando un único archivo con 1 instancia por etiqueta.

Uso:
    python extract_top_attention_features.py --input_dir features/ --output aggregated_features.pt --top_k 100 --labels labels.csv
"""

def load_wsi_features(file_path):
    """
    Carga features de WSI desde archivo .pt

    Returns:
        features: tensor de features [N_patches, feature_dim]
        attention: tensor de scores de atención [N_patches] (si existe)
    """
    data = torch.load(file_path)

    # Manejar diferentes formatos de archivos
    if isinstance(data, dict):
        features = data.get('features', data.get('feat', data.get('embeddings', None)))
        attention = data.get('attention', data.get('att', data.get('scores', None)))
    elif isinstance(data, torch.Tensor):
        features = data
        attention = None
    else:
        raise ValueError(f"Formato de archivo no reconocido: {type(data)}")

    if features is None:
        raise ValueError("No se encontraron features en el archivo")

    return features, attention


def compute_self_attention_scores(features, device='cpu'):
    """
    Calcula scores de self-attention para cada patch usando similarity scoring.

    Para cada patch, calcula su similitud con todos los demás patches y usa
    la suma de similitudes como score de importancia (patches más "representativos"
    del WSI tendrán mayor score).

    Args:
        features: tensor [N, D] con features
        device: dispositivo donde realizar las operaciones

    Returns:
        attention_scores: tensor [N] con scores de atención para cada patch
    """
    features = features.to(device)

    # Normalizar features para calcular similitud coseno
    features_norm = F.normalize(features, p=2, dim=1)  # [N, D]

    # Calcular matriz de similitud: similarity[i,j] = cos(feature_i, feature_j)
    similarity_matrix = torch.mm(features_norm, features_norm.t())  # [N, N]

    # Aplicar softmax por filas para obtener pesos de atención
    attention_weights = F.softmax(similarity_matrix, dim=1)  # [N, N]

    # Score de cada patch = suma de sus similitudes con todos los demás
    # Patches más "centrales" o representativos tendrán mayor score
    attention_scores = attention_weights.sum(dim=1)  # [N]

    return attention_scores.cpu()


def get_top_k_features(features, attention=None, top_k=100, aggregation='attention', device='cpu'):
    """
    Extrae los top-K features según diferentes criterios

    Args:
        features: tensor [N, D] con features
        attention: tensor [N] con scores de atención (opcional)
        top_k: número de features a extraer
        aggregation: método de selección ('attention', 'self_attention', 'norm', 'random', 'variance')
        device: dispositivo donde realizar las operaciones

    Returns:
        top_features: tensor [top_k, D] con los top-k features seleccionados
        indices: índices seleccionados
    """
    n_patches = features.shape[0]

    # Mover a GPU si está disponible
    features = features.to(device)
    if attention is not None:
        attention = attention.to(device)

    # Si top_k es mayor que el número de patches, tomar todos
    if top_k >= n_patches:
        return features.cpu(), torch.arange(n_patches)

    if aggregation == 'attention':
        if attention is not None:
            # Ordenar por scores de atención pre-calculados (de mayor a menor)
            scores = attention
            top_indices = torch.argsort(scores, descending=True)[:top_k]
        else:
            # Fallback: si no hay attention scores, calcular self-attention
            print(f"  ⚠ Advertencia: método 'attention' seleccionado pero no hay scores de atención. Usando 'self_attention' en su lugar.")
            scores = compute_self_attention_scores(features, device=device).to(device)
            top_indices = torch.argsort(scores, descending=True)[:top_k]

    elif aggregation == 'self_attention':
        # Calcular self-attention entre patches del WSI
        scores = compute_self_attention_scores(features, device=device).to(device)
        top_indices = torch.argsort(scores, descending=True)[:top_k]

    elif aggregation == 'norm':
        # Ordenar por norma L2 de los features
        norms = torch.norm(features, dim=1)
        top_indices = torch.argsort(norms, descending=True)[:top_k]

    elif aggregation == 'variance':
        # Seleccionar features con mayor varianza (más informativos)
        variances = torch.var(features, dim=1)
        top_indices = torch.argsort(variances, descending=True)[:top_k]

    elif aggregation == 'random':
        # Muestreo aleatorio
        top_indices = torch.randperm(n_patches, device=device)[:top_k]

    else:
        raise ValueError(f"Método de agregación no reconocido: {aggregation}")

    top_features = features[top_indices]

    # Mover de vuelta a CPU para almacenamiento
    return top_features.cpu(), top_indices.cpu()


def discover_labels_from_directories(input_dir, file_extension='.pt'):
    """
    Descubre automáticamente etiquetas buscando subcarpetas en input_dir.
    Estructura esperada: input_dir/LABEL/*.pt

    Args:
        input_dir: Path al directorio principal
        file_extension: extensión de archivos a buscar

    Returns:
        dict: {label: [list of filenames]}
    """
    input_path = Path(input_dir)
    label_groups = {}

    # Buscar todas las subcarpetas
    for subdir in input_path.iterdir():
        if subdir.is_dir():
            label = subdir.name
            # Buscar archivos con la extensión especificada
            files = list(subdir.glob(f'*{file_extension}'))

            if files:
                # Guardar rutas relativas desde input_dir
                relative_paths = [f.relative_to(input_path) for f in files]
                label_groups[label] = [str(p) for p in relative_paths]

    return label_groups


def aggregate_features_by_label(feature_list, method='mean', device='cpu'):
    """
    Agrega múltiples conjuntos de features en uno solo

    Args:
        feature_list: lista de tensores [K, D]
        method: método de agregación ('mean', 'max', 'concat', 'sum')
        device: dispositivo donde realizar las operaciones

    Returns:
        aggregated: tensor agregado
    """
    # Mover todos a GPU para operaciones rápidas
    feature_list_gpu = [f.to(device) for f in feature_list]

    if method == 'mean':
        # Promedio de todos los features
        stacked = torch.stack(feature_list_gpu)  # [N_wsi, K, D]
        aggregated = torch.mean(stacked, dim=0)  # [K, D]

    elif method == 'max':
        # Máximo elemento a elemento
        stacked = torch.stack(feature_list_gpu)
        aggregated = torch.max(stacked, dim=0)[0]

    elif method == 'sum':
        # Suma
        stacked = torch.stack(feature_list_gpu)
        aggregated = torch.sum(stacked, dim=0)

    elif method == 'concat':
        # Concatenar todos
        aggregated = torch.cat(feature_list_gpu, dim=0)  # [N_wsi*K, D]

    else:
        raise ValueError(f"Método de agregación no reconocido: {method}")

    # Mover de vuelta a CPU
    return aggregated.cpu()


def main():
    parser = argparse.ArgumentParser(
        description='Extrae top-K features con mayor atención de WSI y los agrega por etiqueta'
    )
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='Directorio con archivos .pt de features')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Directorio de salida donde se guardarán los archivos .pt por etiqueta')
    parser.add_argument('--labels', '-l', type=str, required=False,
                        help='Archivo CSV con columnas: filename,label (opcional si se usa --auto_discover)')
    parser.add_argument('--top_k', '-k', type=int, default=100,
                        help='Número de features top a extraer por WSI (default: 100)')
    parser.add_argument('--selection_method', '-s', type=str,
                        default='self_attention',
                        choices=['attention', 'self_attention', 'norm', 'variance', 'random'],
                        help='Método para seleccionar top features (default: self_attention)')
    parser.add_argument('--aggregation_method', '-a', type=str,
                        default='concat',
                        choices=['mean', 'max', 'sum', 'concat'],
                        help='Método para agregar features de múltiples WSI (default: concat)')
    parser.add_argument('--file_extension', '-e', type=str, default='.pt',
                        help='Extensión de archivos de features (default: .pt)')
    parser.add_argument('--save_metadata', action='store_true',
                        help='Guardar metadata con información de agregación')
    parser.add_argument('--device', '-dev', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Dispositivo a usar (default: auto - detecta GPU automáticamente)')
    parser.add_argument('--batch_process', '-b', action='store_true',
                        help='Procesar archivos en batch en GPU (más rápido pero usa más memoria)')
    parser.add_argument('--auto_discover', '-auto', action='store_true',
                        help='Descubrir automáticamente subcarpetas como etiquetas (input_dir/LABEL/*.pt)')

    args = parser.parse_args()

    # Validar argumentos
    if not args.labels and not args.auto_discover:
        parser.error("Debe especificar --labels o --auto_discover")

    # Configurar dispositivo
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\n{'='*80}")
    print(f"CONFIGURACIÓN DE EJECUCIÓN")
    print(f"{'='*80}")
    print(f"Dispositivo: {device}")
    if device.type == 'cuda':
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Batch processing: {'Activado' if args.batch_process else 'Desactivado'}")
    print(f"{'='*80}\n")

    # Obtener etiquetas y archivos
    if args.auto_discover:
        print(f"Descubriendo etiquetas automáticamente desde subcarpetas de: {args.input_dir}")
        label_groups = discover_labels_from_directories(args.input_dir, args.file_extension)

        if not label_groups:
            raise ValueError(f"No se encontraron subcarpetas con archivos {args.file_extension} en {args.input_dir}")

        print(f"Modo: Auto-descubrimiento de subcarpetas")
    else:
        print(f"Leyendo etiquetas desde: {args.labels}")
        df_labels = pd.read_csv(args.labels)

        # Verificar columnas requeridas y usar slide_id si está disponible
        if 'slide_id' in df_labels.columns and 'label' in df_labels.columns:
            # Formato CLAM: case_id, slide_id, label
            label_groups = df_labels.groupby('label')['slide_id'].apply(list).to_dict()
        elif 'filename' in df_labels.columns and 'label' in df_labels.columns:
            # Formato simple: filename, label
            label_groups = df_labels.groupby('label')['filename'].apply(list).to_dict()
        else:
            raise ValueError("El archivo de labels debe tener columnas: ('slide_id' y 'label') o ('filename' y 'label')")

        # Agrupar por etiqueta (ya hecho arriba)
        print(f"Modo: Archivo CSV de etiquetas")

    print(f"\nEncontradas {len(label_groups)} etiquetas únicas:")
    for label, files in label_groups.items():
        print(f"  - {label}: {len(files)} archivos")

    # Procesar cada etiqueta
    input_dir = Path(args.input_dir)
    aggregated_data = {}
    metadata = {
        'top_k': args.top_k,
        'selection_method': args.selection_method,
        'aggregation_method': args.aggregation_method,
        'labels': {}
    }

    for label, filenames in label_groups.items():
        print(f"\n{'='*80}")
        print(f"Procesando etiqueta: {label} ({len(filenames)} archivos)")
        print('='*80)

        label_features = []
        processed_files = []
        total_files = len(filenames)

        for idx, filename in enumerate(filenames, 1):
            # Construir path del archivo
            # Si viene de auto_discover, filename ya incluye la subcarpeta (ej: "BASAL/file.pt")
            # Si viene de CSV, puede o no incluir extensión
            if not filename.endswith(args.file_extension):
                filename = filename + args.file_extension

            file_path = input_dir / filename

            # Mostrar progreso
            remaining = total_files - idx
            progress_bar = f"[{idx}/{total_files}] Restantes: {remaining}"

            if not file_path.exists():
                print(f"  {progress_bar} ⚠ Archivo no encontrado: {file_path}")
                continue

            try:
                # Cargar features
                features, attention = load_wsi_features(file_path)

                # Extraer top-k features
                top_features, indices = get_top_k_features(
                    features,
                    attention,
                    top_k=args.top_k,
                    aggregation=args.selection_method,
                    device=device
                )

                label_features.append(top_features)
                processed_files.append(filename)

                print(f"  {progress_bar} ✓ {filename}: {features.shape[0]} patches → top-{len(top_features)}")

            except Exception as e:
                print(f"  {progress_bar} ✗ Error procesando {filename}: {e}")
                continue

        if not label_features:
            print(f"  ⚠ No se procesaron archivos para la etiqueta {label}")
            continue

        # Agregar todos los features de esta etiqueta
        aggregated = aggregate_features_by_label(label_features, method=args.aggregation_method, device=device)
        aggregated_data[label] = aggregated

        # Liberar memoria GPU si está siendo usada
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Guardar metadata
        metadata['labels'][label] = {
            'n_files': len(processed_files),
            'files': processed_files,
            'feature_shape': list(aggregated.shape)
        }

        print(f"\n  → Agregado final para '{label}': {aggregated.shape}")

    # Guardar resultados
    print(f"\n{'='*80}")
    print("GUARDANDO RESULTADOS")
    print('='*80)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guardar un archivo .pt por etiqueta
    csv_data = []
    for label, features in aggregated_data.items():
        # Crear nombre de archivo limpio (reemplazar caracteres problemáticos)
        safe_label = label.replace('/', '_').replace(' ', '_').replace('-', '_')
        output_file = output_dir / f"{safe_label}.pt"

        # Guardar features como tensor (formato simple)
        torch.save(features, output_file)
        print(f"✓ Guardado {label}: {output_file} - Shape: {features.shape}")

        # Añadir entrada al CSV
        csv_data.append({'filename': f"{safe_label}.pt", 'label': label})

        # Guardar metadata si se solicita
        if args.save_metadata:
            metadata_file = output_dir / f"{safe_label}_metadata.json"
            label_metadata = {
                'label': label,
                'n_files': metadata['labels'][label]['n_files'],
                'files': metadata['labels'][label]['files'],
                'feature_shape': list(features.shape),
                'top_k': args.top_k,
                'selection_method': args.selection_method,
                'aggregation_method': args.aggregation_method
            }
            with open(metadata_file, 'w') as f:
                json.dump(label_metadata, f, indent=2)
            print(f"  └─ Metadata: {metadata_file}")

    # Guardar CSV con mapeo manteniendo formato original (case_id, slide_id, label)
    csv_path = output_dir / 'labels.csv'
    csv_output_data = []
    for item in csv_data:
        # Remover extensión .pt del filename para obtener slide_id
        slide_id = item['filename'].replace('.pt', '')
        csv_output_data.append({
            'case_id': slide_id,  # Usar el mismo valor para case_id y slide_id
            'slide_id': slide_id,
            'label': item['label']
        })

    df_csv = pd.DataFrame(csv_output_data)
    df_csv.to_csv(csv_path, index=False)
    print(f"\n✓ CSV de etiquetas guardado: {csv_path}")
    print(f"  Contiene {len(csv_output_data)} archivos")
    print(f"  Formato: case_id, slide_id, label")

    # Resumen final
    print(f"\n{'='*80}")
    print("RESUMEN")
    print('='*80)
    print(f"Directorio de salida: {output_dir}")
    print(f"Etiquetas procesadas: {len(aggregated_data)}")
    for label, features in aggregated_data.items():
        n_files = metadata['labels'][label]['n_files']
        safe_label = label.replace('/', '_').replace(' ', '_').replace('-', '_')
        print(f"  - {safe_label}.pt: {features.shape} (de {n_files} archivos WSI)")

    print(f"\n✓ Proceso completado exitosamente!")


if __name__ == '__main__':
    main()
