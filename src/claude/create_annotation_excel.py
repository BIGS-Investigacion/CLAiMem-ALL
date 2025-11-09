#!/usr/bin/env python3
"""
Create Excel file for manual annotation of selected representative images.
"""

import json
import pandas as pd
from pathlib import Path

def load_selected_images(json_file):
    """Load selected images from JSON file with distances."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def create_annotation_excel(output_file='representative_images_annotation.xlsx'):
    """
    Create Excel file with all selected representative images for manual annotation.
    """

    # Define the structure for each dataset
    datasets = {
        'ER-Negative': 'selected_images_er_label_0_with_distances.json',
        'ER-Positive': 'selected_images_er_label_1_with_distances.json',
        'HER2-negative': 'selected_images_her2_label_0_with_distances.json',
        'HER2-positive': 'selected_images_her2_label_1_with_distances.json',
        'PR-negative': 'selected_images_pr_label_0_with_distances.json',
        'PR-positive': 'selected_images_pr_label_1_with_distances.json',
        'BASAL': 'selected_images_pam50_label_0_with_distances.json',
        'HER2-enriched': 'selected_images_pam50_label_1_with_distances.json',
        'LUMINAL-A': 'selected_images_pam50_label_2_with_distances.json',
        'LUMINAL-B': 'selected_images_pam50_label_3_with_distances.json',
        'NORMAL-like': 'selected_images_pam50_label_4_with_distances.json',
    }

    # Collect all data
    all_rows = []

    for label, json_file in datasets.items():
        if not Path(json_file).exists():
            print(f"Warning: {json_file} not found, skipping {label}")
            continue

        images = load_selected_images(json_file)

        for img_data in images:
            row = {
                'ETIQUETA': label,
                'IMAGEN': img_data['filename'],
                'DISTANCIA': round(img_data['distance'], 4),
                'ESTRUCTURA GLANDULAR': '',
                'ATIPIA NUCLEAR': '',
                'MITOSIS': '',
                'NECROSIS': '',
                'INFILTRADO_LI': ''
            }
            all_rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(all_rows)

    # Column descriptions for reference
    column_info = {
        'Conservación glandular casi total': 'Bajo',
        'Bien conservada': 'Bajo a moderado',
        'Parcialmente alterada': 'Moderado-Alto',
        'Pérdida de arquitectura': 'Alto',
        'Muy desorganizada': 'Muy alto'
    }

    # Create Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Write main data
        df.to_excel(writer, sheet_name='Anotaciones', index=False)

        # Create reference sheet
        ref_data = {
            'Característica': [
                'Conservación glandular casi total',
                'Bien conservada',
                'Parcialmente alterada',
                'Pérdida de arquitectura',
                'Muy desorganizada',
                '',
                'ESTRUCTURA GLANDULAR',
                'ATIPIA NUCLEAR',
                'MITOSIS',
                'NECROSIS',
                'INFILTRADO_LI'
            ],
            'Valor': [
                'Bajo',
                'Bajo a moderado',
                'Moderado-Alto',
                'Alto',
                'Muy alto',
                '',
                '0-4 (escala)',
                '0-4 (escala)',
                '0-4 (conteo)',
                '0-2 (ausente/presente/extenso)',
                '0-1 (ausente/presente)'
            ]
        }
        ref_df = pd.DataFrame(ref_data)
        ref_df.to_excel(writer, sheet_name='Referencia', index=False)

    print(f"\n{'='*60}")
    print(f"Excel created: {output_file}")
    print(f"{'='*60}")
    print(f"Total images: {len(all_rows)}")
    print(f"\nBreakdown by label:")
    for label in datasets.keys():
        count = df[df['ETIQUETA'] == label].shape[0]
        if count > 0:
            print(f"  {label}: {count} images")
    print(f"{'='*60}")

if __name__ == '__main__':
    create_annotation_excel()