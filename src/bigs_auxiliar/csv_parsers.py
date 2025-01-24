import csv
import os

# Parse the input CSV file with the new format
def parse_cptac_subtypes(input_csv_path, images_dir, output_csv_path):
    # Read the input CSV file
    with open(input_csv_path, mode='r') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    # Prepare the output data
    output_data = []

    for row in rows:
        case_id = row['Sample.ID'].strip().upper()
        label = row['PAM50'].strip().lower()
        
        # Find corresponding images
        for image_file in os.listdir(images_dir):
            if case_id[1:] in image_file:
                slide_id = os.path.splitext(image_file)[0]
                output_data.append({
                    'case_id': case_id,
                    'slide_id': slide_id,
                    'label': label
                })

    # Write the output CSV file
    with open(output_csv_path, mode='w', newline='') as outfile:
        fieldnames = ['case_id', 'slide_id', 'label']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(output_data)

# Parse the input CSV file from TCGA database
def parse_tcga_subtypes(input_csv_path, images_dir, output_csv_path):
    # Read the input CSV file
    with open(input_csv_path, mode='r') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    # Prepare the output data
    output_data = []

    for row in rows:
        case_id = row['CLID'].strip().upper()
        #tumor = row['Tumor or Normal'].strip().lower() == 'tumor'
        #tumor if only_tumor_classification else row['PAM50 and Claudin-low (CLOW) Molecular Subtype'].strip().lower()
        label = row['PAM50 and Claudin-low (CLOW) Molecular Subtype'].strip().lower() 
        
        # Find corresponding images
        for image_file in os.listdir(images_dir):
            if case_id in image_file:
                slide_id = os.path.splitext(image_file)[0]
                output_data.append({
                    'case_id': case_id,
                    'slide_id': slide_id,
                    'label': label
                })

    # Write the output CSV file
    with open(output_csv_path, mode='w', newline='') as outfile:
        fieldnames = ['case_id', 'slide_id', 'label']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(output_data)


if __name__ == '__main__':
    # Path to the input CSV file
    input_csv_path = 'dataset_csv/outsider/mmc2.csv'
    # Path to the directory containing the images
    images_dir = '/media/jorge/Expansion/medicina/patologia_digital/datos/histology/clasificacion_cancer/perfil_molecular/publicas/TCGA-BRCA/flash_frozen'
    # Path to the output CSV file
    output_csv_path = 'dataset_csv/tcga-subtype.csv'
    parse_tcga_subtypes(input_csv_path, images_dir, output_csv_path)
    print('Done!')

    # Path to the input CSV file
    input_csv_path = 'dataset_csv/outsider/prosp-brca-v5.4-public-sample-annotation.csv'
    # Path to the directory containing the images
    images_dir = '/media/jorge/Expansion/medicina/patologia_digital/datos/histology/clasificacion_cancer/perfil_molecular/publicas/CPTAC-BRCA/BRCA'
    # Path to the output CSV file
    output_csv_path = 'dataset_csv/brca-subtype.csv'
    parse_cptac_subtypes(input_csv_path, images_dir, output_csv_path)
    print('Done!')