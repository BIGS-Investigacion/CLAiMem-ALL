import pandas as pd

def calculate_ihc(row):
    """
    Function to calculate the IHC value based on the values of ER, PR, HER2, and PAM50.
    Modify the logic as per your specific requirements.
    """
    if row['ER'] == 'negative' and row['PR'] == 'negative' and row['HER2'] == 'negative':
        return 'Triple-negative'
    elif row['ER'] == 'negative' and row['PR'] == 'negative' and row['HER2'] == 'positive':
        return 'Her2-not-luminal'
    elif row['ER'] == 'positive' and row['HER2'] == 'positive':
        return 'Luminal B(HER2+)'
    elif row['ER'] == 'positive' and row['HER2'] == 'negative' and (row['PAM50'] == 'lumb' or  row['PR']=='negative'):
        return 'Luminal B(HER2-)'
    elif row['ER'] == 'positive' and row['HER2'] == 'negative' and row['PR'] == 'positive' and row['PAM50'] == 'luma':
        return 'Luminal A'
    else:
        return 'Unknown'
    
def calculate_ihc_2(row):
    """
    Function to calculate the IHC value based on the values of ER, PR, HER2, and PAM50.
    Modify the logic as per your specific requirements.
    """
    if row['ER'] == 'negative' and row['PR'] == 'negative' and row['HER2'] == 'negative':
        return 'Triple-negative'
    elif row['ER'] == 'negative' and row['PR'] == 'negative' and row['HER2'] == 'positive':
        return 'HER2'
    elif row['ER'] == 'positive' and row['HER2'] == 'positive':
        return 'HER2'
    elif row['ER'] == 'positive' and row['HER2'] == 'negative' and (row['PAM50'] == 'lumb' or  row['PR']=='negative'):
        return 'Luminal B'
    elif row['ER'] == 'positive' and row['HER2'] == 'negative' and row['PR'] == 'positive' and row['PAM50'] == 'luma':
        return 'Luminal A'
    else:
        return 'Unknown' 

def add_ihc_column(input_csv, output_csv, function):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Ensure required columns exist
    required_columns = ['ER', 'PR', 'HER2', 'PAM50']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Add the IHC column
    df['IHC'] = df.apply(function, axis=1)

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)

# Example usage
database = "cptac"
add_ihc_column(f'data/dataset_csv/{database}_global.csv', f'data/dataset_csv/{database}_ihc_simple.csv', calculate_ihc_2)