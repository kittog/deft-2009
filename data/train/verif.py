import pandas as pd

def remove_nan_rows(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    
    # Supprimer les lignes contenant "nan" dans la colonne 'Parties'
    df = df.dropna(subset=['Parties'])
    
    # Enregistrer le DataFrame modifié dans un nouveau fichier CSV
    df.to_csv(output_csv, index=False)

# Utilisation
input_csv = 'extracted_data_test_it.csv'  # Remplacez par le fichier CSV réel
output_csv = 'extracted_data_test_it_no_nan.csv'  # Nom du nouveau fichier CSV
remove_nan_rows(input_csv, output_csv)
