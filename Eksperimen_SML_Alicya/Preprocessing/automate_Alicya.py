import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def download_data():
    import gdown
    url = "https://drive.google.com/uc?id=1u4eL5GYTfv5AWZ0PUWNP2N9EEpn6n-NS"
    output = "Forest_Cover_Type_Dataset.csv"
    if not os.path.exists(output):
        print("Mengunduh dataset...")
        gdown.download(url, output, quiet=False)
    return output

def load_data(file_path):
    print(f"\nMembaca dataset dari: {file_path}")
    df = pd.read_csv(file_path)
    print("\nInformasi Umum Dataset:")
    print(df.info())
    print("\nStatistik Deskriptif:")
    pd.set_option('display.max_columns', None)
    print(df.describe())
    print("\nDistribusi Target (Cover_Type):")
    print(df['Cover_Type'].value_counts())
    return df

def preprocess_data(df):
    # Hapus duplikat dan missing values
    df = df.drop_duplicates().dropna()

    # Standarisasi kolom numerik
    columns_to_standardize = [
        'Elevation', 'Aspect', 'Slope',
        'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways', 'Hillshade_9am',
        'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
    ]

    scaler = StandardScaler()
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

    return df

def save_preprocessed_data(df, output_dir='preprocessing/namadataset_preprocessing/'):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "forest_cover_clean.csv")
    df.to_csv(output_file, index=False)
    print(f"\nDataset hasil preprocessing disimpan di: {output_file}")
    return output_file

if __name__ == "__main__":
    csv_file = download_data()
    raw_df = load_data(csv_file)
    processed_df = preprocess_data(raw_df)
    save_preprocessed_data(processed_df)
