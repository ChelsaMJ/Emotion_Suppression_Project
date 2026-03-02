<<<<<<< HEAD
import pandas as pd
import numpy as np

WINDOW_SIZE = 20

def build_sequences(csv_path):
    df = pd.read_csv(csv_path)

    au_cols = [col for col in df.columns if "_r" in col]
    au_data = df[au_cols]

    # safe normalisation
    au_data = au_data.fillna(0)

    min_vals = au_data.min()
    max_vals = au_data.max()

    denominator = max_vals - min_vals
    denominator[denominator == 0] = 1  # prevent divide-by-zero

    au_data = (au_data - min_vals) / denominator

    au_data = au_data.replace([np.inf, -np.inf], 0)
    au_data = au_data.fillna(0)

    sequences = []

    for i in range(len(au_data) - WINDOW_SIZE):
        window = au_data.iloc[i:i+WINDOW_SIZE].values
        sequences.append(window)

    return np.array(sequences)
=======
import numpy as np

AU_COLUMNS = [
    'AU01_r','AU02_r','AU04_r','AU05_r','AU06_r',
    'AU07_r','AU09_r','AU10_r','AU12_r','AU14_r',
    'AU15_r','AU17_r','AU20_r','AU23_r','AU25_r',
    'AU26_r','AU45_r'
]

def build_sequence(df):

    available = [c for c in AU_COLUMNS if c in df.columns]

    seq = df[available].values

    # normalize 0–5 → 0–1
    seq = seq / 5.0

    return seq
>>>>>>> 0f2154a8dac9fc6f08a028a2909743dd3e0515e4
