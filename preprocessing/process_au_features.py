import numpy as np

AU_COLUMNS = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r',
    'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r',
    'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r',
    'AU26_r', 'AU45_r'
]


def build_au_sequence(df):

    available_cols = [c for c in AU_COLUMNS if c in df.columns]

    sequence = df[available_cols].values

    return sequence