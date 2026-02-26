import numpy as np
from extract_au_openface import extract_au
from process_au_features import build_au_sequence


def extract_full_features(video_path):

    df = extract_au(video_path)

    au_seq = build_au_sequence(df)

    # Normalize
    au_seq = au_seq / 5.0

    return au_seq