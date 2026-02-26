import os
import numpy as np

from preprocessing.extract_au_openface import extract_au
from preprocessing.build_feature_sequence import build_sequence
from preprocessing.compute_suppression_score import compute_suppression_score


CASME_VIDEO_ROOT = r"D:\8th sem\datasets\Facial Action Unit\Micro Facial Expressions CASME\CASME II\CASME II\CASME2_Compressed video\CASME2_compressed"


def find_videos(root):

    videos = []

    for sub in os.listdir(root):

        sub_path = os.path.join(root, sub)

        for file in os.listdir(sub_path):

            if file.endswith(".avi"):
                videos.append(os.path.join(sub_path, file))

    return videos


def build_dataset():

    videos = find_videos(CASME_VIDEO_ROOT)

    sequences = []
    labels = []

    for vid in videos:

        print("Processing:", vid)

        df = extract_au(vid)

        seq = build_sequence(df)

        score = compute_suppression_score(seq)

        sequences.append(seq)
        labels.append(score)

    np.save("data/features.npy", sequences)
    np.save("data/labels.npy", labels)

    print("Dataset saved.")


if __name__ == "__main__":
    build_dataset()