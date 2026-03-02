import os
import numpy as np
<<<<<<< HEAD
from preprocessing.build_feature_sequence import build_sequences
from preprocessing.compute_suppression_score import compute_score

BASE_DIR = r"G:\NEW Emotion_Suppression_Project-main\Emotion_Suppression_Project-main"
CSV_DIR = os.path.join(BASE_DIR, "data", "raw_csv")
SAVE_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(SAVE_DIR, exist_ok=True)

X = []
y = []

for file in os.listdir(CSV_DIR):
    if file.endswith(".csv"):
        csv_path = os.path.join(CSV_DIR, file)

        sequences = build_sequences(csv_path)

        for seq in sequences:
            score = compute_score(seq)
            X.append(seq)
            y.append(score)

X = np.array(X)
y = np.array(y)

np.save(os.path.join(SAVE_DIR, "X.npy"), X)
np.save(os.path.join(SAVE_DIR, "y.npy"), y)

print("Dataset built.")
print("Shape:", X.shape)
=======

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
>>>>>>> 0f2154a8dac9fc6f08a028a2909743dd3e0515e4
