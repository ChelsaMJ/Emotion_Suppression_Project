import os
from preprocessing.extract_au_openface import extract_aus

BASE_DIR = r"G:\NEW Emotion_Suppression_Project-main\Emotion_Suppression_Project-main"
VIDEO_ROOT = r"G:\capstone data\CASME II\CASME2_Compressed video\CASME2_compressed"
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "raw_csv")

for root, dirs, files in os.walk(VIDEO_ROOT):
    for file in files:
        if file.endswith(".avi"):
            video_path = os.path.join(root, file)
            print("Processing:", video_path)
            extract_aus(video_path, OUTPUT_DIR)

print("All videos processed.")