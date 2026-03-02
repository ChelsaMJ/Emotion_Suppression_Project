import subprocess
import os
<<<<<<< HEAD

OPENFACE_EXE = r"G:\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"

def extract_aus(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    command = [
        OPENFACE_EXE,
=======
import pandas as pd

OPENFACE_PATH = r"D:\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"

def extract_au(video_path, output_dir="of_output"):

    os.makedirs(output_dir, exist_ok=True)

    command = [
        OPENFACE_PATH,
>>>>>>> 0f2154a8dac9fc6f08a028a2909743dd3e0515e4
        "-f", video_path,
        "-out_dir", output_dir,
        "-aus"
    ]

<<<<<<< HEAD
    subprocess.run(command)
=======
    subprocess.run(command)

    csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]

    if len(csv_files) == 0:
        raise Exception("OpenFace failed")

    csv_path = os.path.join(output_dir, csv_files[0])

    df = pd.read_csv(csv_path)

    return df
>>>>>>> 0f2154a8dac9fc6f08a028a2909743dd3e0515e4
