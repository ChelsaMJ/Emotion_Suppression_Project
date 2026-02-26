import subprocess
import os
import pandas as pd

OPENFACE_PATH = r"D:\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"

def extract_au(video_path, output_dir="of_output"):

    os.makedirs(output_dir, exist_ok=True)

    command = [
        OPENFACE_PATH,
        "-f", video_path,
        "-out_dir", output_dir,
        "-aus"
    ]

    subprocess.run(command)

    csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]

    if len(csv_files) == 0:
        raise Exception("OpenFace failed")

    csv_path = os.path.join(output_dir, csv_files[0])

    df = pd.read_csv(csv_path)

    return df