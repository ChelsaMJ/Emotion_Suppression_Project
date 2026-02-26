import subprocess
import os
import pandas as pd

OPENFACE_PATH = r"C:\OpenFace\FeatureExtraction.exe"


def extract_au(video_path, output_folder="openface_output"):

    os.makedirs(output_folder, exist_ok=True)

    command = [
        OPENFACE_PATH,
        "-f", video_path,
        "-out_dir", output_folder,
        "-aus"
    ]

    subprocess.run(command)

    # Find generated CSV
    csv_files = [f for f in os.listdir(output_folder) if f.endswith(".csv")]

    if not csv_files:
        raise Exception("No OpenFace CSV generated")

    csv_path = os.path.join(output_folder, csv_files[0])

    df = pd.read_csv(csv_path)

    return df