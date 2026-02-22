import pandas as pd
import torch

def get_class_weights(label_excel_path, emotion_map):
    df = pd.read_excel(label_excel_path)
    emotion_counts = df['Estimated Emotion'].value_counts()

    total = sum(emotion_counts.values)
    weights = []
    for i in range(len(emotion_map)):
        label = list(emotion_map.keys())[list(emotion_map.values()).index(i)]
        count = emotion_counts.get(label, 1)  # avoid div by 0
        weights.append(total / count)

    norm_weights = torch.tensor(weights, dtype=torch.float)
    return norm_weights