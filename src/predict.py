import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch.serialization
from src.model import EmotionClassifier

# Allow EmotionClassifier to be unpickled safely
torch.serialization.add_safe_globals([EmotionClassifier])

# --- CONFIG ---
model_path = "models/emotion_model.pth"
img_path = r"E:\Emotion_Suppression_Project\data\CASME II\Cropped\sub26\EP18_51\reg_img81.jpg"
excel_path = r"E:\Emotion_Suppression_Project\data\CASME II\CASME2-coding-20140508.xlsx"

EMOTION_MAP = {
    "happiness": 0,
    "disgust": 1,
    "repression": 2,
    "surprise": 3,
    "others": 4,
    "fear": 5
}
INV_EMOTION_MAP = {v: k for k, v in EMOTION_MAP.items()}

# --- LOAD MODEL ---
model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
model.eval()

# --- PREPROCESS IMAGE ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)

# --- PREDICT ---
with torch.no_grad():
    output = model(img)
    predicted_class = output.argmax(dim=1).item()
    predicted_emotion = INV_EMOTION_MAP[predicted_class]
    print(f"----Predicted: {predicted_emotion} (class {predicted_class})")

# --- GROUND TRUTH LOOKUP ---
parts = img_path.split(os.sep)
subject = int(parts[-3][3:])    # sub26 → 26
filename = parts[-2]            # EP18_51

df = pd.read_excel(excel_path)
row = df[(df['Subject'] == subject) & (df['Filename'] == filename)]

if not row.empty:
    emotion = row.iloc[0]['Estimated Emotion'].strip().lower()
    true_class = EMOTION_MAP.get(emotion)
    print(f"----Ground Truth: {emotion} (class {true_class})")
else:
    print("----Could not find ground truth label in Excel.")