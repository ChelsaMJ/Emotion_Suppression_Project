import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from glob import glob
from src.model import EmotionClassifier
import torch.serialization

# Allow EmotionClassifier to be unpickled safely
torch.serialization.add_safe_globals([EmotionClassifier])

# --- CONFIG ---
model_path = "models/emotion_model.pth"
excel_path = "data/CASME II/CASME2-coding-20140508.xlsx"
test_dir = "test_images"

EMOTION_MAP = {
    "happiness": 0,
    "disgust": 1,
    "repression": 2,
    "surprise": 3,
    "others": 4,
    "fear": 5,
    "sadness": 6,
}
INV_EMOTION_MAP = {v: k for k, v in EMOTION_MAP.items()}

# --- Load model ---
model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
model.eval()

# --- Preprocess ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- Load Excel ---
df = pd.read_excel(excel_path)

# --- Predict all images in test_images/ ---
test_images = glob(os.path.join(test_dir, "*.jpg"))
print(f"----Found {len(test_images)} test image(s)\n")

for img_path in test_images:
    # Predict
    img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        pred_class = output.argmax(dim=1).item()
        pred_emotion = INV_EMOTION_MAP[pred_class]

    print(f"----{os.path.basename(img_path)} → Predicted: {pred_emotion} (class {pred_class})")

    # Ground truth extraction
    parts = img_path.split(os.sep)
    try:
        sub_index = next(i for i, part in enumerate(parts) if part.startswith("sub"))
        subject = int(parts[sub_index][3:])
        filename = parts[sub_index + 1]
    except StopIteration:
        print("----Could not parse subject and filename from path")
        continue

    row = df[(df["Subject"] == subject) & (df["Filename"] == filename)]
    if not row.empty:
        gt_emotion = row.iloc[0]["Estimated Emotion"].strip().lower()
        gt_class = EMOTION_MAP.get(gt_emotion, "N/A")
        print(f"----Ground Truth: {gt_emotion} (class {gt_class})")
    else:
        print("----Ground Truth: Not found in Excel")

    print("-" * 50)