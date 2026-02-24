import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import EmotionClassifier
from src.data_loader import CASME2Dataset
from torchvision import transforms


def train_model():

    # -------- DEVICE --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------- PATHS (CORRECTED) --------
    image_root = "data/CASME II/CASME2_RAW/CASME2-RAW"
    label_excel = "data/CASME II/CASME2-coding-20140508.xlsx"

    # -------- TRANSFORMS --------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # -------- DATASET --------
    dataset = CASME2Dataset(image_root, label_excel, transform=transform)
    print("Dataset size:", len(dataset))

    if len(dataset) == 0:
        print("❌ Dataset is empty. Check folder structure.")
        return

    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # -------- MODEL --------
    model = EmotionClassifier(num_classes=7).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # -------- TRAIN LOOP --------
    for epoch in range(10):

        model.train()
        running_loss = 0.0
        correct = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/10")

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

            loop.set_postfix(loss=loss.item())

        accuracy = correct / len(dataset)
        print(f"Epoch {epoch+1} — Loss: {running_loss:.4f}, Accuracy: {accuracy:.4f}")

    # -------- SAVE MODEL --------
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/emotion_model.pth")
    print("✅ Model saved to models/emotion_model.pth")


if __name__ == "__main__":
    train_model()