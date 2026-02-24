import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np

from src.model import EmotionClassifier
from src.data_loader import CASME2Dataset


def compute_class_weights(dataset, num_classes):
    labels = [sample["label"] for sample in dataset.samples]
    class_counts = np.bincount(labels, minlength=num_classes)

    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float)


def train_model(
    image_root,
    label_excel,
    num_classes=7,
    num_epochs=40,
    batch_size=32,
    lr=1e-4,
    val_split=0.2
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----------------------------
    # Data Augmentation
    # ----------------------------
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # ----------------------------
    # Dataset
    # ----------------------------
    full_dataset = CASME2Dataset(image_root, label_excel, transform=train_transform)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Validation uses different transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # ----------------------------
    # Model
    # ----------------------------
    model = EmotionClassifier(num_classes=num_classes).to(device)

    # ----------------------------
    # Class Weights
    # ----------------------------
    class_weights = compute_class_weights(full_dataset, num_classes).to(device)
    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    # ----------------------------
    # Training Loop
    # ----------------------------
    for epoch in range(num_epochs):

        # ===== TRAIN =====
        model.train()
        train_correct = 0
        train_total = 0
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

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
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            loop.set_postfix(loss=loss.item())

        train_acc = train_correct / train_total

        # ===== VALIDATION =====
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"\nTrain Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model, "models/emotion_model.pth")
            print("Model saved.")

    print("\nBest Validation Accuracy:", best_val_acc)