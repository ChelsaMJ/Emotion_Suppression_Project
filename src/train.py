import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import EmotionClassifier
from src.data_loader import CASME2Dataset

from torchvision import transforms
import os

def train_model(
    image_root,
    label_excel,
    num_classes=6,
    num_epochs=10,
    batch_size=32,
    lr=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Dataset & Loader
    dataset = CASME2Dataset(image_root, label_excel, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = EmotionClassifier(num_classes=num_classes).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

            loop.set_postfix(loss=loss.item())

        acc = correct / len(dataset)
        print(f"Epoch {epoch+1} done — Loss: {running_loss:.4f}, Accuracy: {acc:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(mode, "models/emotion_model.pth")
    print("Model saved to models/emotion_model.pth")
    
    