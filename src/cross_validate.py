import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torchvision import transforms

from src.data_loader import CASME2Dataset
from src.model import EmotionClassifier


def cross_validate(
        image_root,
        label_excel,
        num_classes=7,
        k=5,
        epochs=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = CASME2Dataset(image_root, label_excel, transform)

    labels = [s["label"] for s in dataset.samples]
    labels = np.array(labels)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    fold_acc = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):

        print(f"\n---- Fold {fold+1} ----")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32)

        model = EmotionClassifier(num_classes=num_classes).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(epochs):

            model.train()

            for images, labels in train_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # ---- Validation ----
        model.eval()
        correct = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = outputs.argmax(1)

                correct += (preds == labels).sum().item()

        acc = correct / len(val_subset)
        fold_acc.append(acc)

        print("Fold Accuracy:", acc)

    print("\nMean Accuracy:", np.mean(fold_acc))