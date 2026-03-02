import numpy as np
import torch
<<<<<<< HEAD
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import os

BASE_DIR = r"G:\NEW Emotion_Suppression_Project-main\Emotion_Suppression_Project-main"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Safety cleanup
X = torch.nan_to_num(X)
y = torch.nan_to_num(y)

dataset = TensorDataset(X, y)

# 80-20 split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class SuppressionLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

model = SuppressionLSTM(X.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):

    # ---- TRAIN ----
    model.train()
    train_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_X)
            loss = criterion(output, batch_y)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "suppression_model.pth"))
print("Model saved.")
=======
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from models.lstm_model import SuppressionLSTM


class SequenceDataset(Dataset):

    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return torch.tensor(self.X[idx], dtype=torch.float32), \
               torch.tensor(self.y[idx], dtype=torch.float32)


def collate_fn(batch):

    sequences, labels = zip(*batch)

    sequences = pad_sequence(sequences, batch_first=True)

    labels = torch.stack(labels)

    return sequences, labels


def train():

    X = np.load("data/features.npy", allow_pickle=True)
    y = np.load("data/labels.npy", allow_pickle=True)

    dataset = SequenceDataset(X, y)

    loader = DataLoader(dataset, batch_size=4,
                        shuffle=True, collate_fn=collate_fn)

    input_size = X[0].shape[1]

    model = SuppressionLSTM(input_size)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_fn = torch.nn.MSELoss()

    for epoch in range(30):

        total = 0

        for seq, label in loader:

            pred = model(seq).squeeze()

            loss = loss_fn(pred, label)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print("Epoch", epoch, "Loss:", total)

    torch.save(model.state_dict(), "suppression_model.pth")


if __name__ == "__main__":
    train()
>>>>>>> 0f2154a8dac9fc6f08a028a2909743dd3e0515e4
