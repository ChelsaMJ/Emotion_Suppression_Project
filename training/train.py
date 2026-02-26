import torch
from torch.utils.data import DataLoader
from models.lstm_model import SuppressionLSTM
from dataset.dataset_loader import SuppressionDataset
import numpy as np

def train_model(sequences, labels):

    dataset = SuppressionDataset(sequences, labels)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    input_size = sequences[0].shape[1]

    model = SuppressionLSTM(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCELoss()

    for epoch in range(20):

        total_loss = 0

        for seq, label in loader:

            pred = model(seq).squeeze()

            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "suppression_model.pth")

    return model