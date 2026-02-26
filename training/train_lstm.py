import numpy as np
import torch
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