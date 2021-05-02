import torch
from torch import nn
from torch.utils.data import Dataset
from inputs import load_data
from torch import from_numpy


class TrainingDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.data = load_data()[:2]  # train_x, train_y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        X = self.data[0].iloc[idx].to_numpy()
        X = from_numpy(X).float()
        y = self.data[1].iloc[idx]
        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)
        return X, y


class TestDataset(Dataset):
    def __init__(self, transform=None):
        self.data = load_data()[2]  # test_x
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data.iloc[idx].to_numpy()
        X = from_numpy(X).float()
        if self.transform:
            X = self.transform(X)
        return X


def train_model(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 2 == 0:
            loss, current = loss.item(), batch * len(X) if len(X) == 10 else size - len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validate_model(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def predict_test(dataloader, model):
    preds = []
    model.eval()
    with torch.no_grad():
        for X in dataloader:
            pred = model(X)
            pred_category = pred  # .argmax(1)
            preds += pred_category
    return preds


class Torch(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x


if __name__ == '__main__':
    pass
