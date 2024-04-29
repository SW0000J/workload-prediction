import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Batch, DataLoader
from torch_geometric.nn import GCNConv
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from Dataloader import GraphDataset


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target):
        mask = target != -1
        masked_input = input[mask]
        masked_target = target[mask]
        
        if torch.isnan(masked_input).any():
            return torch.tensor(0.0, requires_grad=True)
        elif masked_target.numel() == 0:
            return torch.tensor(0.0, device=input.device, requires_grad=True)
        else:
            return F.mse_loss(masked_input, masked_target)


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.normalize(x, p=2, dim=1)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        return x
    

def train_model(model, train_loader, optimizer, loss_fn, epochs=1):
    model.train()

    for epoch in tqdm(range(epochs), desc="Epochs"):
        total_loss = 0

        for data in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, data.y.float())

            if torch.isnan(loss):
                print("Outputs:", out)
                print("Labels:", data.y)
                print("NaN detected in loss, skipping the batch!")
                continue

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")


def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model_path, num_features, num_classes):
    model = GCN(num_features=num_features, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def evaluate_model(model, loader):
    predictions = []
    actuals = []

    with torch.no_grad():
        for data in tqdm(loader, desc="Eval"):
            out = model(data)

            valid_mask = data.y != -1
            if valid_mask.any():
                valid_pred = out[valid_mask]
                valid_actual = data.y[valid_mask]

                predictions.append(valid_pred.cpu().numpy())
                actuals.append(valid_actual.cpu().numpy())
                #print(predictions)
                #print(actuals)
                break

    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)

    mse = np.mean((predictions - actuals) ** 2)
    print(f"Mean Squared Error: {mse}")
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(predictions[:, 0], label="Predicted Mean CPU Usage Rate")
    plt.plot(actuals[:, 0], label="Actual Mean CPU Usage Rate")
    plt.title("Mean CPU Usage Rate")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(predictions[:, 1], label="Predicted Canonical Memory Usage")
    plt.plot(actuals[:, 1], label="Actual Canonical Memory Usage")
    plt.title("Canonical Memory Usage")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    # Default: Fasle
    new_train = False

    root_dir = "../datas/graphs"
    dataset = GraphDataset(root_dir=root_dir)
    train_dataset, test_dataset = dataset.get_train_test_split()

    if new_train:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        model = GCN(num_features=12, num_classes=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        loss_fn = MaskedMSELoss()

        train_model(model, train_loader, optimizer, loss_fn, epochs=1)
        save_model(model)

    model = load_model('model.pth', num_features=12, num_classes=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    evaluate_model(model, test_loader)


"""
def train_model():
    dataset = GraphDataset()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GCN(num_features=4, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = MaskedMSELoss()

    model.train()

    epoch = 1
    for _ in range(epoch):
        total_loss = 0

        for data in tqdm(train_loader, desc="Train batch"):
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, data.y.float())
            if torch.isnan(loss):
                print("NaN detected!")
                #print("Outputs:", out)
                #print("Labels:", data.y)
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            #print(f"\n{loss.item()}")

        print(f"Total loss: {total_loss/len(train_loader)}")

    torch.save(model.state_dict(), 'model.pth')
    print("Model saved successfully!")
"""