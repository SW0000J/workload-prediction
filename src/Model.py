import torch
import torch_geometric
from torch_geometric.data import Batch, DataLoader
from torch_geometric.nn import GCNConv
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

from Dataloader import GraphDataset


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    

if __name__ == "__main__":
    dataset = GraphDataset()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GCN(num_features=4, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    model.train()
    total_loss = 0
    for data in train_loader:
        print(data)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Total loss: {total_loss}")