import networkx as nx
import numpy as np
import os
import pickle
import random
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx


class GraphDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None, train_val_split=(0.8, 0.2)):
        super(GraphDataset, self).__init__(transform, pre_transform)

        self.graph_path = "../datas/graphs"

        graph_extension = ".gpickle"
        self.graph_files = []

        for filename in os.listdir(self.graph_path):
            if filename.endswith(graph_extension):
                self.graph_files.append(filename)

        random.shuffle(self.graph_files)

        total_graphs = len(self.graph_files)
        self.train_size = int(total_graphs * train_val_split[0])
        self.val_size = total_graphs - self.train_size
        self.train_files = self.graph_files[:self.train_size]
        self.val_files = self.graph_files[self.train_size:]

        print(f"\nTotal graphs: {total_graphs} \nTrain graphs: {self.train_size} \nValidation graphs: {self.val_size}")


    def get(self, idx):
        filename = self.graph_files[idx]
        filepath = os.path.join(self.graph_path, filename)
        
        with open(filepath, "rb") as f:
            graph = pickle.load(f)

        data = from_networkx(graph)
        return data
    

    def __getitem__(self, idx):
        filename = self.graph_files[idx]
        filepath = os.path.join(self.graph_path, filename)
        
        with open(filepath, "rb") as f:
            graph = pickle.load(f)

        data = from_networkx(graph)
        return data
    
        """
        ignore_value = -1

        for node in graph.nodes(data=True):
            required_keys = ["capacity_cpu", "capacity_memory", "request_cpu", "request_memory", "mean_cpu_usage_rate", "canonical_memory_usage"]
            for key in required_keys:
                if key not in node[1]:
                    node[1][key] = ignore_value

        edge_required_keys = ["mean_cpu_usage_rate", "canonical_memory_usage"]
        for u, v, edge_attrs in graph.edges(data=True):
            for key in edge_required_keys:
                if key not in edge_attrs:
                    edge_attrs[key] = ignore_value

        #print(filepath)

        for node_id, node_data in graph.nodes(data=True):
            node_data['x'] = torch.tensor([node_data.get("capacity_cpu", 0),
                                           node_data.get("capacity_memory", 0),
                                           node_data.get("request_cpu", 0),
                                           node_data.get("request_memory", 0)], 
                                           dtype=torch.float)

            node_data['y'] = torch.tensor([node_data.get("mean_cpu_usage_rate", 0),
                                           node_data.get("canonical_memory_usage", 0)], 
                                           dtype=torch.long)

        data = from_networkx(graph)
        return data
        """
    

    def __len__(self):
        return len(self.graph_files)
    

if __name__ == "__main__":
    dataset = GraphDataset()
    train_size, val_size = dataset.get_index()
    data = dataset.get(0)
    print(data)
    