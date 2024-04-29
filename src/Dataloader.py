import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx


class GraphDataset(Dataset):
    def __init__(self, root_dir=None, file_list=None, transform=None, pre_transform=None, train_val_split=(0.8, 0.2)):
        super(GraphDataset, self).__init__(transform, pre_transform)

        if root_dir and not file_list:
            self.graph_path = root_dir

            graph_extension = ".gpickle"
            self.graph_files = []

            for filename in os.listdir(self.graph_path):
                if filename.endswith(graph_extension):
                    self.graph_files.append(filename)

            # Dataset filtering
            self.filtering_data_by_max(max_value=10000)

            random.shuffle(self.graph_files)

            total_graphs = len(self.graph_files)
            self.train_size = int(total_graphs * train_val_split[0])
            self.val_size = total_graphs - self.train_size
            self.train_files = self.graph_files[:self.train_size]
            self.val_files = self.graph_files[self.train_size:]

            print(f"\nTotal graphs: {total_graphs} \nTrain graphs: {self.train_size} \nValidation graphs: {self.val_size}")
        elif file_list:
            self.graph_path = "../datas/graphs"
            self.graph_files = file_list
        else:
            raise ValueError("Invalid constructor parameters")


    def filtering_data_by_max(self, max_value):
        csv_file = os.path.join(self.graph_path, "graph_stats.csv")

        df = pd.read_csv(csv_file)
        filtered_job_ids = df[df["n_node"] >= max_value]["job_id"]
        filtered_file_names = [f"job_{job_id}.gpickle" for job_id in filtered_job_ids]
        self.graph_files = [file for file in self.graph_files if file not in filtered_file_names]

    
    def get_train_test_split(self):
        train_dataset = GraphDataset(root_dir=self.graph_path, file_list=self.train_files)
        validation_dataset = GraphDataset(root_dir=self.graph_path, file_list=self.val_files)

        return train_dataset, validation_dataset


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

        for node_id, node_data in graph.nodes(data=True):
            x_values = [
                node_data["timestamp"],
                node_data["machine_id"],
                node_data["event_type"],
                node_data["capacity_cpu"],
                node_data["capacity_memory"],
                node_data["task_id"],
                node_data["start_time"],
                node_data["end_time"],
                #node_data["user_name"],
                node_data["scheduling_class"],
                node_data["priority"],
                node_data["request_cpu"],
                node_data["request_memory"],
                #node_data["attribute_name"]
            ]
            y_values = [
                node_data["mean_cpu_usage_rate"],
                node_data["canonical_memory_usage"]
            ]

            node_data['x'] = torch.tensor(x_values, dtype=torch.float)
            node_data['y'] = torch.tensor(y_values, dtype=torch.float)

        data = from_networkx(graph)
        return data
    

    def __len__(self):
        return len(self.graph_files)
    

if __name__ == "__main__":
    dataset = GraphDataset()
    train_size, val_size = dataset.get_index()
    data = dataset.get(0)
    print(data)
    