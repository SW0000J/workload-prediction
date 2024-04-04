import torch
from torch_geometric.data import Data, Dataset
import networkx as nx
import numpy as np
import os

class GraphData(Dataset):
    def __init__(self):
        super(GraphData, self).__init__()

        