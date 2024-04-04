import networkx as nx
import numpy as np
import os
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
import torch
from tqdm import tqdm


def sacling_data(vm_lists, features):
    """
        ...
    """
    scaler = MinMaxScaler()

    temp_vm_lists = []

    for monthly_vms in vm_lists:
        temp_monthly_vms = []

        for vm in monthly_vms:
            vm[features] = scaler.fit_transform(vm[features])
            temp_monthly_vms.append(vm)

        temp_vm_lists.append(temp_monthly_vms)

    return temp_vm_lists


def preprocessing_dataset():
    """
        ...
    """
    dataset_path = "../datas/rnd/"
    trace_duration = ["2013-7", "2013-8", "2013-9"]

    vm_lists = []

    for _ in range(len(trace_duration)):
        monthly_vms = []

        for iter in range(1, 501):
            file_path = dataset_path + trace_duration[_] + '/' + f"{iter}.csv"

            vm_data = pd.read_csv(file_path)
            monthly_vms.append(vm_data)

        vm_lists.append(monthly_vms)

        # Only use 1 month data
        if _ == 0:
            break

    features = ["CPU usage [%]", "Memory usage [KB]"]
    
    vm_lists = sacling_data(vm_lists, features)

    return vm_lists


if __name__ == "__main__":
    preprocessing_dataset()