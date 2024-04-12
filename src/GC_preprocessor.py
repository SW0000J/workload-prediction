import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
import torch
from tqdm import tqdm


TOTAL_ITER = 500


def save_graph(job_graph, job_id):
    output_dir = "../datas/graphs"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"job_{job_id}.gpickle")
    with open(output_path, "wb") as f:
        pickle.dump(job_graph, f, pickle.HIGHEST_PROTOCOL)
    #print(f"Graph for job {job_id} saved to {output_path}")

    # Not available to use .write_gpickle() in v.3
    #nx.write_gpickle(job_graph, output_path)


def load_graph(job_id):
    dataset_dir = "../datas/graphs"

    dataset_path = os.path.join(dataset_dir, f"job_{job_id}.gpickle")
    #print(dataset_path)

    with open(dataset_path, "rb") as f:
        graph = pickle.load(f)

    return graph


def get_all_job_ids(dataset_path, csv_range, job_cols_to_use):
    unique_job_ids = set()

    for iter in tqdm(range(csv_range), desc="Collecting unique job IDs"):
        job_df = pd.read_csv(f"{dataset_path}/job_events/part-{iter:05d}-of-00500.csv", usecols=job_cols_to_use)
        unique_job_ids.update(job_df["job_id"].unique())

    return list(unique_job_ids)


def read_tasks_for_job_id(dataset_path, job_id, csv_range, task_cols_to_use, task_usage_cols_to_use, chunksize=10000):
    tasks = pd.DataFrame()
    task_usages = pd.DataFrame()

    for iter in range(csv_range):
        task_df_iter = pd.read_csv(f"{dataset_path}/task_events/part-{iter:05d}-of-00500.csv", usecols=task_cols_to_use)
        filtered_tasks = task_df_iter[task_df_iter["job_id"] == job_id]
        tasks = pd.concat([tasks, filtered_tasks])

        task_usage_df_iter = pd.read_csv(f"{dataset_path}/task_usage/part-{iter:05d}-of-00500.csv", usecols=task_usage_cols_to_use)
        filtered_usages = task_usage_df_iter[(task_usage_df_iter["job_id"] == job_id) & (task_usage_df_iter["task_id"].isin(filtered_tasks["task_id"]))]
        task_usages = pd.concat([task_usages, filtered_usages])

    return tasks, task_usages


def save_preprocessed_graph_data(dataset_path, csv_range, task_cols_to_use, task_usage_cols_to_use):
    preprocessed_dir = "../datas/preprocessed"
    os.makedirs(preprocessed_dir, exist_ok=True)

    for iter in tqdm(range(csv_range), desc="Processing CSV files"):
        task_file_path = f"{dataset_path}/task_events/part-{iter:05d}-of-00500.csv"
        task_usage_file_path = f"{dataset_path}/task_usage/part-{iter:05d}-of-00500.csv"
        
        task_df = pd.read_csv(task_file_path, usecols=task_cols_to_use)
        task_usage_df = pd.read_csv(task_usage_file_path, usecols=task_usage_cols_to_use)
        
        merged_data = pd.merge(task_df, task_usage_df, on=["job_id", "task_id", "machine_id"], how="inner")
        
        for job_id, group in merged_data.groupby("job_id"):
            output_file = os.path.join(preprocessed_dir, f"job_{job_id}.csv")
            if os.path.exists(output_file):
                group.to_csv(output_file, mode='a', header=False, index=False)
            else:
                group.to_csv(output_file, index=False)


def save_graph_status(job_id, n_nodes, n_edges):
    output_file = "../datas/graphs/graph_stats.csv"

    stats_df = pd.DataFrame({"job_id": [job_id], "n_node": [n_nodes], "n_edge": [n_edges]})

    if not os.path.exists(output_file):
        stats_df.to_csv(output_file, index=False)
    else:
        stats_df.to_csv(output_file, mode='a', header=False, index=False)


def make_graph(job_id, output_path, machine_cpu_info, machine_memory_info):
    G = nx.Graph()

    try:
        job_df = pd.read_csv(f"{output_path}/preprocessed/job_{job_id}.csv")
    except FileNotFoundError:
        # print(f"Error: The file for job_id {job_id} does not exist in the specified directory.")
        return G
    
    if not G.has_node(job_id):
        G.add_node(job_id, node_type="job")

    for _, row in job_df.iterrows():
        machine_id = row["machine_id"]
        task_id = row["task_id"]
        
        if not G.has_node(machine_id):
            G.add_node(machine_id, node_type="machine",
                       capacity_cpu=machine_cpu_info[machine_id], 
                       capacity_memory=machine_memory_info[machine_id])

        if not G.has_node(task_id):
            G.add_node(task_id, node_type="task",
                       request_cpu=row["request_cpu"],
                       request_memory=row["request_memory"])
            
        G.add_edge(job_id, machine_id)
        
        G.add_edge(task_id, machine_id, 
                   mean_cpu_usage_rate=row["mean_cpu_usage_rate"],
                   canonical_memory_usage=row["canonical_memory_usage"])
    
    if G.number_of_nodes() < 10 or not nx.is_connected(G):
        G = nx.Graph()
    else:
        save_graph(G, job_id)
        save_graph_status(job_id, G.number_of_nodes(), G.number_of_edges())

    return G


def filtering_node_data(df, low_percent, high_percent):
    low_quantile = df["n_node"].quantile(low_percent)
    high_quantile = df["n_node"].quantile(high_percent)
    
    filtered_df = df[(df["n_node"] > low_quantile) & (df["n_node"] < high_quantile)]
    
    return filtered_df


def show_node_dist():
    data_path = "../datas/graphs/graph_stats.csv"
    
    df = pd.read_csv(data_path)

    filtered_df = filtering_node_data(df, 0.2, 0.6)
    
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_df["n_node"], bins=20, color="skyblue", edgecolor="black")
    plt.title("Distribution of Node Counts")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def preprocessing():
    output_path = "../datas"
    dataset_path = "../datas/clusterdata-2011-2"
    table_names = ["job_events", "machine_attributes", "machine_events", "task_constraints", "task_events", "task_usage"]

    # Add machine node
    print("Read csv!")
    machine_col = ["timestamp", "machine_id", "event_type", "platform_id", "capacity_cpu", "capacity_memory"]
    job_col = ["timestamp", "missing_info", "job_id", "event_type", "user_name", "scheduling_class", "job_name", "logical_job_name"]
    task_col = ["timestamp", "missing_info", "job_id", "task_id", "machine_id", "event_type", "user_name", "scheduling_class",
                 "priority", "request_cpu", "request_memory", "request_disk", "machine_constraint"]
    task_usage_col = ["start_time", "end_time", "job_id", "task_id", "machine_id", "mean_cpu_usage_rate", 
                      "canonical_memory_usage", "assigned_memory_usage", "unmapped_page_cache_memory_usage", 
                      "total_page_cache_memory_usage", "maximum_memory_usage", "mean_disk_io_time", 
                      "mean_local_disk_space_used", "maximum_cpu_usage", "maximum_disk_io_time", "cpi",
                      "mai", "sample_portion", "aggregation_type", "sampled_cpu_usage"]
    
    machine_cols_to_use = ["machine_id", "capacity_cpu", "capacity_memory"]
    job_cols_to_use = ["job_id", "event_type", "user_name"]
    task_cols_to_use = ["job_id", "task_id", "machine_id", "request_cpu", "request_memory"]
    task_usage_cols_to_use = ["job_id", "task_id", "machine_id", "mean_cpu_usage_rate", "canonical_memory_usage"]

    # Read machines dataframe
    machines_df = pd.read_csv(f"{dataset_path}/machine_events/part-00000-of-00001.csv", usecols=machine_cols_to_use)
    machines_info = machines_df.set_index("machine_id").T.to_dict("index")

    csv_range = TOTAL_ITER

    ## If you want to run preprocess_graph_data(), fix parameters
    # save_preprocessed_graph_data(dataset_path, csv_range, task_cols_to_use, task_usage_cols_to_use)

    job_ids = get_all_job_ids(dataset_path, csv_range, job_cols_to_use)

    for job_id in tqdm(job_ids, desc="Processing jobs"):
        graph = make_graph(job_id, output_path, machines_info["capacity_cpu"], machines_info["capacity_memory"])

    return True


    """
    for job_id in tqdm(job_ids, desc="Processing jobs"):
        tasks, task_usages = read_tasks_for_job_id(dataset_path, job_id, csv_range, task_cols_to_use, task_usage_cols_to_use)
        print("read_tasks_for_job_id() loeaded!")
        
        G = nx.Graph()
        G.add_node(job_id, node_type="job")

        for _, task_row in tasks.iterrows():
            task_id = task_row["task_id"]
            G.add_node(task_id, node_type="task", request_cpu=task_row["request_cpu"], request_memory=task_row["request_memory"])
            G.add_edge(job_id, task_id)
            
            machine_id = task_row["machine_id"]

            if pd.notnull(machine_id):
                if not G.has_node(machine_id):
                    G.add_node(machine_id, node_type="machine", capacity_cpu=machines_info["capacity_cpu"][machine_id], 
                               capacity_memory=machines_info["capacity_memory"][machine_id])

                for _, usage_row in task_usages[(task_usages["task_id"] == task_id) & (task_usages["machine_id"] == machine_id)].iterrows():
                    if machine_id == usage_row["machine_id"]:
                        G.add_edge(task_id, machine_id,
                               mean_cpu_usage_rate=usage_row["mean_cpu_usage_rate"],
                               canonical_memory_usage=usage_row["canonical_memory_usage"])
                        
        job_graphs[job_id] = G
        print(G)"""


if __name__ == "__main__":
    #graphs = preprocessing()
    #save_graph(graphs)
    #load_graph(6253708944)
    show_node_dist()