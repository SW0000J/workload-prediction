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


def summarize_task_usage():
    csv_path = "../datas/cluserdata_preprocessed/task_usage"
    save_path = "../datas/cluserdata_preprocessed/task_usage_filtered"
    for csv_name in tqdm(os.listdir(csv_path), desc="Summarize usage"):
        df = pd.read_csv(os.path.join(csv_path, csv_name))
        df.sort_values(["task_id", "machine_id", "start_time", "end_time"], inplace=True)
        df["group"] = ((df["end_time"].shift() != df["start_time"]) | (df["task_id"].shift() != df["task_id"]) | (df["machine_id"].shift() != df["machine_id"])).cumsum()

        result = df.groupby("group").agg(
            start_time=("start_time", "first"),
            end_time=("end_time", "last"),
            task_id=("task_id", "first"),
            machine_id=("machine_id", "first"),
            mean_cpu_usage_rate=("mean_cpu_usage_rate", "mean"),
            canonical_memory_usage=("canonical_memory_usage", "mean")
        ).reset_index(drop=True)

        result.sort_values(["start_time", "end_time", "task_id", "machine_id"], inplace=True)
        result.to_csv(os.path.join(save_path, csv_name), index=False)


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


def add_nodes_and_edges(G, df, node_type, node_name_prefix, key_attrs, additional_attrs=[], job_to=None):
    is_task = "task" in node_type
    task_dict = {}

    previous_node = None
    for _, row in df.iterrows():
        if is_task:
            node_id = f"{node_name_prefix}_{row[key_attrs[0]]}_{row[key_attrs[1]]}_{row[additional_attrs[1]]}"
        else:
            node_id = f"{node_name_prefix}_{row[key_attrs[0]]}_{row[key_attrs[1]]}"

        node_attrs = {}

        for attr in key_attrs + additional_attrs:
            if attr in row:
                value = row[attr]
                if pd.notna(value):
                    node_attrs[attr] = value
                else:
                    continue 

        G.add_node(node_id, type=node_type, **node_attrs)

        if not is_task and previous_node:
            G.add_edge(previous_node, node_id)
        previous_node = node_id

        if is_task:
            task_dict.setdefault(row["task_id"], []).append(node_id)

    if is_task and job_to:
        for nodes in task_dict.values():
            if nodes:
                G.add_edge(job_to, nodes[0])

                for i in range(len(nodes) - 1):
                    G.add_edge(nodes[i], nodes[i+1])

    return task_dict


def add_machine_nodes_and_edges(G, df, node_prefix, machine_id):
    first_node = None
    previous_node = None

    for _, row in df.iterrows():
        node_id = f"{node_prefix}_{machine_id}_{row['timestamp']}_{row['event_type']}"
        node_attrs = {k: v for k, v in row.items() if pd.notna(v)}
        G.add_node(node_id, type="machine_event", **node_attrs)

        if previous_node:
            G.add_edge(previous_node, node_id)
        else:
            first_node = node_id
        previous_node = node_id
    
    return first_node


def connect_tasks_to_machines(G, task_dict, machine_events_path, machine_events_list):
    for task_id, nodes_list in task_dict.items():
        first_task_node = nodes_list[0]
        machine_id = int(G.nodes[first_task_node]["machine_id"])
        
        machine_file_name = f"{machine_id}.csv"
        if machine_file_name in machine_events_list:
            machine_df = pd.read_csv(os.path.join(machine_events_path, machine_file_name)).dropna(
                subset=["timestamp", "event_type", "capacity_cpu", "capacity_memory"]
            )
            machine_df.sort_values(["timestamp", "event_type"], inplace=True)

            first_machine_node = add_machine_nodes_and_edges(G, machine_df, "machine_event", machine_id)

            if first_machine_node:
                G.add_edge(first_machine_node, first_task_node)


def make_graph(output_path):
    # ["job_events", "machine_attributes", "machine_events", "task_constraints", "task_events", "task_usage"]
    preprocessed_path = os.path.join(output_path, "cluserdata_preprocessed")
    job_path = os.path.join(preprocessed_path, "job_events")
    task_events_path = os.path.join(preprocessed_path, "task_events")
    task_usage_path = os.path.join(preprocessed_path, "task_usage_filtered")
    machine_events_path = os.path.join(preprocessed_path, "machine_events")

    machine_events_list = os.listdir(machine_events_path)

    for csv_name in tqdm(os.listdir(job_path), desc="Job_list"):
        if not csv_name.endswith(".csv"):
            continue
        if csv_name not in os.listdir(task_events_path):
            continue
        if csv_name not in os.listdir(task_usage_path):
            continue

        G = nx.Graph()

        datasets = {
            "job_events": ["timestamp", "event_type", "user_name", "scheduling_class"],
            "task_events": ["timestamp", "task_id", "machine_id", "event_type", "user_name", "scheduling_class", "priority", "request_cpu", "request_memory"],
            "task_usage": ["start_time", "end_time", "task_id", "machine_id", "mean_cpu_usage_rate", "canonical_memory_usage"],
            "machine_events": ["timestamp", "event_type", "capacity_cpu", "capacity_memory"]
        }
        
        try:
            job_df = pd.read_csv(os.path.join(job_path, csv_name)).dropna(subset=datasets["job_events"])
            task_events_df = pd.read_csv(os.path.join(task_events_path, csv_name)).dropna(subset=datasets["task_events"])
            task_usage_df = pd.read_csv(os.path.join(task_usage_path, csv_name)).dropna(subset=datasets["task_usage"])
        except pd.errors.EmptyDataError:
            continue

        job_df.sort_values(datasets["job_events"], inplace=True)
        task_events_df.sort_values(datasets["task_events"], inplace=True)
        task_usage_df.sort_values(datasets["task_usage"], inplace=True)

        job_id = os.path.splitext(csv_name)[0]
        first_job_df = job_df.iloc[0]
        first_job_node = f"{'job_events'}_{first_job_df['timestamp']}_{first_job_df['event_type']}"

        # Add job node & edge with timestamp
        add_nodes_and_edges(G, job_df, node_type="job", node_name_prefix="job_events", 
                            key_attrs=datasets["job_events"][:2], additional_attrs=datasets["job_events"][2:])
        #print("Add job node & edge with timestamp: ",G.number_of_nodes(), G.number_of_edges())

        # Add task events node & edge with timestamp
        task_events_dict = add_nodes_and_edges(G, task_events_df, node_type="task_events", node_name_prefix="task_events", 
                            key_attrs=datasets["task_events"][:2], additional_attrs=datasets["task_events"][2:], job_to=first_job_node)
        #print("Add task events node & edge with timestamp: ", G.number_of_nodes(), G.number_of_edges())

        # Add task usage node & edge with timestamp
        task_usage_dict = add_nodes_and_edges(G, task_usage_df, node_type="task_usage", node_name_prefix="task_usage", 
                            key_attrs=datasets["task_usage"][:2], additional_attrs=datasets["task_usage"][2:], job_to=first_job_node)
        #print("Add task usage node & edge with timestamp: ", G.number_of_nodes(), G.number_of_edges())
        
        # Add edge btween task_events & task_usages
        for task_id, events_nodes in task_events_dict.items():
                if task_id in task_usage_dict:
                    G.add_edge(events_nodes[0], task_usage_dict[task_id][0])
        #print("Add edge btween task_events & task_usages: ", G.number_of_nodes(), G.number_of_edges())
        
        # Add edge between tasks & machines
        connect_tasks_to_machines(G, task_events_dict, machine_events_path, machine_events_list)
        connect_tasks_to_machines(G, task_usage_dict, machine_events_path, machine_events_list)
        #print("Add edge between tasks & machines", G.number_of_nodes(), G.number_of_edges())

        #isolated = list(nx.isolates(G))
        #G.remove_nodes_from(isolated)

        if not nx.is_connected(G):
            print("Not connected")
            G.clear()
        else:
            save_graph(G, job_id)
            save_graph_status(job_id, G.number_of_nodes(), G.number_of_edges())
            #print(G.number_of_nodes(), G.number_of_edges())

    return True


def filtering_node_data(df, low_percent, high_percent):
    low_quantile = df["n_node"].quantile(low_percent)
    high_quantile = df["n_node"].quantile(high_percent)
    
    filtered_df = df[(df["n_node"] > low_quantile) & (df["n_node"] < high_quantile)]
    
    return filtered_df


def show_node_dist():
    data_path = "../datas/graphs/graph_stats.csv"
    
    df = pd.read_csv(data_path)

    filtered_df = filtering_node_data(df, 0.1, 0.9)
    
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_df["n_node"], bins=20, color="skyblue", edgecolor="black")
    plt.title("Distribution of Node Counts")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def save_data_by_id(output_path, dataset_path, data_type, header_list):
    save_path = os.path.join(output_path, data_type)
    save_data = os.path.join(dataset_path, data_type)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for file_name in tqdm(os.listdir(save_data)):
        if file_name.endswith(".csv"):
            file_path = os.path.join(save_data, file_name)

            data = pd.read_csv(file_path, usecols=header_list)

            for job_id, group in data.groupby("job_id"):
                group = group.drop(columns=["job_id"])
                output_file_path = os.path.join(save_path, f"{job_id}.csv")

                if os.path.exists(output_file_path):
                    group.to_csv(output_file_path, mode='a', header=False, index=False)
                else:
                    group.to_csv(output_file_path, index=False)


def save_preprocessed_data():
    output_path = "../datas/cluserdata_preprocessed"
    dataset_path = "../datas/clusterdata-2011-2"
    table_names = ["job_events", "machine_attributes", "machine_events", "task_constraints", "task_events", "task_usage"]

    machine_col = ["timestamp", "machine_id", "event_type", "platform_id", "capacity_cpu", "capacity_memory"]
    machine_attrs_col = ["timestamp", "machine_id", "attribute_name", "attribute_value", "attribute_deleted"]
    job_col = ["timestamp", "missing_info", "job_id", "event_type", "user_name", "scheduling_class", "job_name", "logical_job_name"]
    task_col = ["timestamp", "missing_info", "job_id", "task_id", "machine_id", "event_type", "user_name", "scheduling_class",
                 "priority", "request_cpu", "request_memory", "request_disk", "machine_constraint"]
    task_usage_col = ["start_time", "end_time", "job_id", "task_id", "machine_id", "mean_cpu_usage_rate", 
                      "canonical_memory_usage", "assigned_memory_usage", "unmapped_page_cache_memory_usage", 
                      "total_page_cache_memory_usage", "maximum_memory_usage", "mean_disk_io_time", 
                      "mean_local_disk_space_used", "maximum_cpu_usage", "maximum_disk_io_time", "cpi",
                      "mai", "sample_portion", "aggregation_type", "sampled_cpu_usage"]
    task_const_col = ["timestamp", "job_id", "task_id", "attribute_name", "attribute_value", "comparison_operator"]
    
    machine_cols_to_use = ["timestamp", "machine_id", "event_type", "capacity_cpu", "capacity_memory"]
    machine_attrs_cols_to_use = ["timestamp", "machine_id", "attribute_name", "attribute_value", "attribute_deleted"]
    job_cols_to_use = ["timestamp", "job_id", "event_type", "user_name", "scheduling_class"]
    task_cols_to_use = ["timestamp", "job_id", "task_id", "machine_id", "event_type", "user_name", "scheduling_class", "priority", 
                        "request_cpu", "request_memory"]
    task_usage_cols_to_use = ["start_time", "end_time", "job_id", "task_id", "machine_id", "mean_cpu_usage_rate", "canonical_memory_usage"]
    task_const_cols_to_use = ["timestamp", "job_id", "task_id", "attribute_name", "attribute_value"]

    save_data_by_id(output_path, dataset_path, table_names[5], task_usage_cols_to_use)


def preprocessing():
    output_path = "../datas"
    dataset_path = "../datas/clusterdata-2011-2"
    table_names = ["job_events", "machine_attributes", "machine_events", "task_constraints", "task_events", "task_usage"]

    # Add machine node
    print("Read csv!")
    machine_col = ["timestamp", "machine_id", "event_type", "platform_id", "capacity_cpu", "capacity_memory"]
    machine_attrs_col = ["timestamp", "machine_id", "attribute_name", "attribute_value", "attribute_deleted"]
    job_col = ["timestamp", "missing_info", "job_id", "event_type", "user_name", "scheduling_class", "job_name", "logical_job_name"]
    task_col = ["timestamp", "missing_info", "job_id", "task_id", "machine_id", "event_type", "user_name", "scheduling_class",
                 "priority", "request_cpu", "request_memory", "request_disk", "machine_constraint"]
    task_usage_col = ["start_time", "end_time", "job_id", "task_id", "machine_id", "mean_cpu_usage_rate", 
                      "canonical_memory_usage", "assigned_memory_usage", "unmapped_page_cache_memory_usage", 
                      "total_page_cache_memory_usage", "maximum_memory_usage", "mean_disk_io_time", 
                      "mean_local_disk_space_used", "maximum_cpu_usage", "maximum_disk_io_time", "cpi",
                      "mai", "sample_portion", "aggregation_type", "sampled_cpu_usage"]
    task_const_col = ["timestamp", "job_id", "task_id", "attribute_name", "attribute_value", "comparison_operator"]
    
    machine_cols_to_use = ["timestamp", "machine_id", "event_type", "capacity_cpu", "capacity_memory"]
    machine_attrs_cols_to_use = ["timestamp", "machine_id", "attribute_name", "attribute_value", "attribute_deleted"]
    job_cols_to_use = ["timestamp", "job_id", "event_type", "user_name", "scheduling_class"]
    task_cols_to_use = ["timestamp", "job_id", "task_id", "machine_id", "event_type", "user_name", "scheduling_class", "priority", 
                        "request_cpu", "request_memory"]
    task_usage_cols_to_use = ["start_time", "end_time", "job_id", "task_id", "machine_id", "mean_cpu_usage_rate", "canonical_memory_usage"]
    task_const_cols_to_use = ["timestamp", "job_id", "task_id", "attribute_name", "attribute_value"]

    csv_range = TOTAL_ITER

    ## If you want to run preprocess_graph_data(), fix parameters

    # Default: False
    SAVE_NEW_DATA = False

    if SAVE_NEW_DATA:
        save_preprocessed_graph_data(dataset_path, csv_range, task_cols_to_use, task_usage_cols_to_use)

    graph = make_graph(output_path)

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
    graphs = preprocessing()

    #save_graph(graphs)

    #load_graph(6253708944)

    #show_node_dist()

    #save_preprocessed_data()

    #summarize_task_usage()