import networkx as nx
import numpy as np
import os
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
import torch
from tqdm import tqdm


TOTAL_ITER = 500


def save_graph(job_graphs):
    output_dir = "../datas/graphs"
    os.makedirs(output_dir, exist_ok=True)

    for job_id, graph in job_graphs.items():
        output_path = os.path.join(output_dir, f"job_{job_id}.gpickle")
        nx.write_gpickle(graph, output_path)
        print(f"Graph for job {job_id} saved to {output_path}")


def load_graph():
    graph_dir = "../datas/graphs"
    # Todo


def preprocessing():
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

    
    machines_df = pd.read_csv(f"{dataset_path}/machine_events/part-00000-of-00001.csv", header=None, names=machine_col)
    machines_df = machines_df[machine_cols_to_use]

    #csv_range = TOTAL_ITER
    csv_range = 10

    job_graphs = {}

    jobs_df = pd.DataFrame(columns=job_cols_to_use)
    tasks_df = pd.DataFrame(columns=task_cols_to_use)
    task_usages_df = pd.DataFrame(columns=task_usage_cols_to_use)


    for iter in tqdm(range(csv_range), desc="Processing CSV files"):
        job_df = pd.read_csv(f"{dataset_path}/job_events/part-{iter:05d}-of-00500.csv", header=None, names=job_col)
        jobs_df = pd.concat([jobs_df, job_df[job_cols_to_use]])

        task_df = pd.read_csv(f"{dataset_path}/task_events/part-{iter:05d}-of-00500.csv", header=None, names=task_col)
        tasks_df = pd.concat([tasks_df, task_df[task_cols_to_use]])

        task_usage_df = pd.read_csv(f"{dataset_path}/task_usage/part-{iter:05d}-of-00500.csv", header=None, names=task_usage_col)
        task_usages_df = pd.concat([task_usages_df, task_usage_df[task_usage_cols_to_use]])

    for job_id in jobs_df['job_id'].unique():
        G = nx.Graph()
        job_tasks = tasks_df[tasks_df['job_id'] == job_id]

        for _, task_row in job_tasks.iterrows():
            machine_id = task_row['machine_id']
            if pd.notnull(machine_id):
                G.add_node(machine_id, node_type='machine')
                for _, usage_row in task_usages_df[(task_usages_df['job_id'] == job_id) & (task_usages_df['task_id'] == task_row['task_id'])].iterrows():
                    if machine_id == usage_row['machine_id']:
                        G.add_edge(machine_id, job_id,
                                    task_id=task_row['task_id'],
                                    mean_cpu_usage_rate=usage_row['mean_cpu_usage_rate'],
                                    maximum_memory_usage=usage_row['maximum_memory_usage'])

        job_graphs[job_id] = G
        print(G)

    return job_graphs


    """
    dataset_path = "../datas/clusterdata-2011-2"
    table_names = ["job_events", "machine_attributes", "machine_events", "task_constraints", "task_events", "task_usage"]
    
    G = nx.Graph()

    # Add machine node
    print("Setting machine node!")
    machine_col = ["timestamp", "machine_id", "event_type", "platform_id", "capacity_cpu", "capacity_memory"]

    machines_df = pd.read_csv(f"{dataset_path}/machine_events/part-00000-of-00001.csv")
    machines_df = pd.DataFrame(machines_df, columns=machine_col)
    
    for index, row in machines_df.iterrows():
        G.add_node(row["machine_id"], node_type="machine", capacity_cpu=row["capacity_cpu"], capacity_memory=row["capacity_memory"])

    # Define user set
    user_df = pd.DataFrame()
    csv_range = 1

    # Add job node
    print("Setting job node!")
    job_col = ["timestamp", "missing_info", "job_id", "event_type", "user_name", "scheduling_class", "job_name", "logical_job_name"]

    for iter in tqdm(range(csv_range)):
        jobs_df = pd.read_csv(f"{dataset_path}/job_events/part-{iter:05d}-of-00500.csv")
        jobs_df = pd.DataFrame(jobs_df, columns=job_col)

        for index, row in jobs_df.iterrows():
            G.add_node(row["job_id"], node_type="job", user=row["user_name"])

        user_df = pd.concat([user_df, jobs_df["user_name"]])

    # Add task node
    print("Setting task node!")
    tasks_col = ["timestamp", "missing_info", "job_id", "task_id", "machine_id", "event_type", "user_name", "scheduling_class",
                 "priority", "request_cpu", "request_memory", "request_disk", "machine_constraint"]

    for iter in tqdm(range(csv_range)):
        tasks_df = pd.read_csv(f"{dataset_path}/task_events/part-{iter:05d}-of-00500.csv")
        tasks_df = pd.DataFrame(tasks_df, columns=tasks_col)

        for index, row in tasks_df.iterrows():
            G.add_node(row["task_id"], node_type="task", job_id=row["job_id"], cpu_request=row["request_cpu"], memory_request=row["request_memory"])

        user_df = pd.concat([user_df, tasks_df["user_name"]])

    user_set = set(user_df)
    print(user_set)

    # 엣지 추가 예시 (실제 데이터로 채워야 함)
    # 작업과 기계 간 엣지 추가
    for index, row in tasks_df.iterrows():
        G.add_edge(row['task_id'], row['machine_id'])

    # 작업과 작업 간 엣지 추가 (작업이 속한 작업을 기반으로)
    for index, row in tasks_df.iterrows():
        G.add_edge(row['task_id'], row['job_id'])

    # 작업 및 작업과 사용자 간 엣지 추가
    for index, row in jobs_df.iterrows():
        G.add_edge(row['job_id'], row['user_name'])
        for task_row in tasks_df[tasks_df['job_id'] == row['job_id']].iterrows():
            G.add_edge(task_row[1]['task_id'], row['user_name'])"""


if __name__ == "__main__":
    graphs = preprocessing()
    save_graph(graphs)