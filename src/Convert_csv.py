import csv
import gzip
import os
import pandas as pd


def convert_gwat_to_csv(directory):
    output_directory = "./output/" + directory

    os.makedirs(output_directory, exist_ok=True)
    
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    
    for csv_file in csv_files:
        input_path = os.path.join(directory, csv_file)
        output_path = os.path.join(output_directory, csv_file)
        
        with open(input_path, 'r') as f_input, open(output_path, 'w', newline='') as f_output:
            csv_reader = csv.reader(f_input, delimiter=';')
            csv_writer = csv.writer(f_output, delimiter=',')
            
            for row in csv_reader:
                stripped_row = [element.strip() for element in row]
                csv_writer.writerow(stripped_row)


def decompress_google_cluster(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.gz'):
                file_path = os.path.join(root, file)
                # Remove .gz -> [:-3]
                decompressed_file_path = file_path[:-3]
                
                with gzip.open(file_path, 'rb') as f_in:
                    with open(decompressed_file_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                print(f"Decompressed: {file_path} -> {decompressed_file_path}")

    
def sort_gc_dataset(sub_path, col, sort_by, test=True):
    base_directory = f"../datas/clusterdata-2011-2/{sub_path}/"
    output_directory = f"../datas/{sub_path}/"

    csv_files = [file for file in os.listdir(base_directory) if file.endswith('.csv')]

    for csv_file in csv_files:
        df = pd.read_csv(base_directory+csv_file, header=None, names=col)
        df_sorted = df.sort_values(by=sort_by)

        if test:
            os.makedirs(output_directory, exist_ok=True)
            df_sorted.to_csv(output_directory+csv_file, index=False)
            break
        else:
            df_sorted.to_csv(base_directory+csv_file, index=False)


def test_gwat():
    input_directory = "."
    
    sub_directories = [subdir for subdir in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, subdir)) and subdir != 'output']

    print(sub_directories)
    
    for subdir in sub_directories:
        subdir_path = os.path.join(input_directory, subdir)
        convert_gwat_to_csv(subdir_path)


def test_gc():
    base_directory = "../datas/clusterdata-2011-2"
    subdirectories = ["job_events", "machine_attributes", "machine_events", 
                      "task_constraints", "task_events", "task_usage"
                      ]
    decompress_google_cluster(base_directory)


def test_sort_gc():
    machine_col = ["timestamp", "machine_id", "event_type", "platform_id", "capacity_cpu", "capacity_memory"]
    job_col = ["timestamp", "missing_info", "job_id", "event_type", "user_name", "scheduling_class", "job_name", "logical_job_name"]
    task_col = ["timestamp", "missing_info", "job_id", "task_id", "machine_id", "event_type", "user_name", "scheduling_class",
                 "priority", "request_cpu", "request_memory", "request_disk", "machine_constraint"]
    task_usage_col = ["start_time", "end_time", "job_id", "task_id", "machine_id", "mean_cpu_usage_rate", 
                      "canonical_memory_usage", "assigned_memory_usage", "unmapped_page_cache_memory_usage", 
                      "total_page_cache_memory_usage", "maximum_memory_usage", "mean_disk_io_time", 
                      "mean_local_disk_space_used", "maximum_cpu_usage", "maximum_disk_io_time", "cpi",
                      "mai", "sample_portion", "aggregation_type", "sampled_cpu_usage"]
    subdirectories = ["job_events", "machine_attributes", "machine_events", 
                      "task_constraints", "task_events", "task_usage"
                      ]
    
    dtype_options = {"start_time": int, "end_time": int, "job_id": int, "task_id": int,
                     "machine_id": int, "mean_cpu_usage_rate": float, "canonical_memory_usage": float,
                     "assigned_memory_usage": float, "unmapped_page_cache_memory_usage": float,
                     "total_page_cache_memory_usage": float, "maximum_memory_usage": float, "mean_disk_io_time": float,
                     "mean_local_disk_space_used": float, "maximum_cpu_usage": float, "maximum_disk_io_time": float,
                     "cpi": float, "mai": float, "sample_portion": int, "aggregation_type": int, "sampled_cpu_usage": float}

    sort_gc_dataset(subdirectories[5], task_usage_col, ["job_id", "task_id", "machine_id"], False)


if __name__ == "__main__":
    # test_gwat()
    #test_gc()
    test_sort_gc()