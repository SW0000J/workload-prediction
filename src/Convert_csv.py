import csv
import gzip
import os


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


if __name__ == "__main__":
    # test_gwat()
    test_gc()