import os
import csv

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


def main():
    input_directory = "."
    
    sub_directories = [subdir for subdir in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, subdir)) and subdir != 'output']

    print(sub_directories)
    
    for subdir in sub_directories:
        subdir_path = os.path.join(input_directory, subdir)
        convert_gwat_to_csv(subdir_path)


if __name__ == "__main__":
    main()