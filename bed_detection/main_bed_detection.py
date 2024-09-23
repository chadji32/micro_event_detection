import os
import shutil
import glob
from bed_detector import BedDetector, process_videos

def find_variables_directory(patient_path):
    variable_pattern = os.path.join(patient_path, '*_variables')
    variable_directories = glob.glob(variable_pattern)
    if len(variable_directories) == 1:
        return variable_directories[0]
    elif len(variable_directories) > 1:
        print(f"Multiple variable directories found for {patient_path}: {variable_directories}")
        return variable_directories[0]
    else:
        print(f"No variable directory found for {patient_path}")
        return None

def copy_selected_txt_files(src_folder, dest_folder):
    selected_files = ['Classification Arousals.txt', 'Flow Events.txt', 'Snore Events.txt']
    os.makedirs(dest_folder, exist_ok=True)
    for txt_file in selected_files:
        src_file_path = os.path.join(src_folder, txt_file)
        if os.path.exists(src_file_path):
            shutil.copy(src_file_path, dest_folder)
        else:
            print(f"File {txt_file} not found in {src_folder}")

def main():
    base_path = "" #Set the base path
    parent_folders = ['DRI-006', 'DRI-001']
    output_parent_folder = 'output_videos'
    yolo_weights = "yolov3.weights"
    yolo_cfg = "yolov3.cfg"
    coco_names = "coco.names"
    target_size = (320, 240)

    for parent_folder in parent_folders:
        parent_folder_path = os.path.join(base_path, parent_folder)
        if not os.path.exists(parent_folder_path):
            print(f"Parent folder {parent_folder_path} does not exist.")
            continue
        
        patient_folders = os.listdir(parent_folder_path)
        for patient_folder in patient_folders:
            patient_path = os.path.join(parent_folder_path, patient_folder)
            video_folder = os.path.join(patient_path, f'{patient_folder}_video')
            variable_folder = find_variables_directory(patient_path)
            output_patient_folder = os.path.join(output_parent_folder, parent_folder, patient_folder)
            os.makedirs(output_patient_folder, exist_ok=True)

            if variable_folder is None:
                print(f"Skipping bed detection for {patient_folder} due to missing variable directory.")
                continue

            # Initialize Bed Detector
            detector = BedDetector(yolo_weights, yolo_cfg, coco_names)
            process_videos(video_folder, output_patient_folder, detector, target_size)

            # Copy selected .txt files from the variables directory
            output_variable_folder = os.path.join(output_patient_folder, f'{patient_folder}_variables')
            copy_selected_txt_files(variable_folder, output_variable_folder)

            print(f"Processing completed for {patient_folder}")

if __name__ == "__main__":
    main()
