import os
from create_frames import PatientDatasetPreparer

# Define parent directory and sub-directories
parent_directory = "Dataset"
patient_dataset = PatientDatasetPreparer(parent_directory)
patient_dataset.prepare_dataset()
