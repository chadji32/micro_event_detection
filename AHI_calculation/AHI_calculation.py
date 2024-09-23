import torch
from torch.utils.data import DataLoader, random_split
import random
import os
from PIL import Image
import pandas as pd
from datetime import timedelta

def extract_patient_id(frame_path):
    return frame_path.split('/')[2]

def calculate_total_sleep_time(timestamps):
    # Convert timestamps to timedelta
    timestamps = pd.to_timedelta(timestamps)
    
    # Calculate the total duration from the first to the last timestamp
    total_duration = (timestamps.max() - timestamps.min()).total_seconds()
    
    # Convert duration to hours
    total_duration_hours = total_duration / 3600.0
    return total_duration_hours

def calculate_ahi_for_patients(predictions_csv_path):
    df = pd.read_csv(predictions_csv_path)

    df['patient_id'] = df['frame_path'].apply(extract_patient_id)

    patient_ahi = {}

    for patient_id, patient_data in df.groupby('patient_id'):
        patient_data['is_event'] = patient_data['predicted_label'].apply(lambda x: x in [1, 2])

        patient_data['event_group'] = (patient_data['is_event'] != patient_data['is_event'].shift()).cumsum()

        apnea_events = (patient_data['predicted_label'] == 2).sum()
        hypopnea_events = (patient_data['predicted_label'] == 1).sum()

        total_sleep_time = calculate_total_sleep_time(patient_data['timestamp'])

        ahi = (apnea_events + hypopnea_events) / total_sleep_time

        severity = classify_ahi(ahi)

        patient_ahi[patient_id] = {
            'Total Apnea Events': apnea_events,
            'Total Hypopnea Events': hypopnea_events,
            'Total Sleep Time (hours)': total_sleep_time,
            'AHI': ahi,
            'Severity': severity
        }

        print(f"Patient ID: {patient_id}")
        print(f"Total Apnea Events: {apnea_events}")
        print(f"Total Hypopnea Events: {hypopnea_events}")
        print(f"Total Sleep Time (hours): {total_sleep_time:.2f}")
        print(f"AHI: {ahi:.2f} events per hour")
        print(f"Severity: {severity}")
        print("-" * 30)

    return patient_ahi

def classify_ahi(ahi):
    if ahi < 5:
        return "Normal"
    elif 5 <= ahi < 15:
        return "Mild"
    elif 15 <= ahi < 30:
        return "Moderate"
    else:
        return "Severe"

# Set the .csv file name
patient_ahi = calculate_ahi_for_patients("predictions_no_aug.csv")

for patient_id, ahi_data in patient_ahi.items():
    print(f"Patient ID: {patient_id}")
    print(f"AHI: {ahi_data['AHI']:.2f} events per hour")
    print(f"Severity: {ahi_data['Severity']}")

