import os
import re
from datetime import datetime

def parse_file(file_content):

    events = []
    for line in file_content.splitlines():
        match = re.search(r'(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}),(\d{3})-(\d{2}:\d{2}:\d{2}),(\d{3}); (\d+);(Apnea|Hypopnea); (Wake|N1|N2|N3|REM)', line)
        if match:
            start_time = datetime.strptime(match.group(1) + ',' + match.group(2), "%d.%m.%Y %H:%M:%S,%f")
            event_type = match.group(6)
            events.append((start_time, event_type))
    
    return events

def calculate_ahi(events):
    # Calculate total sleep time
    if not events:
        return 0
    
    start_time = events[0][0]
    end_time = events[-1][0]
    total_sleep_time = (end_time - start_time).total_seconds() / 3600.0  # convert to hours

    # Count Apneas and Hypopneas
    apnea_count = sum(1 for event in events if event[1] == "Apnea" or event[1] == "Mixed Apnea")
    hypopnea_count = sum(1 for event in events if event[1] == "Hypopnea" or event[1] == "Obstructive Hypopnea" or event[1] == "Obstructive Apnea")

    # Calculate AHI
    ahi = (apnea_count + hypopnea_count) / total_sleep_time
    return ahi

def classify_ahi(ahi):
    if ahi < 5:
        return "Normal"
    elif 5 <= ahi < 15:
        return "Mild"
    elif 15 <= ahi < 30:
        return "Moderate"
    else:
        return "Severe"

def process_files_in_directory(directory_path):
    results = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'r') as file:
                file_content = file.read()
            
            events = parse_file(file_content)
            ahi = calculate_ahi(events)
            patient_id = filename.split('.')[0]  # Assuming filename without extension is the patient ID
            classification = classify_ahi(ahi)
            results.append((patient_id, ahi, classification))
    
    return results

directory_path = os.getcwd()
ahi_results = process_files_in_directory(directory_path)

# Output the results
for patient_id, ahi, classification in ahi_results:
    print(f"Patient ID: {patient_id}, AHI: {ahi:.2f}, Classification: {classification}")
