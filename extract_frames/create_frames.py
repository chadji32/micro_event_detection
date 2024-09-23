import os
import numpy as np
import pandas as pd
import re
import cv2
from datetime import timedelta
from config import *
import glob

class PatientDatasetPreparer:
    def __init__(self, data_dir, max_patients=None, default_label="No event", fps=25):
        self.data_dir = data_dir
        self.clip_duration = config.clip_duration  # Duration of each clip in seconds
        self.step_size = config.step_size  # Step size for sliding window in seconds
        self.max_patients = max_patients
        self.default_label = default_label
        self.num_frames = int(config.clip_duration * fps)  # Total frames per clip
        self.fps = fps

    # Read "Flow Event.txt" file and extract the start and end event's timestamp for each patient
    def read_label_file(self, label_file):
        with open(label_file, 'r') as file:
            lines = file.readlines()
        
        start_time_str = lines[1].split(": ")[1].strip()
        start_time = pd.to_datetime(start_time_str, format='%d/%m/%Y %H:%M:%S')
        events = []

        for line in lines[4:]:
            parts = line.split(';')
            if len(parts) < 3:
                print(f"Skipping malformed line: {line.strip()}")
                continue 

            event_time = parts[0].split('-')
            try:
                event_start = pd.to_datetime(event_time[0].strip(), format='%d.%m.%Y %H:%M:%S,%f')
                event_end = pd.to_datetime(event_time[1].strip(), format='%H:%M:%S,%f')
                event_end = event_start.replace(hour=event_end.hour, minute=event_end.minute, second=event_end.second, microsecond=event_end.microsecond)
            except ValueError as e:
                print(f"Error parsing date in line: {line.strip()} | Error: {e}")
                continue

            label = parts[2].strip()
            
            # From the "Flow Events.txt" assign same conditions under one class
            if label == 'Mixed Apnea' or label == 'Apnea':
                label = 'Apnea'
            elif label == 'Obstructive Hypopnea' or label == 'Obstructive Apnea' or label == 'Hypopnea':
                label = 'Hypopnea'
            else:
                label = 'No event'
            
            events.append((event_start, event_end, label))

        return events

    # Read .vid files, which contains the metadata for each patient's videos
    def read_vid_file(self, vid_file):
        metadata = {}
        with open(vid_file, 'r') as file:
            lines = file.readlines()
            current_video = None
            file_name = None
            general_start_time = None
            for line in lines:
                line = line.strip()
                if line.startswith("[General]"):
                    current_video = "General"
                elif line.startswith("[Video"):
                    current_video = line.strip('[]')
                elif line.startswith("FileName="):
                    file_name = line.split('=')[1].strip().lower()
                elif line.startswith("Start="):
                    start_time = pd.to_datetime(line.split('=')[1].strip(), format='%d.%m.%Y %H:%M:%S,%f')
                    if current_video == "General":
                        general_start_time = start_time
                    if current_video and file_name:
                        metadata[file_name] = {'start_time': start_time}
    
        if general_start_time:
            metadata['general_start_time'] = general_start_time
    
        return metadata

    def load_patient_data(self, patient_id, base_dir):
        video_dir = os.path.join(base_dir, patient_id, patient_id + '_video')
        variable_dir = os.path.join(base_dir, patient_id, patient_id + '_variables')
    
        label_file = os.path.join(variable_dir, 'Flow Events.txt')
        events = self.read_label_file(label_file)
    
        vid_file_path = glob.glob(os.path.join(video_dir, '*.vid'))
        if not vid_file_path:
            raise FileNotFoundError(f"No .vid files found in directory: {video_dir}")
        vid_file = vid_file_path[0]
        metadata = self.read_vid_file(vid_file)
    
        def extract_digits(filename):
            match = re.search(r'_(\d{1,2})(?=\.asf$)', filename)
            return int(match.group(1)) if match else -1

        # Sort the patients' videos base on the last one or two digits 
        video_files = sorted(
            [f for f in os.listdir(video_dir) if f.endswith('.asf')],
            key=extract_digits
        )
    
        clips = []
        labels = []
        num = 0
        for video_file in video_files:
            video_file_path = os.path.join(video_dir, video_file)
            _, num = self.process_video_file(video_file_path, events, metadata, patient_id, base_dir, num)


    def get_event_label(self, clip_start, clip_end, events, start_time):
        clip_start_time = start_time + pd.Timedelta(seconds=clip_start)
        clip_end_time = start_time + pd.Timedelta(seconds=clip_end)
        for event_start, event_end, event_label in events:
            if event_start < clip_end_time and clip_start_time < event_end:
                return event_label
        return self.default_label

    def process_video_file(self, video_file_path, events, metadata, patient_id, base_dir, num):

        label_mapping = {
            "No event": 0,
            "Hypopnea": 1,
            "Apnea": 2
        }
        
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_file_path}")
            return None, None

        clips = []
        labels = []

        video_file_path_temp = os.path.basename(video_file_path).lower()
        video_file_name = video_file_path_temp.split('/')[-1]

        if video_file_name not in metadata:
            print(f"Error: Metadata for video file {video_file_name} not found")
            return None, None

        video_start_time = metadata[video_file_name]['start_time']
        video_end_time = video_start_time + pd.Timedelta(minutes=40)  # Each video lasts 40 minutes
        clip_start_time = 0

        while clip_start_time + self.clip_duration <= (video_end_time - video_start_time).total_seconds():
            clip_end_time = clip_start_time + self.clip_duration

            # Assign label to the clip
            label = self.get_event_label(clip_start_time, clip_end_time, events, video_start_time)
            clips.append((clip_start_time, clip_end_time))
            labels.append(label)

            clip_start_time = clip_end_time + self.step_size

        sub_dir_name = os.path.basename(base_dir)

        # Create the frames directory
        frames_dir = os.path.join(self.data_dir, f'{sub_dir_name}_frames', patient_id)
        os.makedirs(frames_dir, exist_ok=True)
        csv_data = []

        frame_interval = self.clip_duration / (self.num_frames - 1)

        for clip, label in zip(clips, labels):
            clip_start_time, clip_end_time = clip
            frame_interval = self.clip_duration / 4 

            for frame_idx in range(5):  # Loop exactly 5 times to get 5 frames
                frame_time_sec = clip_start_time + frame_idx * frame_interval
                cap.set(cv2.CAP_PROP_POS_MSEC, frame_time_sec * 1000)
                ret, frame = cap.read()
                if not ret:
                    break
                frame_time = video_start_time + timedelta(seconds=frame_time_sec)
                frame_time_str = frame_time.strftime('%H:%M:%S')
                frame_filename = f"{num}.png"
                frame_filepath = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_filepath, frame)
                csv_data.append([frame_filepath, frame_time_str, label_mapping[label]])
                num += 1

        cap.release()
        df = pd.DataFrame(csv_data, columns=["frame_path", "timestamp", "label"])
        csv_file_path = os.path.join(frames_dir, f'{patient_id}.csv')
        # Append to the CSV file
        if os.path.exists(csv_file_path):
            df.to_csv(csv_file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file_path, index=False)

        return csv_data, num


    def prepare_dataset(self):
        sub_dirs = ['DRI-001', 'DRI-006']
        patient_ids = []
        base_dirs = []

        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(self.data_dir, sub_dir)
            patient_dirs = [d for d in os.listdir(sub_dir_path) if os.path.isdir(os.path.join(sub_dir_path, d))]
            for patient_dir in patient_dirs:
                patient_ids.append(patient_dir)
                base_dirs.append(sub_dir_path)

        if self.max_patients is not None:
            patient_ids = patient_ids[:self.max_patients]
            base_dirs = base_dirs[:self.max_patients]

        print(f"Found patient IDs: {patient_ids}")

        self.load_and_save_data(list(zip(patient_ids, base_dirs)))

    def load_and_save_data(self, patient_dirs):
        for idx, (patient_id, base_dir) in enumerate(patient_dirs):
            print(f"Loading data for patient {idx + 1}/{len(patient_dirs)}: {patient_id}")
            self.load_patient_data(patient_id, base_dir)
        print("All data loaded and saved.")
