import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from collections import Counter

class FrameDataset(Dataset):
    def __init__(self, patient_dirs, transform=None, num_frames=8):
        self.data = []
        self.transform = transform
        self.num_frames = num_frames

        for patient_dir in patient_dirs:
            if os.path.isdir(patient_dir):
                for csv_file in os.listdir(patient_dir):
                    if csv_file.endswith('.csv'):
                        csv_path = os.path.join(patient_dir, csv_file)
                        df = pd.read_csv(csv_path)
                        self.data.append(df)

        self.data = pd.concat(self.data, ignore_index=True)
        self.sample_list = self._create_sample_list()

    def _create_sample_list(self):
        sample_list = []
        current_sample = []
        current_patient = None

        for _, row in self.data.iterrows():
            patient_id = row['frame_path'].split('/')[2]
            
            if current_patient is None:
                current_patient = patient_id
            
            if patient_id != current_patient or len(current_sample) == self.num_frames:
                if len(current_sample) == self.num_frames:
                    
                    labels = [item['label'] for item in current_sample]
                    most_common_label = Counter(labels).most_common(1)[0][0]
                    for item in current_sample:
                        item['label'] = most_common_label
                    sample_list.append(current_sample)
                
                current_sample = []
                current_patient = patient_id

            current_sample.append(row)

        if len(current_sample) == self.num_frames:
            labels = [item['label'] for item in current_sample]
            most_common_label = Counter(labels).most_common(1)[0][0]
            for item in current_sample:
                item['label'] = most_common_label
            sample_list.append(current_sample)

        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        frames_data = self.sample_list[idx]
        
        frames_paths = [item['frame_path'] for item in frames_data]
        frames = [Image.open(frame_path).convert('RGB') for frame_path in frames_paths]

        if self.transform:
            frames = self.transform(frames)

        label = frames_data[0]['label']
        
        # Debugging: Print the current sample details
        #print(f"Sample {idx}:")
        #print(f"  Frame paths: {frames_paths}")
        #print(f"  Labels: {[item['label'] for item in frames_data]}")
        #print(f"  Most common label: {label}")
        #print(f"  Number of frames: {len(frames_paths)}")

        # For the binary model uncomment the following
        # if label == 1 or label == 2:
        #   label = 1
        
        return frames, torch.tensor(label), frames_paths

