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

        for root_dir in patient_dirs:
            for patient_dir in os.listdir(root_dir):
                patient_path = os.path.join(root_dir, patient_dir)
                if os.path.isdir(patient_path):
                    for csv_file in os.listdir(patient_path):
                        if csv_file.endswith('.csv'):
                            csv_path = os.path.join(patient_path, csv_file)
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
        frames_data = self.data.iloc[idx*self.num_frames:(idx+1)*self.num_frames]
        frames_paths = frames_data['frame_path'].tolist()
        timestamps = frames_data['timestamp'].tolist()
        frames = []
    
        for frame_path in frames_paths:
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)
        
        if self.transform:
            frames = self.transform(frames)
    
        label = frames_data.iloc[0]['label']

        # For the binary model uncommend the following
        # if label == 1 or label == 2:
        #   label = 1
        
        return frames, torch.tensor(label), frames_paths, timestamps

