import argparse
from config import Config
import torch
from torch.utils.data import DataLoader, random_split
from transformers import TimesformerForVideoClassification, TimesformerConfig
from load_dataset import FrameDataset
from train import test
import random
import os
from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms as T
import cv2

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Configure model training parameters.")
    parser.add_argument("--path", type=str, default="Dataset", help="Path to the dataset.")
    parser.add_argument("--image_size", type=int, default=128, help="Size of the input images.")
    parser.add_argument("--step_size", type=int, default=10, help="Step size for frame extraction.")
    parser.add_argument("--clip_duration", type=int, default=30, help="Duration of video clips.")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of frames per clip.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--augmentation", type=str, nargs='*', default=[], help="List of augmentation techniques to apply. Options include 'flip', 'color_jitter', 'rotation', 'sketch'.")
    parser.add_argument("--model", type=str, default="models/model_no_augmentations.pt", help="Path to save the model.")
    parser.add_argument("--log_file", type=str, default="models/logs_no_aug.txt", help="Path to save the logs.")
    parser.add_argument("--prediction_file", type=str, default="models/predictions_no_aug.csv", help="Path to save the predictions.")
    parser.add_argument("--save_roc_path", type=str, default="models/roc_no_aug.txt", help="Path to save the ROC curve.")
    
    return vars(parser.parse_args())

def get_transforms():

    transforms_list = [T.Resize((m_config.image_size, m_config.image_size))]
    transforms_list.append(T.ToTensor())
    transforms_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms_list)

def transform(frames):
    resize_transform = get_transforms()
    # Resize and normalize each frame
    resized_frames = torch.stack([resize_transform(frame) for frame in frames])
    # Ensure the tensor has the correct shape [num_frames, num_channels, height, width]
    assert resized_frames.shape == (m_config.num_frames, 3, m_config.image_size, m_config.image_size), f"Unexpected shape: {resized_frames.shape}"
    return resized_frames

def main():

    args = parse_args()
    
    global m_config 
    m_config = Config.from_dict(args)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parent_directory = m_config.path
    patient_dirs = [os.path.join(parent_directory, 'DRI-001_frames'), os.path.join(parent_directory, 'DRI-006_frames')]
    
    test_dataset = FrameDataset(patient_dirs=patient_dirs, transform=lambda frames: transform(frames), num_frames=m_config.num_frames)
    
    test_loader = DataLoader(test_dataset, batch_size=m_config.batch_size, shuffle=False)
    
    # Load pre-trained TimeSformer model
    config = TimesformerConfig.from_pretrained("facebook/timesformer-base-finetuned-k400")
    config.image_size = m_config.image_size 
    
    model = TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400",
        config=config,
        ignore_mismatched_sizes=True
    )
    
    # Adjust the position embeddings
    def resize_pos_embed(posemb, num_patches, hidden_size):
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        gs_old = int(posemb_grid.shape[0] ** 0.5)
        gs_new = int(num_patches ** 0.5)
        print(f'Resizing position embedding grid from {gs_old}x{gs_old} to {gs_new}x{gs_new}')
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, hidden_size).permute(0, 3, 1, 2)
        posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, hidden_size)
        return torch.cat([posemb_tok, posemb_grid], dim=1)
    
    num_patches = (config.image_size // config.patch_size) ** 2
    model.timesformer.embeddings.position_embeddings = torch.nn.Parameter(
        resize_pos_embed(model.timesformer.embeddings.position_embeddings, num_patches, model.config.hidden_size)
    )
    
    model.to(device)

    # Load model
    if m_config.model:
        print(f"Loading model from {m_config.model}")
        model.load_state_dict(torch.load(m_config.model, map_location=device))
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=m_config.lr)
    
    test(model, test_loader, device, m_config.model, m_config.log_file, m_config.prediction_file, m_config.save_roc_path)

if __name__ == "__main__":
    main()
