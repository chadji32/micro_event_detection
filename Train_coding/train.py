import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from config import Config
import os


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_log(message, log_file="training_log.txt"):
    with open(log_file, "a") as f:
        f.write(message + "\n")

def train(model, data_loader, optimizer, device, epoch, num_epochs, log_file="training_log.txt", save_model_path=None):
    loss_meter = AverageMeter()
    model.train()
    tk = tqdm(data_loader, total=len(data_loader), desc='Training', unit='batch', leave=False)
    
    for batch_idx, (frames, labels, _) in enumerate(tk):
        frames, labels = frames.to(device), labels.to(device)
        
        outputs = model(frames)
        
        loss = F.cross_entropy(outputs.logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), labels.size(0))
        tk.set_postfix({"loss": loss_meter.avg})
    
    log_message = f'Train: Average loss: {loss_meter.avg:.4f}'
    print(log_message)
    save_log(log_message, log_file)
    
    # Save the model at the last epoch
    if epoch == num_epochs - 1:
        if save_model_path:
            torch.save(model.state_dict(), save_model_path)
            log_message = f"Model saved to {save_model_path}"
            print(f"Model saved to {save_model_path}")

def validate(model, data_loader, criterion, device, log_file="training_log.txt"):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    model.eval()
    correct = 0
    
    all_preds_o = []
    all_labels_o = []

    tk = tqdm(data_loader, total=len(data_loader), desc='Validation', unit='batch', leave=False)
    
    for batch_idx, (frames, labels, _) in enumerate(tk):
        frames, labels = frames.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(frames)
        
        loss = criterion(outputs.logits, labels)
        preds = outputs.logits.argmax(dim=1)
        correct_this = preds.eq(labels).sum().item()
        correct += correct_this
        acc = correct_this / labels.size(0) * 100.0

        preds_o = outputs.logits.argmax(dim=1).cpu().numpy()
        all_preds_o.extend(preds_o)
        all_labels_o.extend(labels.cpu().numpy())
        
        acc_meter.update(acc, labels.size(0))
        loss_meter.update(loss.item(), labels.size(0))
    
    f1 = f1_score(all_labels_o, all_preds_o, average='macro')
    log_message = (f'Validation: Average loss: {loss_meter.avg:.4f}, '
                   f'Accuracy: {correct}/{len(data_loader.dataset)} ({acc_meter.avg:.2f}%), '
                   f'F1-score: {f1}')
    print(log_message)
    save_log(log_message, log_file)