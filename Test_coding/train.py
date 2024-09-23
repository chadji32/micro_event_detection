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
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.preprocessing import label_binarize

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

def test(model, data_loader, device, save_model_path=None, log_file="training_log.txt", save_csv_path="predictions.csv", save_roc_path="ROC.png"):
    model.eval()
    all_preds_o = []
    all_labels_o = []
    all_probs_o = []
    
    all_preds = []
    all_labels = []
    all_paths = []
    all_timestamps = []
   
    with torch.no_grad():
        for i, (frames, labels, paths, timestamps) in enumerate(data_loader):
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            preds_t = outputs.logits.argmax(dim=1)
            
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            
            all_probs_o.extend(probs)
            all_preds_o.extend(preds)
            all_labels_o.extend(labels.cpu().numpy())
            
            for j in range(len(paths)):
                current_paths = paths[j]
                current_timestamps = timestamps[j]
                
                for k in range(len(current_paths)):
                
                    if i >= len(preds):
                        break
                    path = current_paths[k]
                   
                    timestamp = current_timestamps[k]
                    current_pred = preds_t[i].cpu().item()
                    
                    all_paths.append(path)
                    all_timestamps.append(timestamp)
                    all_preds.append(current_pred)
                
                i += 1

    # Save predictions to CSV
    df = pd.DataFrame({
        'frame_path': all_paths,
        'timestamp': all_timestamps,
        'predicted_label': all_preds
    })
    df.to_csv(save_csv_path, index=False)
    
    f1 = f1_score(all_labels_o, all_preds_o, average='macro')
    precision = precision_score(all_labels_o, all_preds_o, average='macro')
    recall = recall_score(all_labels_o, all_preds_o, average='macro')
    
    log_message = (f'Test Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, '
                   f'F1 Score: {f1:.4f}')
    print(log_message)
    save_log(log_message, log_file)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels_o, all_preds_o)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Print the normalized confusion matrix
    df_cm = pd.DataFrame(cm_normalized, index=[i for i in range(cm.shape[0])], columns=[i for i in range(cm.shape[1])])
    
    print("Normalized Confusion Matrix:")
    print(df_cm)
    
    save_log("Normalized Confusion Matrix:", log_file)
    save_log(df_cm.to_string(), log_file)
    
    # Determine the number of classes
    n_classes = len(np.unique(all_labels_o))

    # Create the ROC curve
    if n_classes == 2:
        
        all_labels_bin = label_binarize(all_labels_o, classes=[0, 1])
        
        fpr, tpr, _ = roc_curve(all_labels_bin, np.array(all_probs_o)[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve for binary classification
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - Binary Classification')
        plt.legend(loc="lower right")
        plt.savefig(save_roc_path)
        plt.close()
    
        print(f"ROC curve saved to {save_roc_path}")
        save_log(f"ROC curve saved to {save_roc_path}", log_file)
