import h5py
import torch
import torch.nn as nn
import argparse
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc, accuracy_score
from torch.utils.data import DataLoader
import numpy as np
import csv
import os

from src.models.s4.s4 import S4
from src.models.s4.s4d import S4D

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# Optimizer
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
parser.add_argument('--file_name', default='S4D_1Lead', type=str, help='Folder Name')
# Scheduler
parser.add_argument('--epochs', default=200, type=int, help='Training epochs')
# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
# Model
parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


output_directory = './s4_results/' + args.file_name
output_filename = 'argparse_config.txt'

if not os.path.exists(output_directory):
    # If it doesn't exist, create the directory
    os.makedirs(output_directory)
    print(f"Directory '{output_directory}' created successfully.")
else:
    print(f"Directory '{output_directory}' already exists.")

output_filepath = f'{output_directory}/{output_filename}'

d_input = 1
d_output = 8

class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, args.lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return nn.functional.sigmoid(x)

# Model
print('==> Building model..')
model = S4Model(
    d_input=d_input,
    d_output=d_output,
    d_model=args.d_model,
    n_layers=args.n_layers,
    dropout=args.dropout,
    prenorm=args.prenorm,
)

# Open the HDF5 file
with h5py.File('x.hdf5', 'r') as f:
    # Load the ECG data
    y_true = pd.read_csv('y.csv')
    ecg_data = f['tracings'][-500:, :, 1].reshape(500, 4096, 1)

# Convert the numpy array to a PyTorch tensor
input_data = torch.from_numpy(ecg_data).float()

# Load the saved model from .pt file
state_dict = torch.load(output_directory + '/model.pt') #

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

batch_size = args.batch_size
num_batches = len(input_data) // batch_size

file_path = output_directory + '/predict.csv'

# Check if the file exists
if os.path.exists(file_path):
    # If the file exists, remove it
    os.remove(file_path)

# open the CSV file in append mode
with open(file_path, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size
        batch_input = input_data[start_index:end_index, :, :]
        batch_output = model(batch_input)

        # write the output from this batch to the CSV file
        writer.writerows(batch_output.tolist())
        print(i)

    # process the final partial batch, if there is one
    if len(input_data) % batch_size != 0:
        start_index = num_batches * batch_size
        end_index = len(input_data)
        partial_batch_input = input_data[start_index:end_index, :, :]
        partial_batch_output = model(partial_batch_input)

        # write the output from the partial batch to the CSV file
        writer.writerows(partial_batch_output.tolist())

y_pred =  pd.read_csv(file_path, header=None).values

# Process the output as needed
print(input_data.shape, y_pred.shape, type(y_pred))

header = y_true.columns.to_numpy().tolist()
print(type(header), header)
y_true = y_true.values[-500:,...]
print(type(y_true), y_true.shape)

# assume y_pred and y_true are NumPy arrays with shape [N, 8]
y_pred_bin = (y_pred > 0.5).astype(int)  # binarize the predictions
precision = precision_score(y_true, y_pred_bin, average=None)
recall = recall_score(y_true, y_pred_bin, average=None)
f1 = f1_score(y_true, y_pred_bin, average=None)
auroc_scores = roc_auc_score(y_true, y_pred, average=None)
auprc_scores = []
pr_curves = []
metrics_list = []


for i in range(len(header)):
    tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred_bin[:, i]).ravel()
    specificity = tn / (tn + fp)
    precision_i, recall_i, thresholds_i = precision_recall_curve(y_true[:, i], y_pred[:, i])
    auprc_i = auc(recall_i, precision_i)
    auprc_scores.append(auprc_i)
    pr_curves.append((precision_i, recall_i))
    accuracy_i = accuracy_score(y_true[:, i], y_pred_bin[:, i], normalize=True)
    metrics_list.append([header[i], precision[i], recall[i], f1[i], specificity, auroc_scores[i], auprc_i, accuracy_i])

# save metrics_list as csv
with open(output_directory + '/evaluation_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Class', 'Precision', 'Recall', 'F1-score', 'Specificity', 'AUROC', 'AUPRC', 'Accuracy'])
    writer.writerows(metrics_list)


# Concatenate all true labels and predicted probabilities for all classes
y_true_all = y_true.ravel()
y_pred_all = y_pred.ravel()

precision_m = precision_score(y_true, y_pred_bin, average='weighted')
recall_m = recall_score(y_true, y_pred_bin, average='weighted')
f1_m = f1_score(y_true, y_pred_bin, average='weighted')
overall_auroc = roc_auc_score(y_true_all, y_pred_all, average=None)


# Calculate precision and recall values at different probability thresholds
precision_all, recall_all, thresholds_all = precision_recall_curve(y_true_all, y_pred_all)
# Calculate overall AUPRC
overall_auprc = auc(recall_all, precision_all) # overall PRC and AUPRC

accuracy = accuracy_score(y_true_all, y_pred_bin.ravel(), normalize=True)


print("Overall metrics - Precision:", precision_m, " Recall:", recall_m, " F1-score:", f1_m, " AUROC:", overall_auroc, " AUPRC:", overall_auprc, "Accuracy", accuracy)