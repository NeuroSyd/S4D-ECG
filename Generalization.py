import h5py
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc, accuracy_score
from torch.utils.data import DataLoader
import numpy as np
import csv
from scipy.signal import butter, filtfilt, resample
import pywt
from scipy import signal
import os

from src.models.s4.s4 import S4
from src.models.s4.s4d import S4D

os.environ['CUDA_VISIBLE_DEVICES']= '1'

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

d_input = 12
d_output = 8
n = 10

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
# Load the saved model from .pt file
state_dict = torch.load('./s4_results/model.pt') #

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Filters
def apply_bandpass_filter(ecg_data, fs=500, lowcut=0.5, highcut=40, order=4):
    """
    Applies a bandpass filter to each lead in ECG data.

    Args:
    ecg_data (numpy.ndarray): numpy array of shape [N, 4096, 12], where N is the number of ECG recordings
    fs (float): Sampling frequency in Hz (default: 500 Hz)
    lowcut (float): Lower cutoff frequency in Hz (default: 0.5 Hz)
    highcut (float): Upper cutoff frequency in Hz (default: 40 Hz)
    order (int): Filter order (default: 4)

    Returns:
    numpy.ndarray: a numpy array of shape [N, 4096, 12], containing the denoised ECG data
    """
    nyq = 0.5*fs
    lowcut = lowcut/nyq
    highcut = highcut/nyq

    # Create an empty array to store the denoised ECG data
    denoised_ecg_data = np.zeros_like(ecg_data)

    # Loop through each lead in the ECG data
    for i in range(ecg_data.shape[0]):
        for j in range(ecg_data.shape[2]):
            # Extract the ECG data for the current lead
            lead_data = ecg_data[i, :, j]

            # Design the bandpass filter
            b, a = butter(order,[lowcut,highcut], btype='band')

            # Apply the bandpass filter to the lead data
            denoised_lead_data = filtfilt(b, a, lead_data)

            # Store the denoised lead data in the denoised ECG data array
            denoised_ecg_data[i, :, j] = denoised_lead_data

    return denoised_ecg_data

def filter_ecg_signal(data, wavelet='db4', level=8, fs=500, fc=0.1, order=6):
    """
    Filter ECG signals using wavelet denoising.

    Args:
        data (numpy array): ECG signal data with shape (n_samples, n_samples_per_lead, n_leads).
        wavelet (str, optional): Wavelet type for denoising. Default is 'db4'.
        level (int, optional): Decomposition level for wavelet denoising. Default is 8.
        fs (float, optional): Sampling frequency of ECG signals. Default is 500 Hz.
        fc (float, optional): Cutoff frequency for lowpass filter. Default is 0.1 Hz.
        order (int, optional): Filter order for Butterworth filter. Default is 6.

    Returns:
        numpy array: Filtered ECG signals.
    """
    nyquist = 0.5 * fs
    cutoff = fc / nyquist
    b, a = signal.butter(order, cutoff, btype='lowpass')

    filtered_signals = np.zeros_like(data)

    for n in range(data.shape[0]):
        for i in range(data.shape[2]):
            ecg_signal = data[n, :, i]
            coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
            cA = coeffs[0]
            filtered_cA = signal.filtfilt(b, a, cA)
            filtered_coeffs = [filtered_cA] + coeffs[1:]
            filtered_signal = pywt.waverec(filtered_coeffs, wavelet)
            filtered_signals[n, :, i] = filtered_signal

    return filtered_signals

# resampling ECG data
def resample_ecg_data(ecg_data, origianl_rate, target_rate, samples):
    """
    Resamples ECG data from 400 Hz to 500 Hz.

    Args:
        ecg_data (np.ndarray): ECG data with shape [N, 4096, 12].

    Returns:
        np.ndarray: Resampled ECG data with shape [N, M, 12], where M is the new number of samples after resampling.
    """
    # Compute the resampling ratio
    resampling_ratio = target_rate / origianl_rate

    # Compute the new number of samples after resampling
    M = int(ecg_data.shape[1] * resampling_ratio)

    # Initialize an array to store the resampled data
    ecg_data_resampled = np.zeros((ecg_data.shape[0], M, ecg_data.shape[2]))

    # Iterate over each channel and resample independently
    for i in range(ecg_data.shape[2]):
        for j in range(ecg_data.shape[0]):
            ecg_data_resampled[j, :, i] = resample(ecg_data[j, :, i], M)
    # Trim the resampled data to the last 4096 samples
    ecg_data_resampled = ecg_data_resampled[:, -samples:, :]
    return ecg_data_resampled


def set_channels_to_zero(ecg_data, n):
    """
    Randomly selects a number of ECG channels to set to zero.

    Args:
    - ecg_data: numpy array of shape (N, 4096, 12) containing ECG data
    - n: maximum number of channels that can be set to zero (up to n-1 channels can be left non-zero)

    Returns:
    - numpy array of shape (N, 4096, 12) with selected channels set to zero
    """

    # Choose number of channels to set to zero (up to n-1)
    num_channels_to_set_zero = n

    # Choose which channels to set to zero
    channels_to_set_zero = np.random.choice(ecg_data.shape[-1], num_channels_to_set_zero, replace=False)

    # Set selected channels to zero
    ecg_data[:, :, channels_to_set_zero] = 0

    return ecg_data


# Making prediction csv
# Open the HDF5 file
with h5py.File('x.hdf5', 'r') as f:
    # Get the HDF5 dataset object
    dataset_names = list(f.keys())
    ecg_dataset = f[dataset_names[1]]

    # Define batch size and total number of samples
    batch_size = args.batch_size  # Set your desired batch size here
    total_samples = ecg_dataset.shape[0]

    # Specify the file path
    file_path = './s4_results/predict_blankout_'+ str(n) +'.csv'

    # Check if the file exists
    if os.path.exists(file_path):
        # If the file exists, remove it
        os.remove(file_path)

    # Loop to read data in batches
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_data = ecg_dataset[start_idx:end_idx]  # Read batch of data
        batch_data = apply_bandpass_filter(batch_data)
        batch_data = filter_ecg_signal(batch_data)
        batch_data = resample_ecg_data(batch_data, 400, 500, 4096)
        batch_data = set_channels_to_zero(batch_data, n)

        # Convert the numpy array to a PyTorch tensor
        input_data = torch.from_numpy(batch_data).float()

        # Process batch_data as needed
        batch_output = model(input_data)

        # Open the CSV file in append mode
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the output from this batch to the CSV file
            writer.writerows(batch_output.tolist())

        # Clear batch_data and input_data from memory to free up space
        del batch_data
        del input_data

        # Print progress or do other operations
        print(f"Processed {end_idx} samples out of {total_samples}")

column_names = ['AF','1dAVb','LBBB','RBBB','PAC','PVC','STD','STE']
y_pred =  pd.read_csv(file_path, names=column_names)

# Process the output as needed
print('y_pred', len(y_pred), type(y_pred))

y_true = pd.read_csv('y.csv')
print('y_true', len(y_true), type(y_true))

# Find the overlapping columns
common_columns = np.intersect1d(y_pred.columns, y_true.columns)
print(common_columns)

# Select the overlapping columns in both DataFrames
y_pred = y_pred[common_columns].values
y_true = y_true[common_columns].values

print(y_pred.shape, y_true.shape)


# assume y_pred and y_true are NumPy arrays with shape [N, 8]
y_pred_bin = (y_pred > 0.5).astype(int)  # binarize the predictions
precision = precision_score(y_true, y_pred_bin, average=None)
recall = recall_score(y_true, y_pred_bin, average=None)
f1 = f1_score(y_true, y_pred_bin, average=None)
auroc_scores = roc_auc_score(y_true, y_pred, average=None)
auprc_scores = []
pr_curves = []
metrics_list = []

for i in range(len(common_columns)):
    tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred_bin[:, i]).ravel()
    specificity = tn / (tn + fp)
    precision_i, recall_i, thresholds_i = precision_recall_curve(y_true[:, i], y_pred[:, i])
    auprc_i = auc(recall_i, precision_i)
    auprc_scores.append(auprc_i)
    pr_curves.append((precision_i, recall_i))
    accuracy_i = accuracy_score(y_true[:, i], y_pred_bin[:, i], normalize=True)
    metrics_list.append([common_columns[i], precision[i], recall[i], f1[i], specificity, auroc_scores[i], auprc_i, accuracy_i])

# save metrics_list as csv
with open('./s4_results/metrics_blankout_'+ str(n) +'.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Class', 'Precision', 'Recall', 'F1-score', 'Specificity', 'AUROC', 'AUPRC', 'Accuracy'])
    writer.writerows(metrics_list)

precision_m = precision_score(y_true, y_pred_bin, average='weighted')
recall_m = recall_score(y_true, y_pred_bin, average='weighted')
f1_m = f1_score(y_true, y_pred_bin, average='weighted')
overall_auroc = roc_auc_score(y_true, y_pred, average='weighted')

# Concatenate all true labels and predicted probabilities for all classes
y_true_all = y_true.ravel()
y_pred_all = y_pred.ravel()
# Calculate precision and recall values at different probability thresholds
precision_all, recall_all, thresholds_all = precision_recall_curve(y_true_all, y_pred_all)
# Calculate overall AUPRC
overall_auprc = auc(recall_all, precision_all) # overall PRC and AUPRC

accuracy = accuracy_score(y_true_all, y_pred_bin.ravel(), normalize=True)


print("Overall metrics - Precision:", precision_m, " Recall:", recall_m, " F1-score:", f1_m, " AUROC:", overall_auroc, " AUPRC:", overall_auprc, "Accuracy", accuracy)