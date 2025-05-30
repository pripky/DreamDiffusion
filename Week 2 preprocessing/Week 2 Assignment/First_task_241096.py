import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

# ----------- STEP 1: Load EEG Data -------------
file_path = r"C:\Users\tarun goyal\Desktop\MWData\MW.txt"
raw_df = pd.read_csv(file_path, delimiter="\t")

print("Preview of loaded EEG data:")
print(raw_df.head())

# ----------- STEP 2: Extract EEG Channels and Trigger -------------
eeg_signals = raw_df.iloc[:, 1:-1].apply(pd.to_numeric, errors='coerce').values
stimulus = raw_df.iloc[:, -1].values

# ----------- STEP 3: Apply Bandpass Filter -------------
def apply_bandpass(data, low, high, fs, order=4):
    nyquist = 0.5 * fs
    low /= nyquist
    high /= nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

sampling_rate = 250  # in Hz
filtered_eeg = apply_bandpass(eeg_signals, 0.1, 30.0, sampling_rate)

print("Filtered first 10 samples of first channel:")
print(filtered_eeg[:10, 0])

# ----------- STEP 4: Find Stimulus Onsets -------------
stim_onsets = [i for i, val in enumerate(stimulus) if val != 0]
print(f"Detected {len(stim_onsets)} stimulus events.")

# ----------- STEP 5: Create Epochs -------------
pre_samples = int(0.2 * sampling_rate)
post_samples = int(0.8 * sampling_rate)

epochs_list = []
for event in stim_onsets:
    start = event - pre_samples
    end = event + post_samples
    if start >= 0 and end < len(filtered_eeg):
        epochs_list.append(filtered_eeg[start:end])

epochs_array = np.array(epochs_list)
print("Shape of extracted epochs:", epochs_array.shape)

# ----------- STEP 6: Resample to Uniform Length -------------
epoch_length = int(np.median([ep.shape[0] for ep in epochs_array]))

def uniform_resample(epoch, target_length):
    original_time = np.linspace(0, 1, epoch.shape[0])
    target_time = np.linspace(0, 1, target_length)
    return np.stack([interp1d(original_time, ch)(target_time) for ch in epoch.T], axis=1)

resampled_epochs = np.array([uniform_resample(ep, epoch_length) for ep in epochs_array])

# ----------- STEP 7: Fill Missing Values (NaNs) -------------
def fill_missing(epoch):
    for channel in range(epoch.shape[1]):
        y = epoch[:, channel]
        if np.isnan(y).all():
            continue
        x = np.arange(len(y))
        y[np.isnan(y)] = np.interp(x[np.isnan(y)], x[~np.isnan(y)], y[~np.isnan(y)])
        epoch[:, channel] = y
    return epoch

interpolated_epochs = np.array([fill_missing(ep) for ep in resampled_epochs])

# ----------- STEP 8: Normalize Epochs -------------
def minmax_scale(epoch):
    return MinMaxScaler().fit_transform(epoch)

normalized_epochs = np.array([minmax_scale(ep) for ep in interpolated_epochs])
print("Final data shape (epochs, time, channels):", normalized_epochs.shape)

# ----------- STEP 9: Plot Example Epoch -------------
plt.figure(figsize=(10, 4))
plt.plot(normalized_epochs[0])
plt.title("First Normalized EEG Epoch")
plt.xlabel("Time Points")
plt.ylabel("Amplitude (Normalized)")
plt.grid(True)
plt.show()

# ----------- STEP 10: Print Sample Values -------------
print("First 10 samples of first channel in first epoch (normalized):")
print(normalized_epochs[0][:10, 0])
