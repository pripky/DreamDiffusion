# Upload and unzip files
from google.colab import files
uploaded_files = files.upload()

import zipfile, os

zip_file = 'data.zip'
with zipfile.ZipFile(zip_file, 'r') as archive:
    archive.extractall('data_unzipped')

os.listdir('data_unzipped')

!unzip data_unzipped/data/MuseData.zip
!unzip data_unzipped/data/MWData.zip

# Locate MU.txt file
for dir_path, subdirs, file_list in os.walk('/content'):
    for filename in file_list:
        if filename == 'MU.txt':
            print(os.path.join(dir_path, filename))

# Read EEG data files
import pandas as pd

mu_df = pd.read_csv("/content/MU.txt", delimiter="\t")
print(mu_df.head())

mw_df = pd.read_csv("/content/MWData/MW.txt", delimiter="\t")
print(mw_df.head())

# Extract EEG and triggers
mu_signals = mu_df.iloc[:, 1:-1].apply(pd.to_numeric, errors='coerce').values
mu_triggers = mu_df.iloc[:, -1].values

mw_signals = mw_df.iloc[:, 1:-1].apply(pd.to_numeric, errors='coerce').values
mw_triggers = mw_df.iloc[:, -1].values

# Bandpass filter
from scipy.signal import butter, filtfilt

def apply_bandpass(data, low, high, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    low_norm = low / nyquist
    high_norm = high / nyquist
    b, a = butter(order, [low_norm, high_norm], btype='band')
    return filtfilt(b, a, data, axis=0)

sampling_rate = 250
mu_filtered = apply_bandpass(mu_signals, 0.1, 30.0, sampling_rate)
mw_filtered = apply_bandpass(mw_signals, 0.1, 30.0, sampling_rate)

# Detect events
mu_events = [idx for idx, val in enumerate(mu_triggers) if val != 0]
print("Found", len(mu_events), "stimulus events for Muse")

mw_events = [idx for idx, val in enumerate(mw_triggers) if val != 0]
print("Found", len(mw_events), "stimulus events for MW")

# Extract epochs
import numpy as np

before = int(0.2 * sampling_rate)
after = int(0.8 * sampling_rate)

def extract_epochs(filtered_data, event_indices):
    epochs = []
    for i in event_indices:
        if i - before >= 0 and i + after < len(filtered_data):
            epochs.append(filtered_data[i - before:i + after])
    return np.array(epochs)

mu_epochs = extract_epochs(mu_filtered, mu_events)
print("Epochs shape (Muse):", mu_epochs.shape)

mw_epochs = extract_epochs(mw_filtered, mw_events)
print("Epochs shape (MW):", mw_epochs.shape)

# Interpolation and resampling
from scipy.interpolate import interp1d

def resample(e, length):
    original_x = np.linspace(0, 1, e.shape[0])
    new_x = np.linspace(0, 1, length)
    return np.array([interp1d(original_x, channel)(new_x) for channel in e.T]).T

def fill_nans(e):
    for ch in range(e.shape[1]):
        x = np.arange(e.shape[0])
        y = e[:, ch]
        if np.all(np.isnan(y)):
            continue
        y[np.isnan(y)] = np.interp(x[np.isnan(y)], x[~np.isnan(y)], y[~np.isnan(y)])
        e[:, ch] = y
    return e

def preprocess_epochs(epochs):
    target_length = int(np.median([ep.shape[0] for ep in epochs]))
    resampled = np.array([resample(e, target_length) for e in epochs])
    interpolated = np.array([fill_nans(e) for e in resampled])
    return interpolated

mu_clean = preprocess_epochs(mu_epochs)
mw_clean = preprocess_epochs(mw_epochs)

print("Cleaned Muse Epochs shape:", mu_clean.shape)
print("Cleaned MW Epochs shape:", mw_clean.shape)

# Normalize epochs
from sklearn.preprocessing import MinMaxScaler

def normalize(e):
    return MinMaxScaler().fit_transform(e)

mu_norm = np.array([normalize(e) for e in mu_clean])
mw_norm = np.array([normalize(e) for e in mw_clean])

print("Normalized Muse shape:", mu_norm.shape)
print("Normalized MW shape:", mw_norm.shape)
