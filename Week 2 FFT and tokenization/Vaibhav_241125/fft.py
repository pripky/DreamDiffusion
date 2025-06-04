
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from eeg_utils import pad_to_128_channels
from temporal_tokenizer import TemporalTokenizer

tokenizer = TemporalTokenizer()



pth_path = 'eeg_signals_raw_with_mean_std.pth'
loaded_data = torch.load(pth_path)
dataset = loaded_data['dataset']



def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sos

def apply_filter(data, sos):
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch, :] = sosfiltfilt(sos, data[ch, :])
    return filtered



def segment_into_epochs(eeg_data, epoch_length_samples=100):
    total_samples = eeg_data.shape[1]
    epochs = []
    start = 0
    while start + epoch_length_samples <= total_samples:
        epoch = eeg_data[:, start:start + epoch_length_samples]
        epochs.append(epoch)
        start += epoch_length_samples
    return epochs



def compute_fft_features(epoch):
    """
    epoch: shape (128, 100) → returns (128, 50) [real magnitudes]
    """
    fft_complex = np.fft.fft(epoch, axis=1)
    fft_magnitude = np.abs(fft_complex[:, :epoch.shape[1] // 2])
    return fft_magnitude



fs = 1000
lowcut = 5
highcut = 95
sos = butter_bandpass(lowcut, highcut, fs)



processed_dataset = []

for sample_idx, sample in enumerate(dataset):
    eeg = sample['eeg'].numpy()

    eeg = eeg[:, 20:]
    eeg = eeg[:, :440] if eeg.shape[1] >= 440 else np.pad(eeg, ((0, 0), (0, 440 - eeg.shape[1])), mode='constant')

    eeg_filtered = apply_filter(eeg, sos)
    epochs = segment_into_epochs(eeg_filtered, epoch_length_samples=100)

    for i, epoch in enumerate(epochs):
        padded_epoch = pad_to_128_channels(epoch)

        mean = np.mean(padded_epoch, axis=1, keepdims=True)
        std = np.std(padded_epoch, axis=1, keepdims=True) + 1e-8
        normalized_epoch = (padded_epoch - mean) / std

        embedded_tokens = tokenizer.tokenize(normalized_epoch) 


        fft_features = compute_fft_features(normalized_epoch) 


        if sample_idx == 0 and i == 0:
            plt.figure(figsize=(10, 5))
            plt.imshow(fft_features, aspect='auto', origin='lower', cmap='magma')
            plt.colorbar(label='Magnitude')
            plt.xlabel('Frequency bin')
            plt.ylabel('Channel')
            plt.title('Spectrogram (Sample 0, Epoch 0)')
            plt.tight_layout()
            plt.savefig("spectrogram_sample0_epoch0.png")
            plt.close()

        processed_dataset.append({
            'eeg_tokens': embedded_tokens,         
            'fft': torch.tensor(fft_features, dtype=torch.float32), 
            'label': sample['label'],
            'subject': sample['subject'],
            'image': sample['image'],
            'epoch_idx': i
        })

torch.save({'dataset': processed_dataset}, 'eeg_tokenized_dataset_with_fft.pth')
print(f"Saved {len(processed_dataset)} epochs with tokens and FFT to 'eeg_tokenized_dataset_with_fft.pth'")
