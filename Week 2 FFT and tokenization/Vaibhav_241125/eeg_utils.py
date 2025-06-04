
import numpy as np

def pad_to_128_channels(eeg_epoch):
    """
    Pads EEG epoch to 128 channels by repeating channels if needed.
    Input shape: (channels, time_samples)
    Output shape: (128, time_samples)
    """
    current_channels = eeg_epoch.shape[0]
    if current_channels >= 128:
        return eeg_epoch[:128, :]  
    else:
        repeat_count = 128 - current_channels
        extra_channels = eeg_epoch[:repeat_count, :]
        padded_epoch = np.vstack([eeg_epoch, extra_channels])
        return padded_epoch
