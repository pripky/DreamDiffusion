import torch
import torch.nn as nn
import numpy as np

class TemporalTokenizer:
    def __init__(self, input_channels=128, time_window=4, embed_dim=1024):
        self.time_window = time_window
        self.input_channels = input_channels
        self.embed_dim = embed_dim
 
        self.linear_proj = nn.Linear(input_channels * time_window, embed_dim)

    def tokenize(self, eeg_epoch):
        """
        eeg_epoch: numpy array of shape (channels, time_samples)
        Returns: torch.Tensor of shape (num_tokens, embed_dim)
        """
        if eeg_epoch.shape[0] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {eeg_epoch.shape[0]}")

        time_samples = eeg_epoch.shape[1]
        if time_samples % self.time_window != 0:
            raise ValueError(f"Time samples ({time_samples}) not divisible by time_window ({self.time_window})")

        num_tokens = time_samples // self.time_window
        reshaped = eeg_epoch.reshape(self.input_channels, num_tokens, self.time_window) 

        reshaped = reshaped.transpose(1, 0, 2)  

        tokens = reshaped.reshape(num_tokens, -1) 

        tokens_tensor = torch.tensor(tokens, dtype=torch.float32)
        embedded = self.linear_proj(tokens_tensor)  

        return embedded
