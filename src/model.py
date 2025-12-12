# src/model.py
import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

def predict_mc_dropout(model, x, n_samples=50):
    model.train()
    preds = []
    for _ in range(n_samples):
        preds.append(model(x).detach().cpu().numpy())
    preds = np.array(preds)
    mean = preds.mean(axis=0)
    lower = np.percentile(preds, 5, axis=0)
    upper = np.percentile(preds, 95, axis=0)
    return mean, lower, upper
