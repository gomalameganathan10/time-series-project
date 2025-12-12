# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from data import generate_synthetic_series
from preprocess import scale_series, create_dataset
from model import LSTMModel

# Hyperparams
input_size = 1
hidden_size = 256
num_layers = 1
dropout = 0.2
lr = 1e-3
epochs = 30
window_size = 20

df = generate_synthetic_series()
scaled, _ = scale_series(df['value'].values)
X, y = create_dataset(scaled, window_size=window_size)

X_tensor = torch.tensor(X, dtype=torch.float32)       # shape: (N, seq_len, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(1, epochs+1):
    model.train()
    optimizer.zero_grad()
    out = model(X_tensor)
    loss = criterion(out, y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}/{epochs}, loss: {loss.item():.5f}")

torch.save(model.state_dict(), "lstm_model.pth")
print("Saved model as lstm_model.pth")
