# src/evaluate.py
import torch
import matplotlib.pyplot as plt
from data import generate_synthetic_series
from preprocess import scale_series, create_dataset
from model import LSTMModel, predict_mc_dropout

df = generate_synthetic_series()
scaled, scaler = scale_series(df['value'].values)
X, y = create_dataset(scaled, window_size=20)
X_tensor = torch.tensor(X, dtype=torch.float32)

model = LSTMModel(input_size=1, hidden_size=256, num_layers=1, dropout=0.2)
model.load_state_dict(torch.load("lstm_model.pth", map_location="cpu"))
model.eval()

mean, lower, upper = predict_mc_dropout(model, X_tensor, n_samples=100)

plt.figure(figsize=(10,5))
plt.plot(df['value'].values[20:], label='True')
plt.plot(scaler.inverse_transform(mean), label='Predicted')
plt.fill_between(range(len(mean)),
                 scaler.inverse_transform(lower).flatten(),
                 scaler.inverse_transform(upper).flatten(), alpha=0.3)
plt.legend()
plt.savefig("../results/eval_plot.png")
print("Saved plot to results/eval_plot.png")
