# Model Benchmarks & Comparison

Scripts and logs comparing different neural architectures on price series forecasting.

### Overview
- Evaluates 20+ model candidates (Dense, Conv1D, LSTM, GRU, BiGRU, TCN, and hybrid ensembles).
- Analyzes loss functions (MSE, MAE, Huber) and regularization impact (Dropout, L2).
- Serves as the experimental foundation leading to the Attention-TCN-BiGRU (ATG) architecture.