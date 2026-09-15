# Experimental Notebooks

Chronological research pipeline for the BullEye AI project.

### Notebooks Overview

- **`baseline.ipynb`** : Initial end-to-end regression model pipeline on asset price series.
- **`model_selection_benchmark.ipynb`** : Exhaustive benchmark comparing 20 neural architectures (MLP, Conv1D, LSTM, GRU, BiGRU, TCN, and hybrids).
- **`horizon_robustness_test.ipynb`** : Multi-step prediction testing ($H \in \{1, 3, 7, 10, 14\}$) uncovering the moving-average lag effect.
- **`classification.ipynb`** : Strategic pivot from price regression to Triple-Barrier directional classification (`BUY`, `HOLD`, `SELL`).
- **`classification_v2.ipynb`** : Refined classification pipeline with dynamic volatility-adjusted thresholds ($\sigma_{20\text{d}} \times \sqrt{H} \times \lambda$) and 17 stationary features.
- **`test_svm_model.ipynb`** : Classical machine learning baseline evaluation using Support Vector Machines.
- **`final_evaluation.ipynb`** : Out-of-sample backtesting, confidence threshold analysis ($\tau > 0.55$), and cross-asset generalization.