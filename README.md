# BullEye AI — Augmenting DCA with Deep Learning

Research project conducted as part of the **Défi G1G2** academic track at **École Centrale de Lille** (2025–2026).

This repository investigates whether Deep Learning can augment a traditional Dollar-Cost Averaging (DCA) strategy by timing market entries on equities and crypto (SPY, QQQ, BTC). It documents the full quantitative pipeline: mathematical foundations, a benchmark of 20 architectures, bias identification, and a transition to dynamic volatility classification.

📄 **Full 58-page thesis available in the repo:** [`final_report_french.pdf`](./final_report_french.pdf) *(in French)*.

---

## ⚡ Key Findings

1. **Theoretical DCA Invariance:** The relative timing outperformance ($\eta$) is mathematically independent of the invested capital ($M$). Alpha depends strictly on price oscillation density and is asymptotically diluted over time by accumulated portfolio size.
2. **The Multi-Step Lag Fallacy:** Moving from $H=1$ to $H=14$ days causes regression models to collapse into a naive lagging indicator ($y_{t+H} \approx y_t$). Without live exogenous data, historical price series alone cannot forecast medium-term turns.
3. **Dynamic Triple-Barrier Pivot:** Reformulating the objective into 3 volatility-adjusted regimes (`BUY`, `HOLD`, `SELL`) with 17 stationary features (momentum, oscillators, intraday candle metrics, VIX, TNX, Fear & Greed).
4. **Calibrated Inertia (ATG):** The hybrid **Attention + TCN + BiGRU** network delivered the best probability calibration. Under strict confidence filtering ($\tau > 0.55$), the model achieves **83.3% precision on the `HOLD` class**, acting as an efficient market noise filter.

---

## 📂 Repository Structure

* **[`notebooks/`](./notebooks/)** — Complete experimental pipeline:
  * `baseline.ipynb` — Initial regression baseline.
  * `model_selection_benchmark.ipynb` — 20-architecture benchmark on price regression.
  * `horizon_robustness_test.ipynb` — Multi-step degradation ($H \in \{1, 3, 7, 10, 14\}$).
  * `classification.ipynb` — Initial Triple-Barrier labeling and direction prediction.
  * `classification_v2.ipynb` — Dynamic volatility barrier tuning ($\lambda = 1.0, H = 3$).
  * `test_svm_model.ipynb` — Baseline classification using Support Vector Machines.
  * `final_evaluation.ipynb` — High-confidence filtering and cross-asset backtests.
* **[`augmented_dca_theory/`](./augmented_dca_theory/)** — Mathematical derivations and numerical proofs of the timing alpha.
* **[`models_benchmarks/`](./models_benchmarks/)** — Evaluation logs and comparative scripts for candidate models.
* **[`macro_analysis/`](./macro_analysis/)** — M2 money supply historical data and trend scripts.
* **[`notes_french/`](./notes_french/)** — Literature takeaways and research notes (French).
* **[`streamlit_old/`](./streamlit_old/)** — Deprecated interactive prototype UI (early development).
* **[`final_report_french.pdf`](./final_report_french.pdf)** — Comprehensive reference thesis detailing all equations and empirical findings.

---

## 🛠️ Tech Stack

* **Language & Frameworks:** Python, TensorFlow / Keras, Scikit-Learn
* **Data Sources:** `yfinance` (SPY, QQQ, BTC-USD, VIX, TNX), CNN Fear & Greed Index
* **Core Architectures:** Dilated Causal TCN, Bidirectional GRU, Self-Attention

---

## 👤 Author

**Jean Frégeville** — École Centrale de Lille