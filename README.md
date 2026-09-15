# BullEye AI — Augmenting DCA with Deep Learning

Research project conducted as part of the **Défi G1G2** academic track at **École Centrale de Lille** (2025–2026)[cite: 1].

This repository investigates whether Deep Learning can augment a traditional Dollar-Cost Averaging (DCA) strategy by timing market entries on equities and crypto (SPY, QQQ, BTC)[cite: 1]. It documents the full quantitative pipeline: mathematical foundations, a benchmark of 20 architectures, bias identification, and a transition to dynamic volatility classification[cite: 1].

📄 **Full 58-page thesis available in the repo:** [`final_report_french.pdf`](./final_report_french.pdf) *(in French)*[cite: 1].

---

## ⚡ Key Findings

1. **Theoretical DCA Invariance:** The relative timing outperformance ($\eta$) is mathematically independent of the invested capital ($M$)[cite: 1]. Alpha depends strictly on price oscillation density and is asymptotically diluted over time by accumulated portfolio size[cite: 1].
2. **The Multi-Step Lag Fallacy:** Moving from $H=1$ to $H=14$ days causes regression models to collapse into a naive lagging indicator ($y_{t+H} \approx y_t$)[cite: 1]. Without live exogenous data, historical price series alone cannot forecast medium-term turns[cite: 1].
3. **Dynamic Triple-Barrier Pivot:** Reformulating the objective into 3 volatility-adjusted regimes (`BUY`, `HOLD`, `SELL`) with 17 stationary features (momentum, oscillators, intraday candle metrics, VIX, TNX, Fear & Greed)[cite: 1].
4. **Calibrated Inertia (ATG):** The hybrid **Attention + TCN + BiGRU** network delivered the best probability calibration[cite: 1]. Under strict confidence filtering ($\tau > 0.55$), the model achieves **83.3% precision on the `HOLD` class**, acting as an efficient market noise filter[cite: 1].

---

## 📂 Repository Structure

* **[`notebooks/`](./notebooks/)** — Complete experimental pipeline:
  * `baseline.ipynb` — Initial regression baseline[cite: 1].
  * `model_selection_benchmark.ipynb` — 20-architecture benchmark on price regression[cite: 1].
  * `horizon_robustness_test.ipynb` — Multi-step degradation ($H \in \{1, 3, 7, 10, 14\}$)[cite: 1].
  * `classification.ipynb` — Initial Triple-Barrier labeling and direction prediction[cite: 1].
  * `classification_v2.ipynb` — Dynamic volatility barrier tuning ($\lambda = 1.0, H = 3$)[cite: 1].
  * `test_svm_model.ipynb` — Baseline classification using Support Vector Machines.
  * `final_evaluation.ipynb` — High-confidence filtering and cross-asset backtests[cite: 1].
* **[`augmented_dca_theory/`](./augmented_dca_theory/)** — Mathematical derivations and numerical proofs of the timing alpha[cite: 1].
* **[`models_benchmarks/`](./models_benchmarks/)** — Evaluation logs and comparative scripts for candidate models[cite: 1].
* **[`macro_analysis/`](./macro_analysis/)** — M2 money supply historical data and trend scripts[cite: 1].
* **[`notes_french/`](./notes_french/)** — Literature takeaways and research notes (French)[cite: 1].
* **[`streamlit_old/`](./streamlit_old/)** — Deprecated interactive prototype UI (early development).
* **[`final_report_french.pdf`](./final_report_french.pdf)** — Comprehensive reference thesis detailing all equations and empirical findings[cite: 1].

---

## 🛠️ Tech Stack

* **Language & Frameworks:** Python, TensorFlow / Keras, Scikit-Learn
* **Data Sources:** `yfinance` (SPY, QQQ, BTC-USD, VIX, TNX), CNN Fear & Greed Index[cite: 1]
* **Core Architectures:** Dilated Causal TCN, Bidirectional GRU, Self-Attention[cite: 1]

---

## 👤 Author

**Jean Frégeville** — École Centrale de Lille[cite: 1]