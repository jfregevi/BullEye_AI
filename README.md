# BullEye AI — Augmenting DCA with Deep Learning

Research project conducted as part of the **Défi G1G2** academic track at **École Centrale de Lille** (2025–2026).

This repository investigates whether Deep Learning can augment a traditional Dollar-Cost Averaging (DCA) strategy by timing market entries on equities and crypto (SPY, QQQ, BTC). It documents the full quantitative journey: mathematical foundations, a benchmark of 20 architectures, the discovery of predictive limitations, and a pragmatic transition to volatility-based classification.

📄 **Full 58-page thesis available in the repo:** [`final_report_french.pdf`](./final_report_french.pdf) *(in French)*.

---

## ⚡ Key Takeaways & Lessons Learned

1. **The DCA Augmented Theory:** Mathematically, an "AI-boosted DCA" only significantly outperforms in highly volatile or ranging markets. In a strong, steady bull market, the AI simply mimics a standard DCA strategy because there are no price drops to exploit.
2. **The Illusion of Regression:** Predicting prices 1 day ahead ($H=1$) looks incredibly accurate, but it's an illusion. When extending the horizon ($H=7$ or $H=14$), the model stops anticipating and simply becomes a lagging indicator, reacting to the market rather than predicting it.
3. **Pivot to Classification:** To counter this, the problem was reframed into predicting market regimes (`BUY`, `HOLD`, `SELL`) using a dynamic barrier adjusted to the asset's 20-day rolling volatility.
4. **Final Verdict & Reality Check:** The best-performing model (an ensemble of Attention, TCN, and BiGRU) proved very effective at predicting market inertia (83.3% precision on the `HOLD` class) but struggled to reliably catch major reversals. Ultimately, the project highlighted that pure historical price data is insufficient to consistently beat the market, and traditional DCA remains an incredibly robust strategy.

---

## 📂 Repository Structure

* **[`notebooks/`](./notebooks/)** — Complete experimental pipeline:
  * `baseline.ipynb` — Initial regression baseline.
  * `model_selection_benchmark.ipynb` — 20-architecture benchmark on price regression.
  * `horizon_robustness_test.ipynb` — Multi-step degradation tests ($H \in \{1, 3, 7, 10, 14\}$).
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
* **Core Architectures Tested:** Dense, SimpleRNN, LSTM, GRU, Conv1D, Dilated Causal TCN, Bidirectional GRU, Self-Attention

---

## 👤 Author

**Jean Frégeville** — École Centrale de Lille