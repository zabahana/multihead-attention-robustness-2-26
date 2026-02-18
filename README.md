# Multi-Head Attention Robustness for Cross-Sectional Asset Pricing

Code and notebooks for the paper **"Inherent Robustness of Multi-Head Attention in Cross-Sectional Asset Pricing: Theory and Empirical Evidence from Finance-Valid Adversarial Attacks"**.

## Overview

This repository implements a feature-token transformer with head-diversity regularization for robust cross-sectional expected-return prediction. We establish theoretically motivated robustness improvements scaling as Ω(1/√H) through information redundancy, ensemble stabilization, and Lipschitz regularization, and validate empirically on S&P 500 stocks under finance-valid adversarial attacks.

## Key Contributions

- **Theoretical framework**: Multi-head attention with head-diversity regularization provides robustness improvements scaling as Ω(1/√H) under mild diversity assumptions
- **Finance-valid attack framework**: Four attack types reflecting realistic market scenarios—measurement error (A1), missingness/staleness (A2), rank manipulation (A3), regime shift (A4)
- **Empirical validation**: Seven models (OLS, Ridge, XGBoost, MLP, Single-Head, Multi-Head, Multi-Head Diversity) evaluated on prediction accuracy and adversarial robustness
- **Architecture-dependent robustness**: Multi-Head and Single-Head gain from adversarial training; Multi-Head Diversity achieves higher robustness under standard training

## Models

| Model | Description |
|-------|-------------|
| **Linear** | OLS, Ridge |
| **XGBoost** | Gradient-boosted trees |
| **MLP** | Non-attention neural baseline |
| **Single-Head** | Transformer with H=1 |
| **Multi-Head** | Transformer with H=8 heads, no diversity regularization |
| **Multi-Head Diversity** | Transformer with H=8 heads + head-diversity regularization |

## Project Structure

```
├── notebooks/          # Jupyter notebooks for data, training, evaluation
│   ├── 01_data.ipynb
│   ├── 02_train_baseline.ipynb
│   ├── 03_train_standard.ipynb
│   ├── 04_train_adversarial.ipynb
│   ├── 05_evaluation.ipynb
│   ├── 06_summary.ipynb
│   └── 07_H_scaling_and_FTTransformer.ipynb
├── src/
│   └── models/        # Feature-token transformer implementation
├── data/               # Cross-sectional data (not in repo; add locally)
└── outputs/            # Robustness results
```

## Setup

1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/zabahana/multihead-attention-robustness-2-26.git
   cd multihead-attention-robustness-2-26
   pip install torch pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
   ```

2. Add cross-sectional data to `data/cross_sectional/` (prices, fundamentals, `master_table.csv`). See `data/cross_sectional/200_tickers_5industries.txt` for ticker list.

## Usage

Run the notebooks in order:

1. **01_data.ipynb** — Load and preprocess cross-sectional data
2. **02_train_baseline.ipynb** — Train baseline models (OLS, Ridge, XGBoost, MLP)
3. **03_train_standard.ipynb** — Standard training for transformer models
4. **04_train_adversarial.ipynb** — Adversarial training for transformer models
5. **05_evaluation.ipynb** — Model evaluation and metrics
6. **06_summary.ipynb** — Results summary
7. **07_H_scaling_and_FTTransformer.ipynb** — H-scaling ablation and FT-Transformer baseline

Training: 2005–2017 | Validation: 2018–2019 | 22 characteristics, 8 attention heads

## License

See repository for license terms.
