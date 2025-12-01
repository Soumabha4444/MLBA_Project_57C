# Multimodal Deep Learning for Stock Price Forecasting Using Fusion Mechanisms

## Project Overview
This project implements a multimodal deep learning framework that forecasts short-term **stock returns** by integrating **numerical** (OHLCV-based market indicators) and **textual** (Reddit news sentiment) data. 

It compares traditional machine learning models like **Ridge Regression** with deep learning-based **LSTM-based** architectures and three fusion architectures that incorporate **SBERT** and **FinBERT** embeddings.  

The objective is to demonstrate how multimodal fusion, sentiment-aware features, and event-specific conditioning can improve statistical accuracy and trading performance in financial forecasting.

---

## Key Features
- **Numerical Modeling:**  
  Ridge Regression, LSTM models, rolling statistics, and lag-return features.
- **Textual Modeling:**  
  SBERT (MiniLM-L6-v2) and an exploratory **FinBERT** experiment for finance-specific embeddings.
- **Fusion Strategies:**  
  Early, Late, and Attention-Based (cross-attention) fusion architectures.
- **Event Tagging:**  
  Macro / firm-specific / geopolitical tag generation and conditional modelling variants.
- **Trading Backtest:**  
  Long/flat strategy with Annualized Return, Sharpe Ratio, and Maximum Drawdown (MDD).
- **Evaluation Metrics:**  
  MAE, RMSE, Directional Accuracy (DA) + bootstrap confidence intervals.
- **Reproducibility:**  
  Fully open-source pipeline with dedicated scripts for training, embedding generation, and event tagging.

---

## Model Summary
| Model | Description | Key Findings |
|:------|:-----------|:-------------|
| **SN (Seasonal Naïve)** | Statistical benchmark | Highest error, no predictive value |
| **RB (Ridge Baseline)** | Strong linear baseline | Very competitive on numerical-only data |
| **LSTM** | Temporal modelling of OHLCV | Similar to Ridge — limited nonlinear signal |
| **MEF (Early Fusion)** | Concatenates numerical + textual early | Small improvements in MAE/RMSE |
| **MLF (Late Fusion)** | Independent branches + fusion head | **Best overall DA and backtest performance** |
| **MAF (Attention Fusion)** | Cross-modal attention | Comparable to MEF/MLF |
| **FinBERT + MLF (Exploratory)** | Finance-tuned embeddings | Small but consistent gains vs SBERT |
| **Event-Tagged Models** | Conditional fusion using event types | Modest improvements on event-heavy days |

---

## Results Summary
| Model | MAE | RMSE | DA (%) |
|:------|:----:|:-----:|:------:|
| SN | 0.017734 | 0.023151 | 49.71 |
| RB | 0.012199 | 0.016520 | 50.36 |
| MEF | 0.011432 | 0.015759 | 49.66 |
| MLF | **0.011444** | **0.015729** | **53.80** |
| MAF | 0.011449 | 0.015783 | 49.66 |

### **Backtest Summary (Long/Flat Strategy)**
| Model | Ann. Return | Sharpe | MDD |
|:------|:------------:|:------:|:----:|
| SN | 1.8% | 0.09 | 27.4% |
| RB | 4.5% | 0.26 | 23.8% |
| MEF | 5.1% | 0.29 | 23.1% |
| **MLF** | **6.3%** | **0.36** | **22.5%** |
| MAF | 4.9% | 0.28 | 23.4% |

The **Late Fusion (MLF)** model achieved the best combination of statistical accuracy and trading performance.

---

## Data Sources
- **Stock Data:** AAPL (Apple Inc.) from [Yahoo Finance](https://finance.yahoo.com/)
- **Textual Data:** [Reddit News Dataset (Kaggle)](https://www.kaggle.com/competitions/reddit-news)
- **Period Covered:** January 2008 – October 2016
- **Target Variable:** Next-day log return r_{t+1} = ln(P_{t+1} / P_t)

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Soumabha4444/MLBA_Project_57C.git
cd MLBA_Project_57C
```

### 2. Create the Conda Environment
```bash
conda env create -f environment.yml
conda activate mlba_project
```

### 3. Run the Project
Main multimodal script:
```bash
python src/stock_forecasting_multimodal.py
```

Generate SBERT or FinBERT embeddings:
```bash
python src/build_text_embeddings.py
python src/build_finbert_embeddings.py
```

Event tagging:
```bash
python src/event_tagging_and_conditional.py
```

Notebook workflow:
```bash
jupyter notebook
```

---

## Reproducibility
This repository includes all scripts used in the paper:

- train_baseline.py  
- train_lstm.py  
- train_multimodal_lstm.py  
- train_multimodal_late_fusion.py  
- train_multimodal_attention_fusion.py  
- build_text_embeddings.py  
- build_finbert_embeddings.py  
- event_tagging_and_conditional.py  

Dependencies and random seeds are fixed in `environment.yml`. CPU support is sufficient for training; FinBERT embeddings benefit from GPU.

---

## Limitations and Future Work
- Limited GPU resources restricted large-scale FinBERT experiments.  
- Event tagging used rule-based methods; future work may involve supervised classifiers.  
- Multi-ticker extensions and temporal transformers could further enhance performance.  
- Backtest uses a simple strategy and can be extended to richer trading logic.

---

## Acknowledgment 
We thank **Prof. Suman Sanyal** for his guidance and valuable feedback throughout the project.

We also acknowledge the **Goa Institute of Management, Panaji**, for providing computational resources and a collaborative research environment.

---

## Citation
If you use this repository or reference this work, please cite it as:

```
Soumabha Nandi, Raunaq Singh Sarna, Suvodeep Saha. 
"Multimodal Deep Learning for Stock Price Forecasting Using Fusion Mechanisms" 
Goa Institute of Management, 2025.
```

---

## Contact
For queries or collaborations:  
**soumabha.nandi2000@gmail.com**
**soumabha.nandi25b@gim.ac.in
