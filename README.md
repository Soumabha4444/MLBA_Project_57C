# Multimodal Deep Learning for Stock Price Forecasting: A Fusion-Based Framework

## 📘 Project Overview
This project explores a multimodal deep learning approach to forecast short-term **stock returns** by combining **numerical** (market indicators) and **textual** (Reddit sentiment) data.  
It compares traditional machine learning models like **Ridge Regression** with deep learning-based **LSTM** and fusion architectures integrating **SBERT embeddings** for sentiment analysis.

The objective is to demonstrate how fusing diverse data modalities can enhance the accuracy and interpretability of stock price forecasting.

---

## 🧩 Key Features
- **Numerical Modeling:** Ridge Regression and LSTM models using OHLCV-based financial indicators.
- **Textual Modeling:** SBERT (MiniLM-L6-v2) embeddings of Reddit headlines as sentiment features.
- **Fusion Strategies:** Early, Late, and Attention-Based fusion for integrating modalities.
- **Evaluation Metrics:** MAE, RMSE, and Directional Accuracy (DA) with bootstrap confidence intervals.
- **Reproducibility:** Fully open-source implementation with documented dependencies and dataset preprocessing.

---

## 🧠 Model Summary
| Model | Description | Key Findings |
|:------|:-------------|:--------------|
| **SN (Seasonal Naïve)** | Statistical benchmark repeating past seasonal patterns | High error, no predictive value |
| **RB (Ridge Baseline)** | Linear regression with L2 regularization | Strong baseline for numerical features |
| **LSTM** | Captures temporal patterns in price data | Similar to Ridge — numerical data mostly linear |
| **MEF (Early Fusion)** | Concatenates modalities before dense layers | Slight reduction in MAE and RMSE |
| **MLF (Late Fusion)** | Combines predictions post-encoding | **Best overall performance (highest DA)** |
| **MAF (Attention Fusion)** | Uses cross-modal attention | Comparable to other fusion methods |

---

## 🧪 Results Summary
| Model | MAE | RMSE | DA (%) |
|:------|:------:|:------:|:------:|
| SN | 0.0177 | 0.0232 | 49.71 |
| RB | 0.0122 | 0.0165 | 50.36 |
| MEF | 0.0114 | 0.0158 | 49.66 |
| MLF | 0.0114 | 0.0157 | **53.80** |
| MAF | 0.0114 | 0.0158 | 49.66 |

📈 The **Late Fusion (MLF)** model achieved the best trade-off between numerical precision and directional accuracy.

---

## 🗂️ Data Sources
- **Stock Data:** AAPL (Apple Inc.) from [Yahoo Finance](https://finance.yahoo.com/)
- **Textual Data:** [Reddit News Dataset (Kaggle)](https://www.kaggle.com/competitions/reddit-news)
- **Period Covered:** January 2008 – October 2016  
- **Target Variable:** Next-day log return \( r_t = \ln(P_t / P_{t-1}) \)

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Soumabha4444/MLBA_Project_57C.git
cd MLBA_Project_57C
```

### 2️⃣ Create the Conda Environment
```bash
conda env create -f environment.yml
conda activate mlba_project
```

### 3️⃣ Run the Project
```bash
python src/stock_forecasting_multimodal.py
```

Alternatively, open the Jupyter notebook (if provided) and execute the workflow cells.

---

## 🔁 Reproducibility
This project ensures reproducibility by providing access to the data, code, and methodology.  
The code and cleaned dataset are available at **[GitHub Repository Link](https://github.com/Soumabha4444/MLBA_Project_57C)**.  

To replicate the results:
1. Clone the repository  
2. Install dependencies (`environment.yml`)  
3. Run the main script or notebook  
All dependencies and software versions are documented in the environment file and README.

---

## 🚧 Limitations and Future Work
- FinBERT could not be implemented due to system resource constraints (GPU memory).  
  Future iterations will include FinBERT-based sentiment embeddings for domain-specific understanding.  
- Expanding datasets across multiple tickers could improve model generalization.  
- Incorporating transformer-based temporal models (e.g., Temporal Fusion Transformer) may enhance long-range trend learning.

---

## 🙏 Acknowledgment
We thank **Prof. Suman Sanyal** for his guidance and valuable feedback throughout the project.  
We also acknowledge the **Goa Institute of Management, Panaji**, for providing computational resources and a collaborative research environment.

---

## 📎 Citation
If you use this repository or reference this work, please cite it as:

```
Soumabha Nandi, Raunaq Singh Sarna, Suvodeep Saha. 
"Multimodal Deep Learning for Stock Price Forecasting: A Fusion-Based Framework." 
Goa Institute of Management, 2025.
```

---

## 📬 Contact
For queries or collaborations:  
📧 **soumabha.nandi@gim.ac.in**
