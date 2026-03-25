# Multi-Domain Data Analysis Engine

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2021b+-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A modular, high-performance analytical engine that integrates **Python** for machine learning and data orchestration with **MATLAB** for advanced numerical modeling and optimization. This project processes diverse datasets across **Finance**, **Healthcare**, and **Natural Language Processing (NLP)** domains.

---

## 🚀 Key Features

- **Hybrid Integration**: Seamless data exchange between Python and MATLAB via standardized CSV interfaces.
- **Financial Forecasting**: Combined ARIMA/Linear Regression (Python) with high-degree polynomial curve fitting (MATLAB).
- **Healthcare Diagnostics**: Scikit-learn classification models integrated with automated PCA dimensionality reduction in MATLAB.
- **NLP & Clustering**: TF-IDF vectorization and Naive Bayes classification (Python) paired with K-means clustering (MATLAB).
- **Automated Visualization**: Generates professional-grade plots and reports in the `output/` directory.

---

## 📁 Project Structure

```text
multi-domain-data-analysis-engine/
├── data/               # Processed datasets (CSV)
├── modules/            # Domain-specific logic
│   ├── financial/      # Stock analysis & curve fitting
│   ├── healthcare/     # Diagnostics & PCA
│   └── text/           # NLP & Clustering
├── output/             # Generated visualizations and reports
├── LICENSE             # Project license
├── README.md           # Project documentation (You are here)
└── requirements.txt    # Python dependencies
```

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.8+**
- **MATLAB** (with Statistics and Machine Learning Toolbox)

### Python Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/YohannesTH/multi-domain-data-analysis-engine.git
   cd multi-domain-data-analysis-engine
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### MATLAB Setup
Ensure the `modules/` subdirectories are added to your MATLAB path or run scripts directly from their respective folders.

---

## 📊 Module Overview

### 1. Financial Analysis
- **Python**: Downloads real-time stock data (default: `AAPL`) via `yfinance`. Implements **ARIMA** and **Linear Regression** to forecast closing prices.
- **MATLAB**: Performs degree-3 and degree-5 **polynomial curve fitting** for trend interpolation and calculation of RMSE.

### 2. Healthcare Analytics
- **Python**: Processes the **Breast Cancer Wisconsin** dataset. Trains **Random Forest** and **Logistic Regression** classifiers with comprehensive performance metrics.
- **MATLAB**: Executes **Principal Component Analysis (PCA)** on scaled features to identify top variance contributors and visualize clusters.

### 3. Text & NLP
- **Python**: Utilizes the **20 Newsgroups** dataset. Applies **TF-IDF vectorization** and **Naive Bayes** for category classification.
- **MATLAB**: Implements **K-means clustering** (k=3) on extracted text features, visualizing document groupings in 2D space.

---

## 📈 Results
All generated artifacts, including prediction plots, feature importance charts, and PCA clusters, are saved to the `output/` directory.

Example outputs:
- `module1_financial_analysis.png`
- `module3_rf_importances.png`
- `module4_healthcare_pca.png`
- `module6_text_clustering.png`

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.