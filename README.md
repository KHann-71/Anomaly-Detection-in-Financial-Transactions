
![nani]([https://link-to-your-image.com/image.png](https://media.licdn.com/dms/image/v2/D4E12AQGrtavhlzr-oA/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1689880445733?e=1759968000&v=beta&t=PTAElv9Kk5nbBuFQ8MeWSGBWA-0Bv7YCb_OR17p2QYY))


# 💳 Anomaly Detection in Financial Transactions

## 1. Introduction
Financial fraud is a critical challenge in the banking and e-commerce sectors.  
Fraudulent transactions often result in significant financial losses and erode customer trust.  
This project focuses on **automated fraud detection** by combining **traditional machine learning** and **Graph Neural Networks (GNN)** to detect anomalies in transaction data.

The pipeline supports:
1. **Tabular Models** – Logistic Regression, Random Forest, Autoencoder.
2. **Graph-based Models** – GCN, GAT, GIN.
3. **Configurable Preprocessing** – All settings are adjustable via `config/config.yaml`.

---

## 2. Datasets

### **Transaction Data**
- **Source:** TODO – Describe your source (synthetic data, Kaggle dataset, internal banking logs, etc.)
- **Files in `data/`**:
  - `transactions.csv` – Main transactions dataset.
  - `fraud_labels.csv` – Fraud labels (if stored separately).
  - `users.json` – User information.
  - `cards.json` – Card details.
  - `merchant.json` – Merchant details.
- **Target Column:** `fraud_label`  
- **Characteristics:**
  - Strong **class imbalance** (fraud cases are much rarer than normal transactions).
  - Rich **categorical** (e.g., merchant category, user region) and **numerical** features (e.g., amount, credit limit).

---

## 3. Methodology

### 3.1 Data Preprocessing
- Load and merge multiple CSV/JSON files.
- Handle missing values, drop irrelevant columns.
- Encode categorical features, scale numerical features.
- Handle class imbalance:
  - **SMOTE** for machine learning models.
  - **Undersampling** for GNN (keeping all fraud cases + a subset of normal transactions).

### 3.2 Graph Construction
- **Node type:** Configurable (`card` or `transaction`).
- **Edge types:**
  - By shared **client**.
  - By shared **merchant**.
  - By **time window** for transaction-based graphs.
- Node features: Amount, log-transformed amount, credit limit.

### 3.3 Models
- **Logistic Regression (LR)** – Baseline linear model.
- **Random Forest (RF)** – Ensemble-based decision trees.
- **Autoencoder** – Unsupervised anomaly detection.
- **GCN** – Graph Convolutional Network.
- **GAT** – Graph Attention Network.
- **GIN** – Graph Isomorphism Network.

### 3.4 Evaluation
- Metrics: Precision, Recall, F1-score, ROC-AUC.
- Threshold tuning via `scripts/evaluate_threshold.py`.

---

## 4. Results

| Model          | Precision | Recall | F1-score | ROC-AUC |
|----------------|-----------|--------|----------|---------|
| Logistic Regression | 0.78      | 0.62   | 0.69     | 0.85    |
| Random Forest       | 0.84      | 0.71   | 0.77     | 0.91    |
| Autoencoder         | 0.72      | 0.65   | 0.68     | 0.82    |
| GCN                 | 0.86      | 0.75   | 0.80     | 0.93    |
| GAT                 | 0.88      | 0.78   | 0.83     | 0.94    |
| GIN                 | 0.87      | 0.79   | 0.83     | 0.94    |

---

## 5. Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Anomaly-Detection-in-Financial-Transactions.git
cd Anomaly-Detection-in-Financial-Transactions

# Install dependencies
pip install -r requirements.txt
