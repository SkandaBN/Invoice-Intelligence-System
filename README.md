# Vendor Invoice Intelligence System
** Freight Cost Prediction & Invoice Risk Flagging **

##  Table of Contents
- < a href="#project-overview">Project Overview</a>
<a href="#business-objectives">Business Objectives</a>
- < a href="#data-sources">Data Sources</a>
- < a href="#eda">Exploratory Data Analysis</a>
= < a href="#models-used">Models Used</a>
- < a href="#metrics">Evaluation Metrics</a>
- < a href="#application">Application</a>
= < a href="#project-structure">Project Structure</a>
- < a href="#how-to-run-this-project">How to Run This Project</a>
- < a href="#author--contact">Author & Contact</a>
----

<h2><a class="anchor" id="project-overview"></a> Project Overview</h2>

This project implements an ** end-to-end machine learning system**designed to support finance teams by:
1. ** Predicting expected freight cost ** for vendor invoices.
2. ** Flagging high-risk invoices ** that require manual review due to abnormal cost, freight, or operational patterns.

---

<h2><a class="anchor" id="business-objectives"></a> Business Objectives</h2>

# 🧾 Vendor Invoice Intelligence System

> An end-to-end machine learning system that automates **freight cost prediction** and
> **invoice risk flagging** to reduce financial leakage and manual workload in vendor invoice processing.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-lightblue?logo=sqlite)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../../../Downloads/LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

---

## 📌 Table of Contents

- [Project Overview](../../../../Downloads/README.md#-project-overview)
- [Business Objectives](../../../../Downloads/README.md#-business-objectives)
- [Project Organization](../../../../Downloads/README.md#-project-organization)
- [Tech Stack](../../../../Downloads/README.md#-tech-stack)
- [Data Source](../../../../Downloads/README.md#-data-source)
- [Module 1 — Freight Cost Prediction](../../../../Downloads/README.md#-module-1--freight-cost-prediction)
- [Module 2 — Invoice Risk Flagging](../../../../Downloads/README.md#-module-2--invoice-risk-flagging)
- [Streamlit Application](../../../../Downloads/README.md#-streamlit-application)
- [Installation](../../../../Downloads/README.md#-installation)
- [Usage](../../../../Downloads/README.md#-usage)
- [Key Results](../../../../Downloads/README.md#-key-results)
- [Future Improvements](../../../../Downloads/README.md#-future-improvements)
- [Author](../../../../Downloads/README.md#-author)
- [License](../../../../Downloads/README.md#-license)

---

## 📖 Project Overview

The **Vendor Invoice Intelligence System** is a production-ready, dual-module ML application
designed to support finance and operations teams with two core AI-driven capabilities:

1. **Freight Cost Prediction** — Forecast expected freight costs for vendor invoices using
   invoice dollar value and quantity, enabling accurate budgeting and vendor negotiations.

2. **Invoice Risk Flagging** — Automatically classify whether a vendor invoice requires
   **manual approval** based on anomalous cost, freight, and delivery patterns — reducing
   exposure to invoice fraud and processing errors.

Both modules are served through a unified **Streamlit web portal** accessible to non-technical
finance teams without any coding knowledge.

---

## 🎯 Business Objectives

| Objective | Solution |
|---|---|
| Reduce manual invoice review workload | Auto-flag only high-risk invoices for human review |
| Improve freight cost forecasting | ML regression model trained on historical invoice data |
| Detect invoice fraud and anomalies | Rule-informed binary classification with Random Forest |
| Enable finance team self-service | Streamlit UI requiring zero technical knowledge |

---

## 📁 Project Organization

```
Invoice-Intelligence-System/
│
├── data/
│   └── inventory.db                        <- SQLite database (vendor_invoice + purchases tables)
│
├── notebooks/
│   ├── Invoice Flagging.ipynb              <- EDA, labeling logic & model training (Classification)
│   └── Predicting Freight Cost.ipynb       <- EDA, feature analysis & model training (Regression)
│
├── freight_cost_prediction/                <- Freight Cost Regression Module
│   ├── data_preprocessing.py              <- Load from SQLite, feature/target selection, train-test split
│   ├── model_evaluation.py                <- Train Linear Regression, Decision Tree, Random Forest; MAE/MSE/R²
│   └── train.py                           <- Orchestrate training, compare models, save best model
│
├── invoice_flagging/                       <- Invoice Risk Classification Module
│   ├── data_preprocessing.py              <- CTE-based SQL aggregation, risk label engineering, scaling
│   ├── model_evaluation.py                <- Random Forest with GridSearchCV (F1 scorer, 5-fold CV)
│   ├── train.py                           <- Full training pipeline, save model + scaler
│   └── models/
│       ├── random_forest_model.pkl        <- Serialized best classifier
│       └── scaler.pkl                     <- Fitted StandardScaler
│
├── inference/                              <- Inference Layer (called by Streamlit app)
│   ├── __init__.py                        <- Exports predict_freight_cost, predict_invoice_flag
│   ├── predict_freight.py                 <- Load model, run regression inference, return DataFrame
│   └── predict_invoice_flag.py            <- Load classifier, run prediction, return flag result
│
├── models/
│   └── predict_freight_cost_model.pkl     <- Serialized best regression model
│
├── app.py                                 <- Streamlit web application (main entry point)
├── requirements.txt                       <- Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.9+ |
| Database | SQLite (via `sqlite3` — Python stdlib) |
| Data Handling | Pandas, NumPy |
| ML — Classification | Scikit-learn `RandomForestClassifier` + `GridSearchCV` |
| ML — Regression | Scikit-learn `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor` |
| Preprocessing | Scikit-learn `StandardScaler`, `train_test_split` |
| Model Persistence | Joblib |
| Visualization | Plotly Express |
| Web Application | Streamlit |
| Notebooks | Jupyter |
| Version Control | Git, GitHub |

---

## 🗄️ Data Source

Data is stored in a **SQLite database** (`data/inventory.db`) with two core tables:

**`vendor_invoice`** — One row per invoice, containing:
- `PONumber`, `Quantity`, `Dollars`, `Freight`
- `InvoiceDate`, `PODate`, `PayDate`

**`purchases`** — Line-item purchase records, containing:
- `PONumber`, `Brand`, `Quantity`, `Dollars`, `ReceivingDate`, `PODate`

### SQL Feature Engineering (Invoice Flagging Module)

A CTE-based query aggregates purchase-level data to the PO level and joins with invoice records:

```sql
WITH purchase_agg AS (
    SELECT
        p.PONumber,
        COUNT(DISTINCT p.Brand)                                    AS total_brands,
        SUM(p.Quantity)                                            AS total_item_quantity,
        SUM(p.Dollars)                                             AS total_item_dollars,
        AVG(julianday(p.ReceivingDate) - julianday(p.PODate))      AS avg_receiving_delay
    FROM purchases p
    GROUP BY p.PONumber
)
SELECT
    vi.Quantity                                                    AS invoice_quantity,
    vi.Dollars                                                     AS invoice_dollars,
    vi.Freight,
    (julianday(vi.InvoiceDate) - julianday(vi.PODate))             AS days_po_to_invoice,
    (julianday(vi.PayDate)     - julianday(vi.InvoiceDate))        AS days_to_pay,
    pa.total_brands,
    pa.total_item_quantity,
    pa.total_item_dollars,
    pa.avg_receiving_delay
FROM vendor_invoice vi
LEFT JOIN purchase_agg pa ON vi.PONumber = pa.PONumber
```

---

## 📦 Module 1 — Freight Cost Prediction

### Objective
Predict the freight cost for a vendor invoice given its **Invoice Dollars** and **Quantity**.

### Features & Target

| Feature | Description |
|---|---|
| `Dollars` | Total invoice dollar value |
| `Quantity` | Number of items in the invoice |
| **`Freight`** | **Target** — freight cost to predict |

### Models Trained & Compared

| Model | Description |
|---|---|
| `LinearRegression` | Baseline — captures linear relationship between inputs and freight |
| `DecisionTreeRegressor` | Non-linear splits, `max_depth=5` |
| `RandomForestRegressor` | Ensemble of trees, `max_depth=5`, `n_estimators=100` |

### Model Selection Logic
All three models are evaluated and the one with **lowest MAE** is automatically saved:

```python
best_model_info = min(results, key=lambda x: x['MAE'])
joblib.dump(best_model, "models/predict_freight_cost_model.pkl")
```

### Evaluation Metrics
- Mean Absolute Error (MAE) — primary selection criterion
- Mean Squared Error (MSE)
- R² Score

---

## 🚨 Module 2 — Invoice Risk Flagging

### Objective
Classify whether a vendor invoice should be **flagged for manual approval** (1)
or **cleared for auto-approval** (0) based on financial and operational signals.

### Risk Labeling Logic (Business Rules → Ground Truth)

```python
def create_invoice_risk_label(row):
    if row['invoice_dollars'] > 10000:       return 1  # High-value invoice
    if row['days_po_to_invoice'] > 30:       return 1  # Delayed invoice submission
    if row['days_to_pay'] > 60:              return 1  # Slow payment risk signal
    if row['avg_receiving_delay'] > 10:      return 1  # Persistent delivery delays
    return 0
```

### Features Used for Classification

| Feature | Description |
|---|---|
| `invoice_quantity` | Number of items in the invoice |
| `invoice_dollars` | Total invoice value |
| `Freight` | Freight cost on the invoice |
| `total_item_quantity` | Aggregated quantity across all PO purchases |
| `total_item_dollars` | Aggregated dollar value across all PO purchases |

### Model — Random Forest + GridSearchCV

```python
param_grid = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [None, 4, 5, 6],
    "min_samples_split": [2, 3, 5],
    "min_samples_leaf":  [1, 2, 5],
    "criterion":         ["gini", "entropy"]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring=make_scorer(f1_score),   # F1 — better than accuracy for imbalanced labels
    cv=5,
    n_jobs=-1
)
```

> **Why F1 as the scoring metric?**
> Invoice flags are inherently imbalanced — most invoices are safe. Optimizing for
> accuracy would produce a model that simply predicts "safe" for everything. F1 balances
> Precision (avoid flooding reviewers with false alarms) and Recall (don't miss real risks).

### Preprocessing Pipeline
Features are scaled using `StandardScaler` fitted on training data only, then persisted
separately as `scaler.pkl` so inference uses the exact same scaling seen during training.

### Evaluation Metrics
- Accuracy
- Classification Report (Precision, Recall, F1-Score per class)

---

## 🖥️ Streamlit Application

The unified **Vendor Invoice Intelligence Portal** (`app.py`) provides two selectable
modules via the sidebar:

### Module A — Freight Cost Prediction UI
- **Inputs:** Invoice Dollars, Quantity
- **Output:** Estimated freight cost as `$X,XXX.XX` metric card

### Module B — Invoice Manual Approval Flag UI
- **Inputs:** Invoice Dollars, Freight Cost, Total Item Quantity, Total Item Dollars
- **Output:**
  - `⚠️ Invoice requires MANUAL APPROVAL` — if model predicts flag = 1
  - `✅ Invoice is SAFE for Auto-Approval` — if model predicts flag = 0

### Sidebar Business Impact Panel
```
📉 Improved cost forecasting
🧾 Reduced invoice fraud & anomalies
⚙️ Faster finance operations
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Invoice-Intelligence-System.git
cd Invoice-Intelligence-System

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Train the Freight Cost Regression Model

```bash
cd freight_cost_prediction
python train.py
# Output: ../models/predict_freight_cost_model.pkl
```

### Step 2 — Train the Invoice Risk Classifier

```bash
cd invoice_flagging
python train.py
# Output: models/random_forest_model.pkl
#         models/scaler.pkl
```

> ⚠️ **Before running either train script**, update the `db_path` variable to point
> to your local copy of `data/inventory.db`. The hardcoded Windows path in the current
> scripts must be replaced with a relative or environment-based path.

### Step 3 — Launch the Streamlit App

```bash
# From the project root directory
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## 🏆 Key Results

| Module | Best Model | Selection Criterion |
|---|---|---|
| Freight Cost Prediction | Random Forest Regressor | Lowest MAE (auto-selected vs. Linear & Decision Tree) |
| Invoice Risk Flagging | Random Forest + GridSearchCV | Best F1-Score across 5-fold cross-validation |

- ✅ Business-rule-informed labeling produces domain-aligned, meaningful training targets
- ✅ CTE-based SQL aggregation enriches invoice features with PO-level purchase patterns
- ✅ Modular `inference/` layer completely decouples prediction logic from the Streamlit UI
- ✅ StandardScaler persisted separately ensures train/inference consistency
- ✅ F1-optimized GridSearchCV prevents the classifier from ignoring minority (risk) class

---

## 🔭 Future Improvements

- [ ] Replace hardcoded `db_path` with `.env` + `python-dotenv` for portability
- [ ] Add **SHAP explainability** — show which features drove each invoice flag decision
- [ ] Build a **FastAPI REST endpoint** to serve both models programmatically
- [ ] Add **Optuna** hyperparameter tuning to replace GridSearchCV for faster optimization
- [ ] Build a **LangChain RAG layer** for natural language querying over invoice history
- [ ] Integrate **OCR pipeline** (Tesseract / AWS Textract) to handle scanned invoice PDFs
- [ ] Add **MLflow** for experiment tracking and model versioning
- [ ] Deploy on **Streamlit Cloud** or **AWS EC2** for team-wide access
- [ ] Add a **Plotly analytics dashboard** tab with invoice trends and risk distributions

---

## 👤 Author

**Skanda BN**  
ML Engineer / Data Scientist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/skanda-bn/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/SkandaBN)
[![Email](https://img.shields.io/badge/Email-sbn39008%40gmail.com-red?logo=gmail)](mailto:sbn39008@gmail.com)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](../../../../Downloads/LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ by Skanda BN | Alliance University, Bengaluru</sub>
</div>
