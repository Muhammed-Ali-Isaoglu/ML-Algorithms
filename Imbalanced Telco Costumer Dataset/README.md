# 📞 Telco Customer Churn Prediction

## 📘 Project Overview
This project focuses on predicting **customer churn** using the **Telco Customer Churn dataset**.  
It demonstrates a complete end-to-end **machine learning pipeline**, including:
- Data preprocessing & feature encoding
- Exploratory data analysis (EDA)
- Handling class imbalance
- Feature scaling
- Model training and hyperparameter tuning
- Evaluation using multiple metrics
- Comparison of optimization techniques (GridSearchCV, RandomizedSearchCV, Optuna)

---

## 📊 Dataset
**Source:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`  
Each record represents a telecom customer, containing demographics, services subscribed, and churn status.

| Feature Type | Example Features |
|---------------|------------------|
| Categorical | Gender, Partner, Contract, InternetService |
| Numerical | Tenure, MonthlyCharges, TotalCharges |
| Target | Churn (Yes/No) |

---

## 🧹 Data Preprocessing

### 🔹 Cleaning
- Replaced invalid string values (`' '`, `'NA'`, etc.) in `TotalCharges` with `NaN`
- Converted data types and filled missing values with median

### 🔹 Encoding Techniques
| Encoding Type | Description | Applied To |
|----------------|--------------|-------------|
| **Label Encoding** | Binary mapping (Yes→1, No→0) | Gender, Partner, Dependents, etc. |
| **One-Hot Encoding** | Categorical → multiple binary columns | Contract, InternetService, PaymentMethod |
| **Ordinal Encoding (Explained)** | Ordered categories mapped numerically | (e.g. Education: High School < PhD) |
| **Target / Mean Encoding (Explained)** | Category replaced by target mean | (Used conceptually for high-cardinality columns) |

---

## 📈 Exploratory Data Analysis (EDA)

Explored the relationship between categorical variables and churn using:
- **Countplots** (gender, contract type, internet services, etc.)
- **Histograms** (tenure, charges)
- **Boxplots** to detect outliers
- **Correlation heatmap** to visualize relationships among numerical features

---

## ⚖️ Handling Class Imbalance

The churn dataset is **imbalanced**, meaning there are fewer “Churn = Yes” cases.  
Techniques applied:
1. **SMOTE (Synthetic Minority Oversampling Technique)** → Balances minority class
2. **Class Weights Adjustment** → Penalizes misclassifications of the minority class
3. **Comparison of pre/post SMOTE distributions** with `Counter()`

---

## ⚙️ Feature Scaling
- Used **StandardScaler** to normalize numerical values for better model performance.

---

## 🤖 Models and Algorithms

### 1️⃣ **Logistic Regression**
- Used as the baseline model
- Tuned with **GridSearchCV** over:
  - `C` (regularization strength)
  - `penalty` (`l1`, `l2`)
  - `solver` (`liblinear`)
- Evaluation Metric: **F1 Score** (preferred for imbalanced datasets)
- Class weight = `'balanced'` to handle imbalance internally

**Interpretation:**
- F1 ≈ 0.63 (moderate performance)
- Small difference between train/test → not overfitting
- ROC-AUC ≈ 0.73 → acceptable discrimination ability

---

### 2️⃣ **Random Forest Classifier**
- Applied **RandomizedSearchCV** for efficient parameter exploration.
- Parameters tuned:
  - `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`
- Scoring metric: **F1 Score**
- Used class weight balancing for fairness across classes

**Results:**
- Improved F1 and ROC-AUC over Logistic Regression
- Good generalization (minimal overfitting)

---

### 3️⃣ **Bayesian Optimization (Optuna)**
- Implemented **Optuna** for intelligent hyperparameter search using Bayesian optimization.
- Dynamically selects next parameter set based on previous performance.
- Evaluated model using 5-fold cross-validation.
- Achieved the best performance with **Random Forest + Optuna**, reaching an average F1 ≈ **0.92** after 50 trials.

**Why Optuna?**
| Method | When to Use | Characteristics |
|--------|--------------|----------------|
| GridSearchCV | Few parameters (<3) | Exhaustive, slow |
| RandomizedSearchCV | Medium-sized search space | Faster, random exploration |
| **Optuna (Bayesian)** | Large/complex search space | Smart, adaptive optimization |

---

## 📊 Evaluation Metrics

| Metric | Description | Use Case |
|--------|--------------|-----------|
| **Accuracy** | % of correct predictions | For balanced data |
| **Precision** | TP / (TP + FP) | Avoid false positives |
| **Recall** | TP / (TP + FN) | Catch all churners |
| **F1 Score** | Harmonic mean of Precision & Recall | Imbalanced datasets |
| **ROC-AUC** | Probability model ranks a random positive higher than a random negative | Overall discrimination power |

### 💡 Real-World Example:
- High Precision → You only flag churners when sure (low false alarms)
- High Recall → You catch all possible churners (even with some false positives)
- Balanced F1 → Good trade-off for retention marketing

---

## 🧠 Results Summary

| Model | Tuning Method | F1 (Test) | ROC-AUC | Notes |
|--------|----------------|-----------|----------|--------|
| Logistic Regression | GridSearchCV | ~0.63 | ~0.73 | Baseline |
| Random Forest | RandomizedSearchCV | ~0.84 | ~0.88 | Better generalization |
| Random Forest | Optuna (Bayesian) | ~0.92 | ~0.90+ | Best overall |

---

## 🔁 Final Model with SMOTE
- Applied SMOTE before Random Forest training.
- Improved minority class recall and balanced overall performance.
- Evaluated using:
  - **Classification report**
  - **Confusion matrix**
  - **ROC-AUC**

---

## 🧩 Key Takeaways
- **Encoding choice matters**: use label encoding for binary, one-hot for nominal.
- **Always scale numerical data** before logistic or SVM-based models.
- **For imbalanced data**:
  - Use F1, not accuracy.
  - Apply SMOTE or adjust class weights.
- **Optuna outperforms grid/random search** in both speed and accuracy for complex models.
- **Visualizations** help uncover bias and relationships before modeling.

---

## 🛠️ Libraries Used
- **NumPy**, **Pandas** → Data manipulation
- **Matplotlib**, **Seaborn** → Visualization
- **scikit-learn** → Preprocessing, modeling, evaluation
- **imblearn (SMOTE)** → Oversampling technique
- **Optuna** → Bayesian hyperparameter optimization

---

## 🚀 How to Run
```bash
# Clone the repository
git clone https://github.com/<yourusername>/ML-Algorithms.git
cd ML-Algorithms/01-Supervised-Learning/Telco-Churn

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook telco_churn_prediction.ipynb
