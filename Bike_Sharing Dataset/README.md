# 🚴‍♂️ Bike Sharing Demand Prediction (Linear Regression & Neural Network)

## 📘 Project Overview
This project predicts hourly **bike rental demand** using two approaches — a **Linear Regression model** and a **Neural Network model** — trained on the same dataset.  
By comparing both, we can see how deep learning captures complex relationships better than traditional regression.

---

## 🎯 Objective
Predict the number of bikes rented (`cnt`) in a given hour based on:
- Weather and environmental conditions
- Temporal features (hour, season, weekday)
- Working day or holiday status

The project aims to evaluate:
- How much linear models can explain demand
- How neural networks improve performance on nonlinear patterns

---

## 📊 Dataset
**File:** `hour.csv`  
**Source:** **Source:** [Kaggle Bike Sharing Dataset](https://www.kaggle.com/datasets/abhishekkrg/bike-sharing-dataset)  
 
**Period:** 2011–2012  
**Target Variable:** `cnt` (total rentals per hour)

| Feature | Description |
|----------|-------------|
| `season`, `mnth`, `hr`, `weekday` | Temporal attributes |
| `temp`, `atemp`, `hum`, `windspeed` | Environmental factors |
| `holiday`, `workingday`, `weathersit` | Contextual indicators |
| `cnt` | Total bike rentals (target) |

---

## ⚙️ Data Preprocessing
Both models share similar preprocessing logic:
- Remove missing values and outliers (using IQR)
- Convert categorical features using one-hot encoding
- Scale numeric features with **MinMaxScaler**
- For Neural Network: additional **cyclical encoding** (`sin`, `cos`) for time features (`hour`, `month`, etc.)
- Align columns between train/test sets

---

## 🧮 Linear Regression Model

### 📘 Implementation Highlights
- Library: **Scikit-learn** (`LinearRegression`)
- Dataset split: 80% training, 20% testing
- Outlier removal for `windspeed` and `cnt`
- Feature expansion via one-hot encoding
- Checked multicollinearity using **Variance Inflation Factor (VIF)**

### 🧠 Training & Evaluation
```python
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
r2_score(y_test, y_pred)
```