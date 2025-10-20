# 🚀 Spaceship Titanic Survival Prediction

## 📄 Description
This notebook builds a **classification model** to predict whether passengers were **transported to another dimension** after the Spaceship Titanic disaster.  
The project is inspired by the [Kaggle “Spaceship Titanic” competition](https://www.kaggle.com/competitions/spaceship-titanic).

We perform complete **data preprocessing**, **feature engineering**, and train multiple ML models to select the best performer (LightGBM).

---

## ⚙️ Steps Overview
1. **Data Loading**
   - `train.csv` and `test.csv` datasets are loaded.
2. **Preprocessing**
   - Splitting the `Cabin` column into deck/number/side
   - Handling missing values using median and mode imputation
   - Creating a combined `services` feature (sum of on-board expenditures)
   - Removing outliers (IQR clipping)
   - One-hot encoding categorical variables
3. **Modeling**
   - Training & validation split
   - Models tested:
     - LightGBM
     - Random Forest
     - AdaBoost
   - Comparison of validation accuracy
4. **Final Model**
   - LightGBM trained on full dataset
   - Predictions generated for test data
   - Submission CSV created

---

## 📊 Dataset Information
| Dataset | Shape | Description |
|----------|--------|-------------|
| `train.csv` | (8693, 14) | Includes target label “Transported” |
| `test.csv`  | (4277, 13) | For final submission |

**Target variable:** `Transported` (True/False)

---

## 🧪 Model Performance
| Model | Validation Accuracy | Notes |
|--------|---------------------|--------|
| LightGBM | **~0.80+** | Best performer |
| Random Forest | ~0.78 | Slightly lower |
| AdaBoost | ~0.76 | Underperforms |

**Internal test accuracy:** ~0.81  

---

## 🧠 Feature Engineering Summary
| Feature | Description |
|----------|-------------|
| `services` | Total expenditure on all on-board amenities |
| `deck`, `side` | Extracted from `Cabin` |
| `HomePlanet` | Filled based on deck mode |
| Boolean columns | Converted to 0/1 |
| Outliers | Clipped for `Age` and `services` |

---

## 🧰 Libraries Used
- **pandas**, **numpy**
- **matplotlib**, **seaborn**
- **scikit-learn**
- **lightgbm**

---

## 📈 Results & Insights
- LightGBM generalized well across validation and test sets.
- Feature `services` improved accuracy notably.
- Outlier removal stabilized model training.
- The final submission file is compatible with Kaggle’s evaluation format.

---

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install lightgbm pandas numpy seaborn matplotlib scikit-learn statsmodels
