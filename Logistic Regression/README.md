
---

## 1️⃣ Ad Click Prediction

**Goal:** Predict whether a user will click an ad based on user information and browsing behavior.

**Dataset:** `ad_click_dataset.csv`   
- **Columns include:** `id`, `full_name`, `age`, `gender`, `device_type`, `ad_position`, `browsing_history`, `time_of_day`, `click`.

**Notebook Features:**
- Data exploration and visualization
- Missing value handling and preprocessing
- Encoding categorical features
- Scaling numeric features
- Logistic Regression model
- Model evaluation (Accuracy, Classification Report, ROC-AUC)
- Multicollinearity check using VIF

---

## 2️⃣ Titanic Survival Prediction

**Goal:** Predict passenger survival on the Titanic based on personal information and ticket details.

**Dataset:** `train.csv`  
- **Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)  
- **Columns include:** `PassengerId`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`, `Survived`.

**Notebook Features:**
- Data exploration and visualization
- Handling missing values
- Feature engineering (`Family_size`)
- Encoding categorical features
- Scaling numeric features
- Logistic Regression model
- Model evaluation (Accuracy, Classification Report)
- Multicollinearity check using VIF

---

## 📦 Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-projects.git
cd ml-projects
