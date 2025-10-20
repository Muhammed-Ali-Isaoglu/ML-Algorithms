# 🌳 Decision Tree Classifier (Iris Dataset)

## 📄 Description
This notebook demonstrates how to use a **Different ML Algorithms** on the famous Iris dataset to predict flower species based on sepal and petal measurements.

## ⚙️ Algorithms Used
- DecisionTreeClassifier (from sklearn)
- SVM
- Random Forest

## 🧪 Dataset
- **Source**: `Iris.csv` (UCI Machine Learning Repository)
- **Features**:
  - SepalLengthCm
  - SepalWidthCm
  - PetalLengthCm
  - PetalWidthCm
- **Target**: Species

## 📊 Results
| Model | Accuracy |
|--------|-----------|
| Decision Tree | 96.67% |
| Random Forest | 96.67% |
| SVM | 100% |

## 🧠 Key Insights
- SVM performed best on this dataset.
- Decision Tree and Random Forest had similar accuracy.
- Data was clean and balanced, no missing values.

## 🛠️ Tools
- Python, pandas, seaborn, matplotlib, sklearn

## 🚀 Run It
```bash
jupyter notebook decision_tree_iris.ipynb
