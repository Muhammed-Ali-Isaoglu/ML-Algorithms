
---

## 1️⃣ Image Classification Project

**Goal:** Build a Convolutional Neural Network (CNN) to classify images in a binary classification task.

**Dataset:** Folder-based image dataset (`train/images`, `valid/images`)  

**Techniques Used:**
- **Data Cleaning:** Removed corrupted or unreadable images using `OpenCV`.
- **Duplicate Removal:** Used `hashlib.md5` to remove duplicate images.
- **Data Augmentation:** Rescaled images with `ImageDataGenerator`.
- **Model Architecture:**  
  - Convolutional layers (`Conv2D`) + ReLU activation  
  - MaxPooling layers (`MaxPooling2D`)  
  - Fully connected layers (`Dense`) + Dropout for regularization  
  - Output layer with sigmoid activation for binary classification
- **Training & Evaluation:**  
  - Loss: `binary_crossentropy`  
  - Optimizer: `Adam`  
  - Metrics: Accuracy  
  - Visualization: Accuracy and validation accuracy plots

---

## 2️⃣ E-Commerce Review Prediction (Multi-Input Model)

**Goal:** Predict product ratings (1-5) using a combination of **text reviews** and **tabular features**.

**Dataset:** `Womens Clothing E-Commerce Reviews.csv`  
- Columns include `Title`, `Review Text`, `Age`, `Positive Feedback Count`, `Class Name`, etc.  

**Techniques Used:**

### 🔹 Data Preprocessing
- Filled missing values for text (`Title`, `Review Text`) and categorical columns (`Class Name`).
- Removed irrelevant columns (`Unnamed: 0`, `Clothing ID`, `Department Name`, `Division Name`).
- One-hot encoding for categorical features (`Class Name`).
- Scaling numerical features (`Age`, `Positive Feedback Count`) using `StandardScaler`.

### 🔹 Text Processing
- Combined `Title` + `Review Text`.
- Tokenized text with `Tokenizer` (vocab size = 1000) and padded sequences (max length = 100).
- Text input fed into an **Embedding layer** → **LSTM** → **1D Convolution (Conv1D)** → **GlobalMaxPooling1D**.

### 🔹 Tabular Data Processing
- Tabular input processed through **Dense layers**.
- Numerical and categorical features combined.

### 🔹 Model Architecture (Multi-Input)
- **Text Branch:** Embedding → LSTM → Conv1D → GlobalMaxPooling1D
- **Tabular Branch:** Dense layers on tabular inputs
- **Merged Branch:** Concatenated text + tabular features → Dense → Dropout → Output (softmax for 5 classes)
- **Loss Function:** `sparse_categorical_crossentropy`
- **Optimizer:** Adam
- **Metrics:** Accuracy

---

## 3️⃣ Techniques & Tools Used Across Projects

| Technique | Description |
|-----------|-------------|
| **CNN (Conv2D + MaxPooling)** | For feature extraction from images |
| **LSTM** | Captures sequential patterns in text reviews |
| **Conv1D + GlobalMaxPooling1D** | Extracts local text features and reduces sequence length |
| **Multi-Input Model** | Combines text embeddings with tabular features for prediction |
| **Dropout** | Prevents overfitting |
| **StandardScaler** | Normalizes numerical features |
| **One-Hot Encoding** | Converts categorical features into binary vectors |
| **Data Cleaning** | Handling missing values, duplicates, corrupted images |
| **Visualization** | `matplotlib` & `seaborn` for exploratory analysis |

---

## 📦 Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-projects.git
cd ml-projects
