# **Logistic Regression Model for Spam Classification**

This document outlines the workflow of a **Logistic Regression** model implemented using **PyTorch** to classify emails as spam or ham. The process includes **data loading & preprocessing, model definition, training, testing, and prediction**.

---

## **1. Data Loading & Preprocessing (`load_data`)**
This function reads the email dataset and prepares the data for training.

### **Loading the dataset**  
- Reads `email_spam_dataset.csv` using **pandas**.  
- Renames columns (`v1` → `Labels`, `v2` → `Email`).  
- Converts labels:  
  - "ham" → `0`  
  - "spam" → `1`  

### **Text Processing**  
- **Converts to lowercase** for uniformity.  
- **Removes punctuation** using `str.translate()`.  
- **Tokenizes & removes stopwords** using **NLTK**'s `stopwords.words()`.  
- **Applies stemming** with `PorterStemmer` to reduce words to their root forms.  

### **Feature Extraction (Vectorization)**  
- Uses `CountVectorizer` from **scikit-learn** to transform text into a numerical feature matrix.  
- Converts `X` (email content) into a NumPy array.  
- Converts labels `Y` into a tensor (`torch.tensor`).  
- Saves the trained `CountVectorizer` as a **pickle file** for future use.  

---

## **2. Model Definition (`LogisticRegression` class)**
- Defines a simple logistic regression model using `torch.nn.Linear()`.  
- Applies a **sigmoid activation function** (`torch.sigmoid()`) to obtain probabilities between 0 and 1.  

  \[
  y = \sigma(WX + b)
  \]

### **Model Details**
- **`in_features`**: Number of input features (from vectorized emails).  
- **`out_features`**: Output (binary classification → `1` neuron).  

---

## **3. Model Training & Testing (`Train_Test_Model` class)**
Handles training, testing, and evaluation of the model.

### **(a) Data Splitting (`__get_train_test_split`)**
- Splits the dataset into **training (70%)** and **testing (30%)** using `train_test_split()`.

### **(b) Training (`train()`)**
1. **Forward pass**:  
   - Computes predictions using `self.model(self.X_train)`.  
2. **Loss computation**:  
   - Uses **Binary Cross Entropy Loss (BCELoss)** for classification.  
3. **Backward pass**:  
   - Computes gradients (`loss.backward()`).  
   - Updates weights using **Adam optimizer** (`optimizer.step()`).  
4. **Logging**:  
   - Prints loss at every **500 epochs**.  
5. **Completion**:  
   - Returns the trained model.  

---

## **4. Model Evaluation (`test()`)**
1. **Model inference mode (`torch.inference_mode()`)**  
   - Disables gradient computation for faster evaluation.  
2. **Predicts on test data (`self.model(self.X_test)`).**  
3. **Computes loss & accuracy**  
   - Uses `torchmetrics.classification.BinaryAccuracy(threshold=0.5)`.  
4. **Saves the trained model** using `pickle`.  

---

## **5. Making Predictions (`predict()`)**
- Converts model output probabilities into **binary predictions (0 or 1)**.  
- Returns predicted labels.  

---

## **6. Execution (`main()`)**
- Loads data.  
- Initializes **Logistic Regression model**.  
- Defines **loss function** (`BCELoss()`) and **optimizer** (`Adam`).  
- Trains the model (`train()`).  
- Evaluates on test data (`test()`).  
- Saves the trained model as `spam_classifier.pkl`.  

---

## **7. Summary of Workflow**

| **Step**       | **Process** |
|---------------|------------|
| **1. Load Data** | Read CSV, preprocess text (remove stopwords, stemming), convert to numerical features |
| **2. Define Model** | Logistic regression with `nn.Linear()` & `sigmoid` activation |
| **3. Train Model** | Forward pass → Loss calculation → Backpropagation → Optimizer update |
| **4. Evaluate Model** | Compute accuracy & save trained model |
| **5. Predict Labels** | Convert probabilities to `0` (ham) or `1` (spam) |
| **6. Save Model** | Stores trained model & vectorizer for future use |

This workflow enables **email spam classification** using **Logistic Regression** in PyTorch. 🚀
