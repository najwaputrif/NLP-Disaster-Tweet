# 🌪️ Natural Language Processing with Disaster Tweets

This project is part of the **Data Mining & Business Intelligence** course assignment, conducted by **Group 8** in the **2024/2025 academic year**.

The case study is based on the following Kaggle competition:  
[**Natural Language Processing with Disaster Tweets**](https://www.kaggle.com/competitions/nlp-getting-started/overview)

---

## 🧠 Project Description

The main objective of this competition is to **predict whether a tweet is related to a real disaster event**. Participants are provided with a labeled dataset containing text data (tweets), where each tweet is marked as either `1` (disaster) or `0` (not disaster). The challenge lies in building a machine learning model that can accurately perform this binary classification based on the tweet's content.

Since the input data is entirely textual, **Natural Language Processing (NLP)** techniques are essential to transform raw text into structured input suitable for both traditional machine learning and deep learning models.

The project workflow includes:

- **Exploratory Data Analysis (EDA):** Analyzing tweet length, word frequency, punctuation usage, and class balance.
- **Text Preprocessing:** Cleaning the text by removing URLs, HTML tags, special characters, stopwords, and converting all text to lowercase. Lemmatization and stemming are applied as well.
- **Feature Extraction:** Using **TF-IDF (Term Frequency - Inverse Document Frequency)** to transform textual data into numerical feature vectors.

### 🧪 Modeling

We implemented a wide range of models to compare performance:

#### Classical Machine Learning Models:
- **Logistic Regression**
- **Random Forest Classifier**
- **Decision Tree Classifier**
- **Support Vector Machine (SVM)**
- **Naive Bayes (MultinomialNB)**

Each of these models was tuned using **GridSearchCV** or **RandomizedSearchCV** for optimal hyperparameter selection.

#### Boosting Techniques:
- **XGBoost Classifier**
- **LightGBM Classifier**

These gradient boosting models were included to enhance performance through iterative learning and were also subjected to hyperparameter tuning.

#### Ensemble Learning:
- A **Voting Classifier** was constructed using the top 3 performing models:
  - Decision Tree
  - Random Forest
  - Logistic Regression

This ensemble aimed to leverage the strengths of each model for improved generalization.

### 🤖 Final Model – BERT

As a final step, we implemented **BERT (Bidirectional Encoder Representations from Transformers)** using the `transformers` library. BERT is a pre-trained deep learning model designed to understand the contextual meaning of words in a sentence. It significantly improves performance by leveraging transfer learning from large-scale text corpora.

We used:
- `bert-base-uncased` model
- Tokenization using BERT tokenizer
- Fine-tuning using a classification head
- Training with GPU acceleration for better efficiency

### 🏁 Evaluation

All models were evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

This allowed us to compare both traditional and deep learning models comprehensively to select the best-performing approach.


---

## 📁 Folder Structure

```
📦disaster-tweets-nlp
 ┣ 📂notebooks/
 ┃ ┗ 📜NLP-Disaster-Tweet.ipynb
 ┣ 📂data/
 ┃ ┣ 📜train.csv          # Training data with labels
 ┃ ┣ 📜test.csv           # Unlabeled test data for prediction
 ┃ ┗ 📜submission.csv     # Final submission file with predictions
 ┗ 📜README.md
```

---

## 🚀 How to Run

1. Clone this repository:
   ```
   git clone https://github.com/najwaputrif/NLP-Disaster-Tweet.git
   ```

2. Open the Jupyter notebook located in the `notebooks/` folder:
   ```
   notebooks/NLP-Disaster-Tweet.ipynb
   ```

3. Ensure the required libraries are installed, such as:
   - `pandas`, `numpy`, `scikit-learn`
   - `nltk`, `re`, `string`
   - `transformers`, `torch` (for BERT)

4. Run the notebook sequentially to replicate preprocessing, modeling, and final prediction steps.

---

## 📌 Additional Notes

- The dataset was obtained from Kaggle and is not included in full due to file size and licensing limitations.
- `submission.csv` contains the final predictions using the best-performing model.
- For running BERT, GPU acceleration is recommended to reduce training and inference time.
