# 🌪️ Natural Language Processing with Disaster Tweets

This project is part of the **Data Mining & Business Intelligence** course assignment, conducted by **Group 8** in the **2024/2025 academic year**.

The case study is based on the following Kaggle competition:  
[**Natural Language Processing with Disaster Tweets**](https://www.kaggle.com/competitions/nlp-getting-started/overview)


## 🧠 Project Description

The main objective of this competition is to **predict whether a tweet refers to a real disaster event**. The dataset contains labeled tweets, where each entry is marked as either `1` (disaster) or `0` (not disaster). The key challenge is building a machine learning model that can effectively perform this binary classification based solely on the tweet text.

Because the input data is purely textual, various **Natural Language Processing (NLP)** techniques were applied to transform raw text into structured numerical input suitable for machine learning and deep learning models.

The overall project workflow includes:

- **Exploratory Data Analysis (EDA):** Examining tweet length, word frequency, punctuation, and class distribution.
- **Text Preprocessing:** Removing noise such as URLs, HTML tags, special characters, and stopwords; converting to lowercase; applying lemmatization and stemming.
- **Feature Extraction:** Using **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text into feature vectors.



### 🧪 Modeling

We implemented a variety of models to evaluate performance:

#### 🧩 Classical Machine Learning Models:
- **Logistic Regression**
- **Random Forest Classifier**
- **Decision Tree Classifier**
- **Support Vector Machine (SVM)**
- **Naive Bayes (MultinomialNB)**

All models underwent **hyperparameter tuning** using `GridSearchCV` or `RandomizedSearchCV` to improve performance.

#### ⚡ Boosting Models:
- **XGBoost Classifier**
- **LightGBM Classifier**

These models use gradient boosting to improve accuracy by learning from previous errors. Both were also tuned using cross-validation techniques.

#### 🤝 Ensemble Model:
We combined the three best-performing models using a **Voting Classifier**:
- Decision Tree
- Random Forest
- Logistic Regression

This ensemble approach leverages the strengths of each model to enhance robustness and generalization.


### 🤖 Final Model – BERT

As a final step, we utilized **BERT (Bidirectional Encoder Representations from Transformers)**, a powerful pre-trained model for language understanding.

Implementation details:
- Pre-trained model: `bert-base-uncased`
- Tokenization using BERT tokenizer
- Fine-tuning with a classification head
- Trained on GPU to optimize performance

BERT significantly improved contextual understanding and helped capture nuanced meanings in tweets.


### 🏁 Evaluation Metrics

All models were evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

This allowed for comprehensive performance comparison between traditional and deep learning models.


## 📁 Folder Structure

```
📦disaster-tweets-nlp
 ┣ 📂data/
 ┃ ┣ 📜train.csv          # Training data with labels
 ┃ ┣ 📜test.csv           # Unlabeled test data for prediction
 ┃ ┗ 📜submission.csv     # Final prediction results
 ┣ 📂notebooks/
 ┃ ┣ 📜NLP-Disaster-Tweet.py       # Exported notebook as .py
 ┃ ┗ 📜NLP-Disaster-Tweet.md       # Link to full notebook in Colab
 ┣ 📜README.md
 ┗ 📜NLP_codes.ipynb (unusable on GitHub preview)
```

> 📎 **Interactive notebook:**  
> View the full Jupyter Notebook on Google Colab:  
> 👉 [Open in Colab](https://colab.research.google.com/drive/1jCxJLkmW64Db5NunZsQxrDXgszTeJfKk?usp=sharing)



## 📌 Additional Notes

- The dataset used was sourced from Kaggle and is not included here due to size and licensing restrictions.
- The `submission.csv` file includes predictions from the best-performing model.
- GPU acceleration is recommended for BERT-based training and inference to significantly reduce processing time.
