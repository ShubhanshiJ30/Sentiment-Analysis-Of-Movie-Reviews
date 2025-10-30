## Project Description: Sentiment Analysis on IMDB Movie Reviews

### Introduction
This project focuses on sentiment analysis using a dataset of IMDB movie reviews. The goal is to classify reviews as either positive or negative based on the text content. We implement and compare various machine learning models to achieve this.

### Dataset
The dataset consists of 50,000 movie reviews from IMDB, evenly split into 25,000 positive and 25,000 negative reviews. Each review is labeled with a sentiment.

### Steps and Methods

1. **Data Exploration and Visualization:**
   - Explored the dataset structure and visualized sentiment distribution using plots.

2. **Data Preprocessing:**
   - Cleaned and preprocessed review text by removing HTML tags, URLs, special characters, and stopwords.
   - Tokenized and stemmed words using NLTK.
   - Checked for and removed duplicate entries in the dataset.

3. **Exploratory Data Analysis (EDA):**
   - Generated word clouds to visualize common words in positive and negative reviews.
   - Counted and displayed the most frequent words in both positive and negative reviews using bar plots.

4. **Feature Extraction:**
   - Used TF-IDF Vectorizer to convert text data into numerical features for model training.

5. **Model Training and Evaluation:**
   - Split the dataset into training and testing sets.
   - Trained machine learning models such as Logistic Regression and Linear Support Vector Classifier (SVC).
   - Evaluated models using accuracy score, confusion matrix, and classification report.

6. **Hyperparameter Tuning:**
   - Used Grid Search CV to find the best parameters for the Linear SVC model.

### Results

- **Logistic Regression:**
  - Achieved an accuracy of 89.10%.

- **Linear Support Vector Classifier:**
  - Achieved an accuracy of 89.11%.

- **Grid Search CV for Hyperparameter Tuning:**
  - Found the best parameters for Linear SVC (`{'C': 1, 'loss': 'hinge'}`) with a cross-validation accuracy of 89.56%.

- **Linear Support Vector Classifier with Best Hyperparameters:**
  - Achieved an accuracy of 89.56%.

### Conclusion
This project successfully implemented sentiment analysis on IMDB movie reviews, demonstrating effective preprocessing techniques and feature extraction using TF-IDF. The Linear SVC model with tuned hyperparameters provided the best performance with an accuracy of 89.56%. These results highlight the capability of machine learning in classifying sentiment based on text data.
