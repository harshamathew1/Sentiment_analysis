# Sentiment Analysis on Airline Reviews
## Overview
This project performs sentiment analysis on airline reviews to predict the sentiment (positive, negative, or neutral) and identify key topics that influence user feedback. The project uses a combination of traditional machine learning techniques and advanced deep learning models, including BERT and a custom-developed neural network, to achieve sentiment classification and topic modeling.
## Dataset
The dataset used in this project is stored in an Excel file, reviews.xlsx. It contains customer reviews with various features related to airline services, such as Overall_Rating, Review, Value For Money, and more.
## Project Structure
1. Data Loading and Exploration: The dataset is loaded, and initial exploration is done to 
   understand its structure and contents.
2. Data Preprocessing:
* Missing values are handled.
* Text data is cleaned by removing stopwords, lemmatization, and tokenization.
* Features that are not relevant to sentiment analysis are dropped.
3. Sentiment Analysis:
* VADER Sentiment Analysis: Preprocessed text is analyzed using VADER to label sentiments.
* Logistic Regression: A logistic regression model is trained to predict sentiment based on TF-IDF 
   vectorized text.
* BERT Sentiment Analysis: A BERT model is trained and fine-tuned for sentiment prediction.
4. Topic Modeling:
* Latent Dirichlet Allocation (LDA): LDA is used to identify topics within the reviews.
* BERTopic: BERTopic is applied for topic modeling using a BERT-based sentence transformer and 
  UMAP for dimensionality reduction.
5. Custom Neural Network Model: A neural network model is developed from scratch for sentiment prediction using TF-IDF features.
6. Visualization: Various plots and word clouds are generated to visualize the sentiment distribution and topic modeling results.
## Input and Output
### Input
* The primary input is the reviews.xlsx file containing airline reviews.
* The script expects text data in the Review column for sentiment analysis.
### Output
* Predicted sentiment for each review.
* Identified topics from the reviews.
* Various plots visualizing sentiment distribution and topics.
## Parameters in Functions
### Data Preprocessing
* clean_text(text): Cleans the review text by removing non-alphanumeric characters, stopwords, 
  and performing lemmatization.
* text: A string containing the review text.
* Returns: Cleaned text as a string.
## Logistic Regression Model
* LogisticRegression: Scikit-learn's logistic regression model is used with balanced class 
  weights.
* X_train, X_test: TF-IDF vectorized text data.
* y_train, y_test: Encoded sentiment labels.
* Returns: Predicted sentiment labels and model evaluation metrics.
## BERT Sentiment Analysis
* ReviewsDataset(reviews, labels, tokenizer, max_len): Custom dataset class for BERT input.
* reviews: List of review texts.
* labels: List of sentiment labels.
* tokenizer: BERT tokenizer.
* max_len: Maximum token length for BERT input.
* Returns: Encoded inputs for BERT.
## Topic Modeling
* display_topics(model, feature_names, no_top_words): Displays the top words in each topic 
  identified by LDA.
* model: Fitted LDA model.
* feature_names: Names of features from the TF-IDF vectorizer.
* no_top_words: Number of top words to display per topic.
* Returns: Printed topics with top words.
## Custom Neural Network Model
* model.fit(X_train, y_train): Trains the custom neural network model.
* X_train: Training data (TF-IDF vectors).
* y_train: Training labels (encoded sentiments).
* Returns: Trained model and training history.
## Dependencies
The project requires the following python libraries: pandas, numpy, matplotlib, seaborn, nltk, scikit-learn, torch(PyTorch), transformers, bertopic, wordcloud, and umap-learn.
