# Sentiment Analysis on Movie Reviews

A comprehensive sentiment analysis project using the IMDB movie review dataset. This project involves data preprocessing, feature extraction, and binary classification to distinguish between positive and negative reviews using a Logistic Regression model. This project is fully documented and includes error handling, visualizations, and model evaluation metrics, making it a great addition to showcase your skills in machine learning, data processing, and NLP.

## Project Structure

This project is organized into well-defined steps that provide clarity and facilitate easy understanding and replication of the sentiment analysis workflow:

1. **Importing Libraries**: Import essential libraries for data processing, visualization, and modeling.
2. **Data Loading and Initial Exploration**: Load and validate the dataset, check for missing values, and ensure column consistency.
3. **Data Preprocessing**: Text cleaning and normalization, including tokenization, lemmatization, and stopword removal.
4. **Dataset Splitting**: Train-test split while maintaining class balance.
5. **TF-IDF Vectorization**: Convert text into numerical features using a TF-IDF vectorizer for modeling.
6. **Model Training and Cross-Validation**: Train a logistic regression model and evaluate it with cross-validation.
7. **Evaluation and Visualization**: Evaluate the model using accuracy, precision, recall, F1 score, ROC curve, and confusion matrix.

## Features and Techniques Used

### Data Preprocessing

- **Tokenization and Lemmatization**: Clean and standardize text by removing special characters, stopwords, and applying lemmatization.
- **Error Handling**: Implement error handling for file loading and ensure columns are in the correct format.
- **Text Vectorization**: Use TF-IDF vectorization to represent text as numerical features, capturing both unigrams and bigrams.

### Modeling

- **Model Choice**: Logistic Regression for binary classification due to its simplicity and effectiveness for text classification tasks.
- **Cross-Validation**: Evaluate the model's generalization capability with 5-fold cross-validation.
- **Model Saving**: Save the vectorizer and trained model with version control to enable reproducibility.

### Evaluation and Visualization

- **Evaluation Metrics**: Display accuracy, precision, recall, F1 score, ROC-AUC, and a detailed classification report.
- **Confusion Matrix**: Visualize classification results to understand model performance better.
- **ROC Curve**: Display the ROC curve with AUC to assess model quality.

## Results and Insights

This project demonstrates effective data preprocessing and model training for sentiment analysis. It achieves high accuracy and provides insightful visualizations for understanding model performance.

## Getting Started

1. **Install Required Libraries**: Ensure that required libraries are installed:
    ```bash
    pip install pandas numpy matplotlib seaborn nltk scikit-learn joblib
    ```

2. **Run the Jupyter Notebook**: Open each step in Jupyter Notebook and follow the comments to understand each process.

## Files

- `SentimentAnalysis.ipynb`: The main notebook file containing all code and explanations.
- `tfidf_vectorizer_*.pkl`: Saved TF-IDF vectorizer for reusability.
- `sentiment_logistic_model_*.pkl`: Saved Logistic Regression model for inference.

## Future Improvements

- **Experiment with Deep Learning**: Use deep learning models (e.g., LSTM, BERT) for potentially higher accuracy.
- **Hyperparameter Tuning**: Perform grid search for optimizing model parameters.
- **Sentiment Analysis on Unseen Data**: Evaluate the model on new datasets to test generalization.

## Acknowledgments

This project is based on the IMDB dataset. Special thanks to the open-source community for providing libraries that make such projects accessible and efficient.

---

Feel free to fork and contribute to this project!
