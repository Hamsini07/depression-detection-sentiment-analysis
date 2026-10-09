# depression-detection-sentiment-analysis
Depression-related text analysis using NLP, TF-IDF, and machine learning models.
# Depression Detection Using Sentiment Analysis

## Project Overview
This project analyzes social media text using Natural Language Processing (NLP) and Machine Learning to identify depression-related patterns and classify sentiment. It uses two datasets, Reddit and Twitter, to compare the performance of three supervised machine learning algorithms: Decision Tree, Artificial Neural Network (ANN), and K-Nearest Neighbors (KNN).

The project includes text preprocessing, TF-IDF feature extraction, model training, hyperparameter tuning, performance evaluation, and saving trained models for future predictions.

## Objectives
- Analyze depression-related text from Reddit and sentiment from Twitter.
- Preprocess text data to make it suitable for machine learning.
- Convert text into numerical features using TF-IDF.
- Train and compare Decision Tree, ANN, and KNN classifiers.
- Evaluate model performance using standard classification metrics.
- Save the best-performing model for each dataset for future use.

## Technologies and Libraries
- **Programming Language:** Python
- **Environment:** Jupyter Notebook
- **Data Processing:** Pandas, NumPy
- **Natural Language Processing:** NLTK
- **Feature Extraction:** Scikit-learn TF-IDF Vectorizer
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn, WordCloud
- **Model Saving:** Joblib

## Datasets
The project uses two CSV datasets:

1. **Reddit Depression Dataset:** Contains cleaned social media posts classified into Depression and Not Depression.
2. **Twitter Sentiment Dataset:** Contains tweets classified into Negative and Positive sentiment.

The notebook removes rows with missing text or labels, validates the binary labels, and removes Twitter entries that become empty after preprocessing.

## Methodology

### 1. Data Preprocessing
The Twitter text is converted to lowercase, cleaned using regular expressions, filtered to retain alphabetic characters, processed to remove English stopwords, and lemmatized using WordNet. The Reddit dataset already contains cleaned text, so a separate normalization function is used for new Reddit text during prediction.

### 2. Feature Extraction
TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numerical features. The notebook uses unigrams and bigrams with a maximum of 5,000 features.

### 3. Model Training and Hyperparameter Tuning
Three machine learning models are implemented:

- **Decision Tree:** Learns classification rules from the text features.
- **Artificial Neural Network (ANN):** Uses Scikit-learn's MLPClassifier to learn complex patterns.
- **K-Nearest Neighbors (KNN):** Uses cosine distance to classify text based on nearby samples.

GridSearchCV with 3-fold cross-validation is used to tune the model parameters. The datasets are divided into 80% training data and 20% testing data using stratified sampling and random seed 42.

### 4. Model Evaluation
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The notebook also generates word clouds, class distribution plots, Decision Tree visualizations, ANN learning curves, and a model accuracy comparison chart.

## Results
The notebook generates a comparison table containing the accuracy, precision, recall, F1-score, and training time of each model on both datasets. The results are calculated when the notebook is executed, allowing the performance of the models to be compared using the same train-test split for each dataset.

## Model Saving
After evaluation, the best-performing model for each dataset is selected based on test accuracy, with F1-score used to break ties. The complete TF-IDF and classifier pipeline is saved using Joblib.

The output includes:
- `model.joblib` – Saved TF-IDF and classifier pipeline.
- `meta.json` – Model details, labels, preprocessing information, and evaluation metrics.
- `results.csv` – Model comparison results.

These files are stored in the `models2` directory, with separate folders for Reddit and Twitter.

## How to Run the Project

1. Download or clone this repository.
2. Install the required Python libraries.
3. Place the required CSV datasets in the same directory as the notebook or inside a `data` folder.
4. Open `Sentiment Analysis project.ipynb` in Jupyter Notebook.
5. Run the notebook cells in order.
6. Review the generated evaluation metrics, plots, and saved models.

### Installation
Install the main dependencies using:

```bash
pip install wordcloud nltk scikit-learn pandas matplotlib seaborn joblib
```

## Limitations
- Social media text may contain slang, spelling errors, and ambiguous expressions.
- Public datasets may have limitations in size, labeling quality, and representativeness.
- Model performance may vary across platforms, languages, and user groups.
- Selecting the best model using test-set performance can lead to optimistic estimates; an independent final test set is recommended for more rigorous evaluation.

## Future Enhancements
- Explore transformer-based models such as BERT and MentalBERT.
- Evaluate the models on larger and more diverse datasets.
- Develop and improve a Streamlit web interface for interactive predictions.
- Improve interpretability and assess potential bias in predictions.

## Disclaimer
This project is intended for educational and research purposes. Its predictions are not medical diagnoses and should not replace professional mental health assessment.

