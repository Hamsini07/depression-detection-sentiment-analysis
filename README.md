# Depression Detection Using Sentiment Analysis

An NLP and Machine Learning project that analyzes social media text from Reddit and Twitter using TF-IDF and three classification models: Decision Tree, Artificial Neural Network (ANN), and K-Nearest Neighbors (KNN), with an interactive Streamlit web interface for text prediction.

## Project Overview

Depression is an important mental health concern, and social media platforms contain large amounts of user-generated text expressing people's thoughts, feelings, and experiences.

This project uses Natural Language Processing (NLP) and Machine Learning techniques to analyze social media text from Reddit and Twitter and identify depression-related language patterns and sentiment.

The project implements three supervised machine learning models: Decision Tree, Artificial Neural Network (ANN), and K-Nearest Neighbors (KNN). Text data is preprocessed and converted into numerical features using TF-IDF before model training and evaluation.

An interactive **Streamlit web interface** is also implemented, allowing users to enter text and obtain predictions using the trained machine learning models.

The project covers the complete workflow, including text preprocessing, feature extraction, model training, hyperparameter tuning, performance evaluation, model saving, and interactive prediction through the web interface.

**Note:** The Reddit dataset classifies posts as Depression or Not Depression, whereas the Twitter dataset classifies tweets as Negative or Positive sentiment. These represent different classification tasks.

## Objectives

- Analyze depression-related text from Reddit and sentiment from Twitter.
- Preprocess text data using NLP techniques.
- Convert text into numerical features using TF-IDF.
- Implement and compare Decision Tree, ANN, and KNN classifiers.
- Tune model parameters using cross-validation.
- Evaluate model performance using standard classification metrics.
- Save the best-performing model pipeline for future predictions.
- Develop an interactive Streamlit interface for user input and prediction.

## Technologies and Libraries

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Development Environment | Jupyter Notebook |
| Data Processing | Pandas, NumPy |
| Natural Language Processing | NLTK |
| Feature Extraction | TF-IDF |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Web Interface | Streamlit |
| Model Saving | Joblib |

## Datasets

The project uses two CSV datasets:

### 1. Reddit Depression Dataset
- Contains cleaned Reddit posts.
- Binary classification labels: Depression (1) and Not Depression (0).
- Uses the `clean_text` and `is_depression` columns.

### 2. Twitter Sentiment Dataset
- Contains tweets collected for sentiment classification.
- Binary classification labels: Negative (0) and Positive (1).
- Uses the `text` and `sentiment` columns.

The notebook handles missing text and labels and removes Twitter entries that become empty after preprocessing.

## Methodology

### 1. Data Preprocessing

Text preprocessing helps remove unnecessary information and prepare the text for machine learning.

The Twitter preprocessing steps include:

- **Lowercasing:** Converts all text into lowercase.
- **Regex Cleaning:** Removes numbers, punctuation, special characters, and unwanted symbols.
- **Stopword Removal:** Removes common English words that contribute less to classification.
- **Lemmatization:** Reduces words to their base forms using WordNet.

The Reddit dataset already contains cleaned text. A separate normalization function is also used for new Reddit text during prediction.

### 2. Feature Extraction – TF-IDF

TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numerical features that machine learning models can process.

The notebook uses:

- Maximum features: 5,000.
- Unigrams: Individual words.
- Bigrams: Combinations of two consecutive words.
- N-gram range: (1, 2).

This helps the models identify both individual words and meaningful word combinations.

### 3. Model Training and Hyperparameter Tuning

Three supervised machine learning models are implemented:

**Decision Tree Classifier**
- Learns classification rules from the input text features.
- Uses decision-based splitting to classify text.
- Provides visualizations that help understand the model's decisions.
- Hyperparameters such as maximum tree depth and minimum samples per leaf are tuned.

**Artificial Neural Network (ANN)**
- Implemented using Scikit-learn's `MLPClassifier`.
- Learns complex and non-linear patterns from TF-IDF features.
- Uses the ReLU activation function and Adam optimizer.
- Uses early stopping to help reduce overfitting.
- Tests different hidden-layer configurations.

**K-Nearest Neighbors (KNN)**
- Classifies text based on similarity to nearby training samples.
- Uses cosine distance to compare text feature vectors.
- Tests different values of K and weighting strategies.

**Training Configuration**
- Training and testing split: 80% and 20%.
- Stratified sampling is used to maintain class proportions.
- Random seed: 42.
- GridSearchCV with 3-fold cross-validation is used for hyperparameter tuning.

### 4. Model Evaluation

The models are evaluated using the following metrics:

- **Accuracy:** Measures the overall percentage of correct predictions.
- **Precision:** Measures how many predicted positive samples are actually positive.
- **Recall:** Measures how many actual positive samples are correctly identified.
- **F1-Score:** Combines precision and recall into a single metric.
- **Confusion Matrix:** Shows correct and incorrect predictions for each class.

The notebook also generates:

- Word clouds.
- Class distribution plots.
- Decision Tree visualizations.
- ANN learning curves.
- Model accuracy comparison charts.

## Results

The notebook generates a comparison table for all three models across both datasets.

The evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- Training time

The results are calculated when the notebook is executed, allowing the models to be compared using the same train-test split for each dataset.

The best-performing model for each dataset is selected based on test accuracy, with F1-score used to break ties.

## Streamlit Web Interface

The project includes an interactive web application developed using **Streamlit**.

The interface allows users to enter their own text and obtain predictions using the trained machine learning models.

### Key Features

- Simple and user-friendly interface.
- Text input for entering social media-style content.
- Integration with trained machine learning models.
- Generates predictions based on the selected dataset and its classification labels.
- Displays prediction results through an interactive web interface.
- Allows users to interact with the machine learning system without manually executing notebook cells.

### Running the Streamlit Application

Install Streamlit if it is not already installed:

```bash
pip install streamlit
```

Run the application using:

```bash
streamlit run app2UI.py
```

After execution, Streamlit provides a local URL through which the application can be accessed in a web browser.

## Model Saving

After evaluation, the selected model pipeline is saved using Joblib.

The complete pipeline includes TF-IDF feature extraction and the trained classifier.

The generated output files include:

| File | Description |
|---|---|
| `model.joblib` | Saved TF-IDF and classifier pipeline |
| `meta.json` | Model details, labels, preprocessing information, and evaluation metrics |
| `results.csv` | Comparison results of the trained models |

These files are stored in the `models2` directory, with separate folders for Reddit and Twitter.

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-url>
```

### Step 2: Navigate to the Project Directory

```bash
cd depression-detection-sentiment-analysis
```

### Step 3: Install the Required Libraries

```bash
pip install wordcloud nltk scikit-learn pandas numpy matplotlib seaborn joblib streamlit
```

### Step 4: Add the Datasets

Place the required CSV datasets in the location specified by the notebook.

Required files:

- `depression_dataset_reddit_cleaned.csv`
- `Twitter-Dataset-Trimmed.csv`

### Step 5: Run the Jupyter Notebook

Open the project notebook:

`Sentiment Analysis project.ipynb`

Run the notebook cells in order to preprocess the datasets, train the models, evaluate their performance, and save the trained model pipelines.

### Step 6: Run the Streamlit Interface

After the required trained model files are generated and available at the paths expected by the application, execute:

```bash
streamlit run app2UI.py
```

The application will open in your browser using the local URL provided by Streamlit.

## Limitations

- Social media text may contain slang, spelling errors, abbreviations, and ambiguous expressions.
- Public datasets may have limitations in size, labeling quality, and representativeness.
- Model performance may vary across platforms, languages, and user groups.
- The Reddit and Twitter datasets have different classification labels, so their results should not be treated as identical prediction tasks.
- Selecting the best model using test-set performance can lead to optimistic estimates. An independent final test set is recommended for more rigorous evaluation.

## Future Enhancements

- Explore transformer-based models such as BERT, RoBERTa, and MentalBERT.
- Evaluate models on larger and more diverse datasets.
- Improve the Streamlit interface with additional interactive visualizations and prediction insights.
- Improve model interpretability using explainable AI techniques.
- Assess potential bias and improve model generalization.

## Disclaimer

This project is developed for educational and research purposes only. Its predictions are based on patterns in social media text and should not be considered medical diagnoses or a replacement for professional mental health assessment.

## Team

**Team 5**

Department of Computer Science and Engineering

