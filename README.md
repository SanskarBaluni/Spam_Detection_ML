# Email Spam Classifier

## Goal

The goal of this project is to build a classifier that can predict whether a given email subject is classified as "Spam" or "Ham" (Not Spam). The classifier will be trained on a dataset containing various email subjects and their corresponding classifications.

## Implementation

The Email Spam Classifier was implemented using the following steps:

### 1. CountVectorizer

The CountVectorizer technique was applied to the text of the emails. CountVectorizer converts the text data into a numerical representation, enabling the use of machine learning algorithms that require numerical input.

### 2. Naive Bayes Classifier

The MultinomialNB method from the Naive Bayes Classifier was employed for email subject classification. Naive Bayes is a classification technique based on Bayes' Theorem, assuming independence among predictors. It assumes that the presence of a specific feature in a class is unrelated to the presence of any other feature.

## Understanding Naive Bayes

Naive Bayes is a classification technique based on Bayes' Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.

## Python Libraries Used

The following Python libraries were used for implementing the Email Spam Classifier:

- pandas
- numpy
- CountVectorizer
- MultinomialNB from sklearn

## CountVectorizer Functionality

CountVectorizer converts a collection of text documents into a matrix of token counts. This implementation produces a sparse representation of the counts using scipy.sparse.csr_matrix. If an a-priori dictionary is not provided and no feature selection is performed by the analyzer, the number of features will be equal to the vocabulary size found by analyzing the data.

  
## Usage

1. Run the web app:
<pre>
streamlit run app-Git.py
</pre>

2. Open your web browser and go to `http://localhost:8501` to access the web app.

3. Input the sms which needs to be classified.

4. Click on the **Predict** button to generate the prediction result.
