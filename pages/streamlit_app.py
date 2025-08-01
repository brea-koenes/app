import streamlit as st
import re
import string
import nltk
import joblib
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models.phrases import Phraser
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define preprocessing function (same as in notebook)
def preprocess_text_for_phrasing(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return tokens

# Load pickled objects (trained on preprocessed & phrased tokens)
with open('phraser.pkl', 'rb') as f:
    phraser = joblib.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = joblib.load(f)
with open('final_model.pkl', 'rb') as f:
    final_model = joblib.load(f)

# Define the best threshold found during model evaluation
best_thresh = 0.5

# Streamlit App Layout
st.title('Product Outage Classifier')
st.write('Enter customer text to predict if it indicates a product outage.')

# Text input from user
user_input = st.text_area("Enter text here:")

if st.button('Predict'):
    if user_input:
        # Preprocess the input text
        processed_text = preprocess_text_for_phrasing(user_input)
        phrased_text = phraser[processed_text]
        tfidf_text = tfidf_vectorizer.transform([' '.join(phrased_text)]).toarray()

        # Make prediction
        proba = final_model.predict_proba(tfidf_text)[:, 1]
        prediction = (proba >= best_thresh).astype(int)[0]

        # Display results
        st.subheader('Prediction Results:')
        st.write(f"Predicted Probability of Outage: {proba[0]:.4f}")

        if prediction == 1:
            st.error("Prediction: Likely Outage")
        else:
            st.success("Prediction: No Outage")
    else:
        st.warning("Please enter some text to get a prediction.")

st.markdown("---")
st.write("Note: This is a demonstration app. The accuracy depends on the model's performance and the quality of the input text.")
