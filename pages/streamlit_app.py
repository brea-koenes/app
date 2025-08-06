# Import in case of missing imports
import streamlit as st
import re
import string
import nltk
import joblib
import numpy as np
import torch
import transformers
import lightgbm as lgb

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models.phrases import Phraser
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define TF-IDF preprocessing function 
def preprocess_text_for_phrasing(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return tokens

# Load pickled objects from notebook
with open('phraser.pkl', 'rb') as f:
    phraser = joblib.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = joblib.load(f)
with open('bert_scaler.pkl', 'rb') as f:
    bert_scaler = joblib.load(f)
with open('lgb_model.pkl', 'rb') as f:
    final_model = joblib.load(f)

# Load BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')
bert_model.eval()

# Define the best threshold found during model evaluation
best_thresh = 0.5

# Function to extract BERT [CLS] embedding and scale it
def get_bert_embedding(text, max_length=128):
    encoded = tokenizer(
        [text],
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to('cpu')

    with torch.no_grad():
        outputs = bert_model(**encoded)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    return bert_scaler.transform(cls_embedding)

# App introduction
st.title('Product Outage Classifier')
st.markdown(
    """
    <div style="background-color:#f7f7fa; border-radius:12px; padding:2em 2em 1em 2em; margin-bottom:2em; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
        <span style="font-size:1.15em;">
        <b>This application showcases a machine learning model designed to classify social media posts as either product outage-related or not outage-related.</b>
        <br><br>
        The app is built to support <b>Starbucks</b> in identifying customer-reported product availability issues from public sentiment data.
        <br><br>
        The interface allows users to input text, which is then processed by a trained classification model. The model returns a prediction indicating whether the post is likely referencing a product outage.
        <br><br>
        <b>Project Overview:</b>
        <ul style="margin-top:0.5em; margin-bottom:0.5em;">
            <li><b>Problem Statement:</b> Reliance on inventory data alone can miss product outages due to under-reporting by store staff. This app leverages customer sentiment to fill those gaps.</li>
            <li><b>Methodology:</b> Data preparation, model training, and evaluation using natural language processing and machine learning techniques.</li>
            <li><b>Results:</b> Evaluation metrics such as <b>F1 score, precision, and recall</b> were used to assess and optimize model effectiveness, resulting in a model that can predict product outages.</li>
        </ul>
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# Text input from user
user_input = st.text_area("Enter customer text:")

# Button to trigger prediction
if st.button('Predict'):
    if user_input:
        # TF-IDF preprocessing
        processed_text = preprocess_text_for_phrasing(user_input)
        phrased_text = phraser[processed_text]
        tfidf_text = tfidf_vectorizer.transform([' '.join(phrased_text)]).toarray()

        # BERT embedding
        bert_text = get_bert_embedding(user_input)

        # Combine features
        combined_features = np.concatenate((bert_text, tfidf_text), axis=1)

        # Predict
        proba = final_model.predict_proba(combined_features)[:, 1]
        prediction = (proba >= best_thresh).astype(int)[0]

        # Display results
        st.subheader('Prediction Results:')
        st.write(f"Predicted Probability of Outage: {proba[0]:.4f}")

        if prediction == 1:
            st.error("Prediction: Likely Product Outage")
        else:
            st.success("Prediction: No Product Outage")
    else:
        st.warning("Please enter some text to get a prediction.")
