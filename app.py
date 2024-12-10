import streamlit as st
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz
import torch
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are downloaded
nltk_resources = ['stopwords', 'wordnet']
for resource in nltk_resources:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

# Download NLTK stopwords if not present
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Load the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Symptom list
known_symptoms = [
    'fever', 'cold', 'runny nose', 'sneezing', 'rash', 'dizziness', 'weakness', 'loss of appetite',
    'cough', 'muscle pain', 'joint pain', 'chest pain', 'back pain', 'wrist pain', 'constipation', 'throat pain',
    'flu', 'breathlessness', 'stomach pain', 'migraine', 'ache', 'sore', 'burning', 'itching', 'swelling',
    'infection', 'inflammation', 'cramps', 'ulcers', 'bleeding', 'irritation', 'anxiety', 'depression',
    'insomnia', 'cancer', 'diabetes', 'hypertension', 'allergies', 'weight loss', 'weight gain', 'hair loss',
    'blurred vision', 'ear pain', 'palpitations', 'urinary frequency', 'numbness', 'tingling', 'yellow eyes', 'yellow skin',
    # Add more symptoms as needed
]

# Precompute embeddings
symptom_embeddings = model.encode(known_symptoms, convert_to_tensor=True)

# Helper functions for text normalization
def normalize_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # Lowercase and remove non-alphabetic characters
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Match symptom
def match_symptom(user_input):
    normalized_input = normalize_text(user_input)

    # Try fuzzy matching first
    result = process.extractOne(normalized_input, known_symptoms, scorer=fuzz.partial_ratio)
    if result:
        best_match, score = result[0], result[1]
        if score > 80:
            return best_match
        else:
            # Fall back to SBERT embeddings
            user_embedding = model.encode(normalized_input, convert_to_tensor=True)
            cos_scores = util.cos_sim(user_embedding, symptom_embeddings)
            max_score = torch.max(cos_scores).item()
            if max_score > 0.7:
                best_match_idx = torch.argmax(cos_scores)
                return known_symptoms[best_match_idx]
            else:
                return "No clear match found"
    else:
        return "No clear match found"

# Streamlit UI
st.title("Symptom Matcher")
st.write("Enter a description of your symptoms, and the app will try to match it to a known symptom.")

user_input = st.text_input("Describe your symptom:")
if st.button("Find Match"):
    if user_input.strip():
        matched_symptom = match_symptom(user_input)
        st.write(f"**Best Matched Symptom:** {matched_symptom}")
    else:
        st.warning("Please enter a symptom description.")
