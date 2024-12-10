import streamlit as st
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz
import torch
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Function to ensure NLTK resources are downloaded
def ensure_nltk_resources(resources):
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)

# Ensure required NLTK resources are available
ensure_nltk_resources(['stopwords', 'wordnet'])

# Initialize models
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Symptom list
known_symptoms = [
    'fever', 'cold', 'runny nose', 'sneezing', 'rash', 'dizziness', 'weakness', 'loss of appetite',
    'cough', 'muscle pain', 'joint pain', 'chest pain', 'back pain', 'wrist pain', 'constipation', 'throat pain',
    'flu', 'breathlessness', 'stomach pain', 'migraine', 'ache', 'sore', 'burning', 'itching', 'swelling',
    'infection', 'inflammation', 'cramps', 'ulcers', 'bleeding', 'irritation', 'anxiety', 'depression',
    'insomnia', 'cancer', 'diabetes', 'hypertension', 'allergies', 'weight loss', 'weight gain', 'hair loss',
    'blurred vision', 'ear pain', 'palpitations', 'urinary frequency', 'numbness', 'tingling', 'night sweats',
    'dry mouth', 'excessive thirst', 'frequent urination', 'acne', 'bruising', 'confusion', 'memory loss',
    'hoarseness', 'wheezing', 'itchy eyes', 'dry eyes', 'difficulty swallowing', 'restlessness', 'yellow skin',
    'yellow eyes', 'bloating', 'gas', 'hiccups', 'indigestion', 'heartburn', 'mouth sores', 'nosebleeds',
    'ear ringing', 'decreased appetite', 'unusual sweating', 'dark urine', 'light-colored stools', 'blood in urine',
    'blood in stool', 'frequent infections', 'delayed healing', 'high temperature', 'low blood pressure', 'thirst',
    'dehydration', 'skin burn', 'sweating', 'feeling cold', 'feeling hot', 'head pressure', 'double vision',
    'eye pain', 'eye redness', 'eye discharge', 'hearing loss', 'balance problems', 'taste changes', 'smell changes',
    'rapid breathing', 'irregular heartbeat', 'chest tightness', 'lightheadedness', 'fainting', 'unsteady gait',
    'clumsiness', 'loss of coordination', 'seizures', 'tremors', 'shakiness', 'nervousness', 'panic attacks',
    'mood swings', 'irritability', 'agitation', 'difficulty concentrating', 'foggy mind', 'hallucinations', 'paranoia',
    'euphoria', 'apathy', 'lack of motivation', 'social withdrawal', 'exhaustion', 'muscle weakness', 'muscle cramps',
    'muscle stiffness', 'joint stiffness', 'bone pain', 'bone fractures', 'sprains', 'strains', 'tendonitis', 'bursitis',
    'arthritis', 'gout', 'fibromyalgia', 'sciatica', 'herniated disc', 'spinal stenosis', 'back spasms', 'neck pain',
    'whiplash', 'carpal tunnel syndrome', 'sinus pressure', 'sinus headache', 'eczema', 'psoriasis', 'hives', 'impetigo',
    'herpes', 'shingles', 'warts', 'moles', 'skin lesions', 'skin bumps', 'skin discoloration', 'skin dryness', 'skin peeling',
    'skin cracking', 'skin burning', 'skin tenderness', 'skin redness', 'skin swelling', 'skin blisters', 'hair thinning',
    'hair breakage', 'hair brittleness', 'hair shedding', 'hair graying', 'hair texture changes', 'hair growth abnormalities',
    'nail discoloration', 'nail brittleness', 'joint instability',
    'muscle atrophy', 'joint dislocation', 'joint deformity', 'bone deformity', 'bone tenderness', 'bone swelling', 'bone redness',
    'joint locking', 'joint clicking', 'joint popping', 'muscle atrophy due to disuse', 'skin dryness due to weather', 'high blood pressure',
    'skin peeling due to eczema', 'skin burning due to dermatitis', 'skin swelling due to injury', 'skin tenderness due to inflammation',
    'skin blisters due to friction', 'skin ulcers due to pressure', 'skin sores due to infection', 'skin growths due to cancer',
    'skin discoloration due to sun exposure', 'skin dryness due to dehydration', 'skin cracking due to cold weather', 'skin itching due to insect bites',
    'eye dryness', 'eye irritation', 'eye redness', 'eye swelling', 'eye tearing', 'eye strain', 'eye sensitivity to light', 'eye watering',
    'ear dryness', 'ear fullness', 'ear congestion', 'ear fluid', 'ear wax buildup', 'ear infection', 'tinnitus', 'balance disorder',
    'taste distortion', 'metallic taste', 'bitter taste', 'sweet taste alteration', 'savory taste changes', 'smell distortion',
    'reduced smell', 'increased smell', 'loss of smell', 'rapid breaths', 'heavy breathing', 'shallow breathing', 'uneven heart rate',
    'heart skipping beats'
]

# Symptom synonyms dictionary
symptom_synonyms = {
    'depression': ['sad', 'unhappy', 'low mood', 'feeling down', 'depressed', 'feeling sad'],
    'back spasm': ['back is spasming', 'back spasms', 'back spasm'],
    'headache': ['head pain', 'head hurts', 'migraine'],
    'allergies': ['allergy'],
    'knee pain': ['knee is paining', 'pain in knees'],
    # Add more symptom synonyms here
}

# Precompute embeddings
symptom_embeddings = model.encode(known_symptoms, convert_to_tensor=True)

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Helper function for text normalization
def normalize_text(text):
    """
    Normalize the input text by:
    - Converting to lowercase
    - Removing non-alphabetic characters
    - Removing stopwords
    - Lemmatizing words
    """
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # Lowercase and remove non-alphabetic characters
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Function to map synonyms to standardized symptoms
def map_synonym(user_input):
    """
    Check if any synonym of known symptoms is present in the user input.
    Returns the standardized symptom if a synonym is found, else None.
    """
    for symptom, synonyms in symptom_synonyms.items():
        for synonym in synonyms:
            # Use regular expression to match whole words
            pattern = r'\b' + re.escape(synonym.lower()) + r'\b'
            if re.search(pattern, user_input):
                return symptom
    return None

# Function to match user input to known symptoms
def match_symptom(user_input):
    """
    Matches the user input to the best-known symptom using:
    1. Synonym mapping
    2. Fuzzy matching
    3. Semantic similarity with SBERT
    """
    # Normalize user input
    normalized_input = normalize_text(user_input)
    
    # Step 1: Synonym mapping
    synonym_match = map_synonym(normalized_input)
    if synonym_match:
        return synonym_match
    
    # Step 2: Fuzzy matching on known symptoms
    fuzzy_result = process.extractOne(normalized_input, known_symptoms, scorer=fuzz.partial_ratio)
    if fuzzy_result and fuzzy_result[1] > 80:
        return fuzzy_result[0]
    
    # Step 3: Semantic similarity with SBERT
    user_embedding = model.encode(normalized_input, convert_to_tensor=True)
    cos_scores = util.cos_sim(user_embedding, symptom_embeddings)
    max_score = torch.max(cos_scores).item()
    if max_score > 0.7:
        best_match_idx = torch.argmax(cos_scores)
        return known_symptoms[best_match_idx]
    
    return "No clear match found"

# Streamlit App UI
st.title("🩺 Symptom Matcher App")
st.write("Enter a description of your symptoms, and the app will try to match it to a known symptom.")

# User input
user_input = st.text_input("Describe your symptom:")

# Match button
if st.button("Find Match"):
    if user_input.strip():
        matched_symptom = match_symptom(user_input)
        st.write(f"**Best Matched Symptom:** {matched_symptom}")
    else:
        st.warning("Please enter a symptom description.")
