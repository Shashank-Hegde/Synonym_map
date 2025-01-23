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
symptom_list = [
'fever', 'cold', 'runny nose', 'sneezing', 'rash', 'back spasm', 'dizziness', 'weakness', 'loss of appetite', 'cough', 'muscle pain', 'joint pain',
'chest pain', 'back pain', 'constipation', 'throat pain', 'diarrhea', 'flu', 'shortness of breath', 'rapid breathing', 'stomach pain', 'migraine',
'skin burning', 'itching', 'swelling', 'vomiting', 'infection', 'inflammation', 'cramp', 'bleeding', 'irritation', 'anxiety', 'depression','congestion',
'nausea', 'swollen lymph nodes', 'insomnia', 'cancer', 'diabetes', 'allergy', 'weight loss', 'weight gain', 'hair loss', 'blurred vision', 'ear pain',
'numbness', 'dry mouth', 'frequent urination', 'acne', 'confusion', 'memory loss', 'difficulty swallowing', 'restlessness', 'bloating',
'gas', 'indigestion', 'heartburn', 'mouth sore', 'nosebleed', 'ear ringing', 'dark urine', 'blood in urine', 'blood in stool', 'high blood pressure',
'low blood pressure', 'excessive thirst', 'dehydration', 'skin burning', 'sweat', 'eye pain',  'eye discharge', 'ear discharge', 'jaundice',
'hearing loss', 'balance problem', 'irregular heartbeat', 'fainting', 'tremor', 'nervousness', 'panic attack', 'mood swing', 'difficulty concentrating',
'hallucination', 'lack of motivation', 'exhaustion', 'bone pain', 'wrist pain', 'sprain', 'strain', 'arthritis', 'gout', 'headache', 'injury', 'chills','mouth pain',
'leg pain', 'hand pain', 'arm pain', 'foot pain', 'knee pain', 'shoulder pain', 'hip pain', 'jaw pain', 'tooth pain','sleepy', 'bone fracture','sleepy','back bone issue',
'female issue', 'thyroid', 'piles', 'asthma','pneumonia','sugar',
# weakness symtom
'eye weakness','leg weakness'
  #'yellow eyes', 'red eyes'
]

# Symptom synonyms dictionary
symptom_synonyms = {
   }
# NEW CODE COMMENT: Words to exclude from mapping to symptoms through fuzzy/embedding
filtered_words = ['got', 'old']  # We can add more words here if needed

# Precompute embeddings
symptom_embeddings = model.encode(symptom_list, convert_to_tensor=True)

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Symptom keywords, body parts, and intensity words
symptom_keywords = ['pain', 'discomfort', 'ache', 'sore', 'burning', 'itching', 'tingling', 'numbness', 'trouble']

# Intensity words with assigned percentages
intensity_words = {
    'horrible': 100, 'terrible': 95, 'extremely':90, 'very':85, 'really':85, 'worse':85, 'intense':85, 'severe':80,
    'quite':70, 'high':70, 'really bad':70, 'moderate':50, 'somewhat':50, 'fairly':50, 'trouble':40,
    'mild':30, 'slight':30, 'a bit':30, 'a little':30, 'not too severe':30, 'low':20, 'continuous': 60, 'persistent': 60, 'ongoing': 60, 'constant': 60, 'a lot':70,
}
body_parts = [
    'leg', 'eye', 'hand', 'arm', 'head', 'back', 'chest', 'wrist', 'throat', 'stomach',
    'neck', 'knee', 'foot', 'shoulder', 'ear', 'nail', 'bone', 'joint', 'skin','abdomen',
    #Add more
    'mouth', 'nose', 'mouth', 'tooth', 'tongue', 'lips', 'cheeks', 'chin', 'forehead',
    'elbow', 'ankle', 'heel', 'toe', 'finger', 'thumb', 'palm', 'fingers', 'soles',
    'palms', 'fingertips', 'instep', 'calf', 'shin', 'ankle', 'heel', 'toes', 'fingers',
    'fingertips', 'instep', 'calf', 'shin', 'heel', 'toes', 'fingertips', 'instep', 'calf', 'shin',
    'lumbar', 'thoracic', 'cervical', 'gastrointestinal', 'abdominal', 'rectal', 'genital',
    'urinary', 'respiratory', 'cardiac', 'pulmonary', 'digestive', 'cranial', 'facial',
    'ocular', 'otologic', 'nasal', 'oral', 'buccal', 'lingual', 'pharyngeal', 'laryngeal',
    'trigeminal', 'spinal', 'peripheral', 'visceral', 'biliary', 'renal', 'hepatic'
]

# NEW CODE COMMENT: Symptoms that must only be detected if their exact word or synonyms are found
strict_symptoms = ['itching']
#strict_symptoms = []

# Words to exclude from mapping to symptoms through fuzzy/embedding
filtered_words = ['got', 'old']  # We can add more words here if needed

# Precompute embeddings
symptom_embeddings = model.encode(symptom_list, convert_to_tensor=True)

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Symptom keywords, body parts, and intensity words
symptom_keywords = ['pain', 'discomfort', 'ache', 'sore', 'burning', 'itching', 'tingling', 'numbness', 'trouble']

# Intensity words with assigned percentages
intensity_words = {
    'horrible': 100, 'terrible': 95, 'extremely':90, 'very':85, 'really':85, 'worse':85, 'intense':85, 'severe':80,
    'quite':70, 'high':70, 'really bad':70, 'moderate':50, 'somewhat':50, 'fairly':50, 'trouble':40,
    'mild':30, 'slight':30, 'a bit':30, 'a little':30, 'not too severe':30, 'low':20, 'continuous': 60, 'persistent': 60, 'ongoing': 60, 'constant': 60, 'a lot':70,
}
body_parts = [
    'leg', 'eye', 'hand', 'arm', 'head', 'back', 'chest', 'wrist', 'throat', 'stomach',
    'neck', 'knee', 'foot', 'shoulder', 'ear', 'nail', 'bone', 'joint', 'skin','abdomen',
    #Add more
    'mouth', 'nose', 'mouth', 'tooth', 'tongue', 'lips', 'cheeks', 'chin', 'forehead',
    'elbow', 'ankle', 'heel', 'toe', 'finger', 'thumb', 'palm', 'fingers', 'soles',
    'palms', 'fingertips', 'instep', 'calf', 'shin', 'ankle', 'heel', 'toes', 'fingers',
    'fingertips', 'instep', 'calf', 'shin', 'heel', 'toes', 'fingertips', 'instep', 'calf', 'shin',
    'lumbar', 'thoracic', 'cervical', 'gastrointestinal', 'abdominal', 'rectal', 'genital',
    'urinary', 'respiratory', 'cardiac', 'pulmonary', 'digestive', 'cranial', 'facial',
    'ocular', 'otologic', 'nasal', 'oral', 'buccal', 'lingual', 'pharyngeal', 'laryngeal',
    'trigeminal', 'spinal', 'peripheral', 'visceral', 'biliary', 'renal', 'hepatic'
]

def normalize_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def extract_intensities_in_clause(text):
    text_lower = text.lower()
    found_intensity = None
    found_value = 0

    # Check multi-word intensities first
    for phrase, val in intensity_words.items():
        if re.search(r'\b' + re.escape(phrase) + r'\b', text_lower):
            if val > found_value:
                found_value = val
                found_intensity = phrase

    return found_intensity, found_value if found_intensity else (None, 0)

def extract_symptom_keywords_clause(text):
    keywords_found = [kw for kw in symptom_keywords if re.search(r'\b' + re.escape(kw) + r'\b', text)]
    return keywords_found

def extract_body_parts_clause(text):
    body_parts_found = [bp for bp in body_parts if re.search(r'\b' + re.escape(bp) + r'\b', text)]
    return body_parts_found

def map_synonym(user_input):
    for symptom, synonyms in symptom_synonyms.items():
        for synonym in synonyms:
            pattern = r'\b' + re.escape(synonym.lower()) + r'\b'
            if re.search(pattern, user_input.lower()):
                return symptom
    return None

def try_all_methods(normalized_input):
    # Attempt fuzzy matching
    fuzzy_result = process.extractOne(normalized_input, symptom_list, scorer=fuzz.partial_ratio)
    candidate_symptom = None
    if fuzzy_result and fuzzy_result[1] > 80:
        candidate_symptom = fuzzy_result[0]
    else:
        # Attempt SBERT embeddings only if fuzzy not successful
        user_embedding = model.encode(normalized_input, convert_to_tensor=True)
        cos_scores = util.cos_sim(user_embedding, symptom_embeddings)
        max_score = torch.max(cos_scores).item()
        if max_score > 0.7:
            best_match_idx = torch.argmax(cos_scores)
            candidate_symptom = symptom_list[best_match_idx]

    # If candidate_symptom is due to a filtered word, discard it
    if candidate_symptom:
        for fw in filtered_words:
            if re.search(r'\b' + re.escape(fw) + r'\b', normalized_input):
                if fuzz.ratio(fw, candidate_symptom) > 70:
                    return None

    return candidate_symptom

def remove_redundant_symptoms(symptoms):
    sorted_symptoms = sorted(symptoms, key=len, reverse=True)
    filtered = []
    for sym in sorted_symptoms:
        if not any(sym in existing_sym for existing_sym in filtered):
            filtered.append(sym)
    return filtered

# NEW CODE COMMENT: General function to decide if a symptom should be added based on strict rules
def should_add_symptom(symptom, clause):
    # If symptom is in strict_symptoms, verify synonyms appear directly
    if symptom in strict_symptoms:
        if map_synonym(clause) == symptom:
            return True
        else:
            return False
    else:
        # If not a strict symptom, no special check needed
        return True

def detect_symptoms_in_clause(clause):
    results = []
    normalized_input = normalize_text(clause)

    # Synonym match
    synonym_match = map_synonym(clause)
    if synonym_match:
        # Check if allowed to add (in case synonym_match is strict)
        if should_add_symptom(synonym_match, clause):
            results.append(synonym_match)

    # Body part + keyword
    kw_found = extract_symptom_keywords_clause(normalized_input)
    bp_found = extract_body_parts_clause(normalized_input)
    if kw_found and bp_found:
        for bp in bp_found:
            for kw in kw_found:
                combined_symptom = f"{bp} {kw}"
                if combined_symptom in symptom_list:
                    if should_add_symptom(combined_symptom, clause):
                        results.append(combined_symptom)
                else:
                    combined_res = try_all_methods(normalize_text(combined_symptom))
                    if combined_res and should_add_symptom(combined_res, clause):
                        results.append(combined_res)

    # Fallback to general symptom detection
    if not results:
        final_res = try_all_methods(normalized_input)
        if final_res and should_add_symptom(final_res, clause):
            results.append(final_res)

    filtered_results = remove_redundant_symptoms(results)
    return list(set(filtered_results))

def detect_symptoms_and_intensity(user_input):
    clauses = re.split(r'[.,;]|\band\b', user_input, flags=re.IGNORECASE)
    clauses = [c.strip() for c in clauses if c.strip()]

    final_results = []
    for clause in clauses:
        # Extract intensity from clause
        intensity_word, intensity_value = extract_intensities_in_clause(clause)
        symptoms = detect_symptoms_in_clause(clause)

        for sym in symptoms:
            if intensity_word:
                final_results.append((sym, intensity_word, intensity_value))
            else:
                final_results.append((sym, None, 0))

    return final_results

# Streamlit UI
st.title("🩺 Multi-Symptom & Intensity Matcher")
st.write("Enter a description of your symptoms. The system will extract multiple symptoms, determine their intensities, and show a percentage for intensity.")

user_input = st.text_input("Describe your symptom(s):")

if st.button("Find Symptoms"):
    if user_input.strip():
        matched_symptoms = detect_symptoms_and_intensity(user_input)
        if matched_symptoms:
            st.write("**Detected Symptoms:**")
            for (symptom, intensity_word, intensity_value) in matched_symptoms:
                if intensity_word:
                    st.write(f"- {symptom} (Intensity: {intensity_word} ~ {intensity_value}% )")
                else:
                    st.write(f"- {symptom} (Intensity: Not specified)")
        else:
            st.write("No clear match found")
    else:
        st.warning("Please enter a symptom description.")
