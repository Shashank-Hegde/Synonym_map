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

# Symptom list (extended with body part + pain terms)
known_symptoms = [
    'fever', 'cold', 'runny nose', 'sneezing', 'rash', 'dizziness', 'weakness', 'loss of appetite', 'headache',
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
    'heart skipping beats',
    'leg pain', 'eye pain', 'hand pain', 'arm pain', 'foot pain', 'knee pain', 'shoulder pain', 'neck pain'
]

# Symptom synonyms dictionary
symptom_synonyms = {
    # General Symptoms:
    'fever': ['elevated temperature', 'high body temperature', 'pyrexia', 'febrile state', 'raised temperature', 'chills and fever', 'body overheating'],
    'cold': ['common cold', 'viral cold', 'respiratory infection', 'head cold', 'seasonal cold', 'upper respiratory infection', 'nasal cold'],
    'runny nose': ['nasal discharge', 'drippy nose', 'clear runny nose', 'watery nose', 'rhinorrhea', 'nose leakage', 'constant dripping nose'],
    'sneezing': ['frequent sneezing', 'nose clearing', 'repetitive sneezing', 'nasal sneezing'],
    'rash': ['skin rash', 'skin irritation', 'dermatitis', 'eczema', 'hives', 'itchy rash', 'raised red bumps', 'skin inflammation'],
    'dizziness': ['feeling lightheaded', 'spinning sensation', 'balance issues', 'vertigo', 'faintness', 'feeling woozy', 'head spinning', 'swaying feeling'],
    'weakness': ['general weakness', 'muscle weakness', 'feeling drained', 'fatigue', 'lack of strength'],
    'loss of appetite': ['poor appetite', 'decreased appetite', 'appetite loss', 'reduced hunger', 'lack of desire to eat'],
    'cough': ['persistent cough', 'dry cough', 'wet cough', 'hacking cough', 'persistent throat clearing', 'productive cough', 'barking cough'],
    'muscle pain': ['muscle soreness', 'muscle tenderness', 'muscle stiffness', 'muscle strain', 'muscle ache', 'muscle discomfort', 'deep muscle pain'],
    'joint pain': ['joint discomfort', 'arthralgia', 'joint stiffness', 'joint inflammation', 'pain in the joints', 'swollen joints'],
    'chest pain': ['sharp chest pain', 'chest discomfort', 'chest pressure', 'pain in chest', 'tightness in chest', 'stabbing chest pain'],
    'back pain': ['lower back pain', 'upper back pain', 'spinal pain', 'pain in the back', 'chronic back pain', 'lumbar pain', 'muscle pain in back'],
    'wrist pain': ['wrist discomfort', 'pain in the wrist', 'carpal pain', 'wrist joint pain', 'wrist injury', 'wrist sprain'],
    'constipation': ['difficult bowel movement', 'hard stools', 'infrequent bowel movement', 'bowel obstruction', 'intestinal blockage'],
    'throat pain': ['painful throat', 'swollen throat', 'throat irritation', 'swallowing pain', 'inflamed throat'],
    'flu': ['influenza', 'seasonal flu', 'viral flu', 'flu-like illness', 'grippe'],
    'breathlessness': ['shortness of breath', 'dyspnea', 'labored breathing', 'air hunger', 'tight chest', 'shallow breathing'],
    'stomach pain': ['gastric pain', 'abdominal cramps', 'stomach cramps', 'digestive discomfort', 'bellyache'],
    'migraine': ['throbbing headache', 'severe headache', 'pounding headache', 'intense head pain'],
    #'ache': ['pain', 'discomfort', 'soreness', 'tenderness'],
    'sore': ['painful', 'tender', 'raw', 'irritated'],
    'burning': ['scalding pain', 'stinging sensation', 'fiery pain', 'tingling pain'],
    'itching': ['pruritus', 'skin irritation', 'itchy sensation', 'skin scratching'],
    'swelling': ['edema', 'fluid retention', 'puffy skin', 'swollen area', 'inflammation'],
    'infection': ['bacterial infection', 'viral infection', 'pathogenic infection', 'contagion', 'sepsis'],
    'inflammation': ['swelling', 'redness', 'heat', 'pain', 'immune response'],
    'cramps': ['muscle cramps', 'spasms', 'twinges', 'sudden pain'],
    'ulcers': ['open sores', 'lesions', 'wounds', 'erosions'],
    'bleeding': ['hemorrhaging', 'blood loss', 'blood discharge'],
    #'irritation': ['discomfort', 'agitation', 'sensitivity', 'inflammation', 'reaction'],
    'anxiety': ['nervousness', 'stress', 'unease', 'worry', 'apprehension'],
    'depression': ['sadness', 'low mood', 'despair', 'melancholy', 'downheartedness','unhappy','feeling sad', 'depressed'],
    'insomnia': ['difficulty sleeping', 'sleeplessness', 'restlessness at night', 'lack of sleep', 'sleep deprivation'],
    'cancer': ['malignant tumor', 'neoplasm', 'oncological disease'],
    'diabetes': ['high blood sugar', 'diabetic condition', 'type 1 diabetes', 'type 2 diabetes'],
    'hypertension': ['high blood pressure', 'elevated blood pressure'],
    'allergies': ['allergic reaction', 'hay fever', 'immune response', 'hypersensitivity', 'seasonal allergies'],
    'weight loss': ['unexplained weight loss', 'weight reduction', 'loss of body mass'],
    'weight gain': ['increase in weight', 'gain in body mass'],
    'hair loss': ['alopecia', 'thinning hair', 'balding', 'hair shedding'],
    'blurred vision': ['fuzzy vision', 'impaired vision', 'visual distortion', 'cloudy vision'],
    'ear pain': ['otalgia', 'ear discomfort', 'pain in the ear', 'ear pressure', 'ear ache'],
    'palpitations': ['heart palpitations', 'rapid heartbeats', 'heart racing', 'fluttering in the chest'],
    'urinary frequency': ['frequent urination', 'urinary urgency', 'increased urination'],
    'numbness': ['lack of sensation', 'tingling', 'loss of feeling', 'pins and needles'],
    'tingling': ['pins and needles', 'numbness', 'prickling sensation'],
    'night sweats': ['excessive sweating at night', 'nighttime perspiration', 'sweating during sleep'],
    'dry mouth': ['xerostomia', 'cottonmouth', 'thirsty mouth'],
    'excessive thirst': ['polydipsia', 'increased thirst'],
    'frequent urination': ['urinary frequency', 'increased urination'],
    'acne': ['pimples', 'blemishes', 'skin breakouts', 'zits'],
    'bruising': ['contusion', 'hematoma', 'skin discoloration from injury'],
    'confusion': ['disorientation', 'mental fog', 'cognitive impairment', 'lack of clarity'],
    'memory loss': ['amnesia', 'forgetfulness', 'memory impairment'],
    'hearing loss': ['no sound hearing', 'nothing listening', 'unable to hear'],
    'hoarseness': ['raspy voice', 'voice change', 'laryngeal discomfort'],
    'wheezing': ['whistling breath', 'labored breathing', 'asthmatic wheeze'],
    'itchy eyes': ['eye irritation', 'allergic eyes', 'dry eyes', 'burning eyes'],
    'dry eyes': ['ocular dryness', 'eye irritation', 'burning sensation in eyes'],
    'difficulty swallowing': ['dysphagia', 'trouble swallowing', 'painful swallowing'],
    'restlessness': ['anxiety', 'unease', 'nervousness', 'inability to relax'],
    'yellow skin': ['jaundice', 'yellowish skin tone'],
    'yellow eyes': ['scleral jaundice', 'yellowing of the eyes'],
    'bloating': ['abdominal bloating', 'swollen abdomen', 'fullness'],
    'gas': ['flatulence', 'intestinal gas', 'bloating', 'belching'],
    'hiccups': ['singultus', 'involuntary contractions of diaphragm'],
    'indigestion': ['dyspepsia', 'upset stomach', 'acid reflux', 'heartburn'],
    'heartburn': ['acid reflux', 'gastric reflux', 'burning sensation in chest'],
    'mouth sores': ['canker sores', 'oral ulcers', 'blisters in the mouth'],
    'nosebleeds': ['epistaxis', 'bleeding from the nose'],
    'ear ringing': ['tinnitus', 'ringing in the ears', 'buzzing in ears'],
    'decreased appetite': ['loss of appetite', 'reduced hunger', 'poor appetite'],
    'unusual sweating': ['excessive sweating', 'hyperhidrosis'],
    'dark urine': ['brown urine', 'tea-colored urine', 'concentrated urine'],
    'light-colored stools': ['pale stools', 'clay-colored stools'],
    'blood in urine': ['hematuria', 'bloody urine'],
    'blood in stool': ['hematochezia', 'melena'], 
    'allergies': ['allergy'],
}

# Precompute embeddings
symptom_embeddings = model.encode(known_symptoms, convert_to_tensor=True)

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Symptom keywords, body parts, and intensity words
symptom_keywords = ['pain', 'discomfort', 'ache', 'sore', 'burning', 'itching', 'tingling', 'numbness']

# Intensity words with weights
intensity_words = {
    'severe': 3, 'very': 3, 'really': 3, 'extremely': 3, 'intense': 3,
    'worse': 3, 'terrible': 3, 'horrible': 3, 'mild': 1, 'slight': 1,
    'a bit': 1, 'a little': 1, 'not too severe': 1, 'somewhat': 2,
    'moderate': 2, 'chronic': 2, 'persistent': 2, 'constant': 2, 'high': 3,
    'low': 1, 'fairly': 2, 'quite': 2
}
body_parts = [
    'leg', 'eye', 'hand', 'arm', 'head', 'back', 'chest', 'wrist', 'throat', 'stomach',
    'neck', 'knee', 'foot', 'shoulder', 'ear', 'nail', 'bone', 'joint', 'skin'
]

def normalize_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def extract_intensity(text):
    text_lower = text.lower()
    found_intensities = [(word, intensity_words[word]) for word in intensity_words if word in text_lower]
    if not found_intensities:
        return None
    # Return the intensity with the highest weight
    return max(found_intensities, key=lambda x: x[1])[0]

def extract_symptom_keywords(text):
    keywords_found = [kw for kw in symptom_keywords if re.search(r'\b' + re.escape(kw) + r'\b', text)]
    return keywords_found

def extract_body_parts_func(text):
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
    fuzzy_result = process.extractOne(normalized_input, known_symptoms, scorer=fuzz.partial_ratio)
    if fuzzy_result and fuzzy_result[1] > 85:
        return fuzzy_result[0]

    # Attempt SBERT embeddings
    user_embedding = model.encode(normalized_input, convert_to_tensor=True)
    cos_scores = util.cos_sim(user_embedding, symptom_embeddings)
    max_score = torch.max(cos_scores).item()
    if max_score > 0.6:
        best_match_idx = torch.argmax(cos_scores)
        return known_symptoms[best_match_idx]

    return None

def detect_symptoms_and_intensity(user_input):
    """
    Identify multiple symptoms and their intensities from user input.
    Steps:
    1. Split input into sentences or clauses.
    2. For each clause, attempt:
       - Synonym mapping
       - Keyword + body part combination
       - Fuzzy + SBERT fallback
    3. Extract intensity per clause or overall.
    4. Return a list of (symptom, intensity) tuples.
    """

    # Split by periods, commas, or "and"
    clauses = re.split(r'[.,;]|\band\b', user_input, flags=re.IGNORECASE)
    clauses = [c.strip() for c in clauses if c.strip()]

    results = []
    # Determine overall intensity if multiple appear, pick strongest
    overall_intensity = extract_intensity(user_input)  # Overall intensity

    for clause in clauses:
        normalized_input = normalize_text(clause)
        # Synonym mapping
        symptom_from_syn = map_synonym(clause)
        if symptom_from_syn:
            results.append((symptom_from_syn, overall_intensity))
            continue

        # Keyword + body part
        kw_found = extract_symptom_keywords(normalized_input)
        bp_found = extract_body_parts_func(normalized_input)
        if kw_found and bp_found:
            combined_symptom = f"{bp_found[0]} {kw_found[0]}"
            if combined_symptom in known_symptoms:
                results.append((combined_symptom, overall_intensity))
                continue
            else:
                # Try fuzzy or SBERT for combined
                combined_res = try_all_methods(normalize_text(combined_symptom))
                if combined_res:
                    results.append((combined_res, overall_intensity))
                    continue

        # No direct synonym or body-part+keyword match, try all methods on clause
        final_res = try_all_methods(normalized_input)
        if final_res:
            results.append((final_res, overall_intensity))

    # Deduplicate results (if needed)
    unique_results = list(dict.fromkeys(results))
    return unique_results

# Streamlit UI
st.title("🩺 Symptom Matcher App with Multiple Symptoms & Intensity")
st.write("Enter a description of your symptoms, and the app will try to match multiple known symptoms and estimate their intensity.")

user_input = st.text_input("Describe your symptom(s):")

if st.button("Find Symptoms"):
    if user_input.strip():
        matched_symptoms = detect_symptoms_and_intensity(user_input)
        if matched_symptoms:
            st.write("**Detected Symptoms:**")
            for symptom, intensity in matched_symptoms:
                if intensity:
                    st.write(f"- {symptom} (Intensity: {intensity})")
                else:
                    st.write(f"- {symptom} (Intensity: Not specified)")
        else:
            st.write("No clear match found")
    else:
        st.warning("Please enter a symptom description.")
