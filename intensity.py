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
symptom_list = [
    'fever', 'cold', 'runny nose', 'sneezing', 'rash', 'dizziness', 'weakness', 'loss of appetite',
    'cough', 'muscle pain', 'joint pain', 'chest pain', 'back pain', 'constipation', 'throat pain', 'diarrhea',
    'flu', 'breathlessness', 'stomach pain', 'migraine','sore', 'burning', 'itching', 'swelling','acidity','muscle strain','wrist pain', 'vomiting', 'hip pain',
    'infection', 'inflammation', 'cramps', 'ulcers', 'bleeding', 'irritation', 'anxiety', 'depression','muscle injury', 'nausea','swollen lymph nodes',
    'insomnia', 'cancer', 'diabetes', 'hypertension', 'allergies', 'weight loss', 'weight gain', 'hair loss',
    'blurred vision', 'ear pain', 'palpitations', 'urinary frequency', 'numbness', 'tingling','increased appetite',
    'dry mouth', 'excessive thirst', 'frequent urination', 'acne', 'bruising', 'confusion', 'memory loss',
    'hoarseness', 'wheezing', 'itchy eye', 'dry eyes', 'difficulty swallowing', 'restlessness', 'yellow skin',
    'yellow eyes', 'bloating', 'gas', 'hiccups', 'indigestion', 'heartburn', 'mouth sore', 'nosebleed', 'sore throat',
    'ear ringing', 'decreased appetite', 'dark urine', 'light colored stool', 'blood in urine','skin itching',
    'blood in stool', 'delayed healing', 'high temperature', 'low blood pressure', 'thirst','allergy',
    'dehydration', 'skin burn', 'sweating', 'feeling cold', 'feeling hot', 'head pressure', 'double vision',
    'eye pain', 'red eyes', 'eye discharge', 'hearing loss', 'balance problem', 'taste changes', 'smell change','feeling full',
    'rapid breathing', 'irregular heartbeat', 'chest tightness', 'lightheadedness', 'fainting', 'unsteady gait',
    'clumsiness', 'loss of coordination', 'seizures', 'tremor', 'shakiness', 'nervousness', 'panic attack','convulsion', 'nose pain',
    'mood swing', 'agitation', 'difficulty concentrating', 'foggy mind', 'hallucination', 'paranoia','delusion', 'shortness of breath',
    'euphoria', 'apathy', 'lack of motivation', 'social withdrawal', 'exhaustion', 'muscle weakness', 'muscle cramps','anhedonia',
    'muscle stiffness', 'joint stiffness', 'bone pain', 'bone fracture', 'sprain', 'strain', 'tendonitis', 'bursitis', 'skin lump','skin lesion',
    'arthritis', 'gout', 'fibromyalgia', 'sciatica', 'herniated disc', 'spinal stenosis', 'spasm', 'neck pain','skin pain',
    'whiplash', 'carpal tunnel syndrome', 'sinus pressure', 'headache', 'eczema', 'psoriasis', 'hive', 'impetigo', 'ulcer',
    'herpes', 'shingle', 'wart', 'mole', 'skin lesions', 'skin bump', 'skin discoloration', 'skin dryness',
    'skin cracking', 'skin burning', 'skin tenderness', 'skin redness', 'skin swelling', 'skin blisters', 'hair thinning', 'injury',
    'hair texture changes', 'hair growth abnormalities',
    'nail splitting','fatigue',
    'nail melanonychia', 'nail leukonychia','dermatitis',
    'nail onychomycosis', 'nail paronychia',
    'nail ridges', 'joint instability','cellulitis',
    'muscle atrophy', 'joint dislocation', 'joint deformity', 'bone deformity', 'bone tenderness', 'bone swelling', 'bone redness',
    'joint locking', 'joint clicking', 'joint popping', 'muscle atrophy due to disuse', 'skin dryness due to weather', 'high blood pressure',
    'eye dryness', 'eye irritation', 'eye redness', 'eye swelling', 'eye tearing', 'eye strain', 'eye sensitivity to light', 'eye watering',
    'ear dryness', 'ear fullness', 'congestion', 'ear fluid', 'ear wax buildup', 'ear infection', 'tinnitus', 'balance disorder','chills',
    'taste distortion', 'taste change', 'smell distortion', 'ear ache',
    'reduced smell', 'increased smell', 'loss of smell', 'rapid breaths', 'heavy breathing', 'shallow breathing', 'uneven heart rate','smell change',
    'heart skipping beats','leg pain', 'eye pain', 'hand pain', 'arm pain', 'foot pain', 'knee pain', 'shoulder pain', 'neck pain'
]

# Symptom synonyms dictionary
symptom_synonyms = {
  'back spasm': [
        'back is spasming', 'back spasms', 'back spasm', 'spinal contraction', 'muscle cramp in back', 'tight back muscles', 'back tightening', 'muscle spasm in lower back', 'spine spasming',
        'back muscle jerk', 'severe back cramp', 'spine knot', 'muscle twitch in back', 'spinal spasm', 'back stiffening', 'lower back stiffness', 'muscle contraction in back', 'back cramping',
        'muscle twitching in spine', 'sharp back pain', 'sudden back pain', 'painful muscle contraction', 'sharp spasms in back', 'throbbing back pain', 'spine twinge', 'muscle jerking in back',
        'painful back convulsion', 'tense back', 'tightness in back', 'cramping back muscles', 'pulled back muscle', 'twisting back pain', 'cramped spine', 'jerking back muscles', 'painful back tightening',
        'spinal muscle strain', 'back tension', 'muscle knots in back', 'spine tightening pain', 'painful back stiffness', 'spinal muscles seizing', 'back spasm attack', 'muscle discomfort in back',
        'intense back strain', 'stiffened spine', 'spinal muscles hardening', 'acute back spasm', 'back contorting', 'back stiffness attack', 'back muscle tension', 'back pain with spasms'
    ],
    'headache': [
        'head pain', 'throbbing headache', 'pounding head', 'splitting headache', 'severe headache', 'migraine-like ache', 'cranial ache', 'head pressure', 'sinus headache', 'tension headache',
        'hammering pain in skull', 'aching brain', 'full-head ache', 'temple-throbbing pain', 'dull ache behind eyes', 'stabbing head sensation', 'skull-crushing pressure', 'nagging ache in head',
        'relentless cranial pounding', 'forehead-tightening discomfort', 'vice-like grip on head', 'pulsating headache', 'dull throb', 'piercing head agony', 'continuous headache hum', 'low-level head strain',
        'top-of-head soreness', 'subcranial ache', 'stabbing darts of pain in scalp', 'brain-squeezing feeling', 'top-heavy ache', 'ear-to-ear head ache', 'all-encompassing head discomfort', 'band-like pressure around head',
        'persistent noggin ache', 'head tenderness', 'scalp-aching feeling', 'sensitive head region', 'brainache', 'mind-throbbing torment', 'front-lobe pressure', 'crown-of-head tension',
        'behind-the-eyes ache', 'skull-tight discomfort', 'never-ending head throb', 'grating ache inside skull', 'sinus-pressured ache', 'temple pounding', 'brain pulsation pain', 'cephalic torment',
        'oppressive ache under cranium', 'subtle persistent ache', 'gnawing head discomfort', 'dull pounding drumbeat in head', 'hammering inside skull walls', 'unyielding head tension', 'rote ache cycling through head',
        'cranium under siege', 'deep-set head pang', 'swirling headache sensation', 'anchor-like pressure in head'
    ],
    'migraine': [
        'intense one-sided headache', 'migraine aura', 'pulsating pain in head', 'photophobia-associated headache', 'debilitating headache', 'migraine attack', 'searing half-skull ache', 'throbbing temple migraine',
        'nausea-laced head pain', 'light-sensitive head torture', 'migraine episode', 'crippling one-sided ache', 'sharp lancing head pain', 'skull-splitting half-side ache', 'throbbing migraine pulse', 'debilitating cranial assault',
        'severe sensory headache', 'disabling one-sided throb', 'catastrophic temple pounding', 'migraine-induced nausea', 'half-head agony', 'sharp lancing head pain', 'stabbing head sensation', 'pulsating migraine',
        'tension-triggered migraine', 'blinding headache', 'brain-splitting side ache', 'overwhelming migraine pressure', 'incapacitating headache event', 'shattering unilateral head pain', 'sensitive to slightest sound',
        'migraine meltdown', 'severe sensitivity headache', 'hammering half-head ache', 'aura shimmer leading to pain', 'throbbing unilateral agony', 'needle-like head stab', 'crushing half-skull sensation',
        'crippling light-triggered pain', 'migraine climate inside head', 'tidal wave of head torment', 'migraine crescendo', 'migraine-flare crisis', 'incapacitating halo of pain', 'ear-to-temple throbbing on one side'
    ],
    'allergies': [
        'allergy', 'allergic reaction', 'allergic response', 'hay fever', 'allergic rhinitis', 'pollen sensitivity', 'dust mite allergy', 'food allergy', 'skin allergy', 'seasonal allergies',
        'environmental allergies', 'allergic condition', 'allergic response to pollen', 'sensitive to allergens', 'sneezing due to allergies', 'wheezing from allergic reaction',
        'swollen nasal passages', 'runny nose from allergies', 'sinus congestion from allergies', 'allergic rashes', 'eczema flare-up', 'hives', 'itchy skin from allergens', 'swollen face from allergies',
        'respiratory allergy', 'allergic reactions in skin', 'excessive histamine release', 'redness from allergy', 'swollen throat from allergies', 'asthma attack triggered by allergens', 'increased mucus production',
        'throat irritation due to allergens', 'difficulty breathing from allergies', 'sneezing fits due to pollen', 'allergic asthma', 'seasonal allergic reactions', 'itchy nose', 'nasal discharge from allergies',
        'blocked sinuses', 'itchy throat from allergies', 'dry throat from allergies', 'allergy flare-up', 'anaphylactic reaction', 'anaphylaxis', 'allergic dermatitis', 'rashes from allergens', 'swelling of lips',
        'swollen tongue', 'red eyes from allergies', 'tearing eyes from allergies', 'itchy and watery eyes', 'difficulty in breathing due to allergens'
    ],
    'fever': [
        'high temperature', 'elevated body temperature', 'feeling feverish', 'fevering', 'running a fever', 'burning up', 'feeling internally hot', 'having a temperature', 'spiking a fever', 'febrile state',
        'raised core temperature', 'overheated body', 'intense body heat', 'thermal imbalance', 'body overheating', 'raging fever', 'heated condition', 'abnormally warm body', 'pyrexia', 'uncontrolled internal heat',
        'feeling aflame', 'body heat surging', 'hot to the touch', 'internal ignition of warmth', 'body temperature surging', 'excessive warmth inside', 'bodily heat overload', 'intense flush', 'thermometer reading high',
        'scorching internal climate', 'burning sensation from within', 'sweltering body feel', 'thermal elevation', 'heated bloodstream', 'furnace-like feeling', 'feeling like an oven', 'heat radiating under skin',
        'internal fire', 'ignited from the inside', 'excessive internal warmth', 'body boiling over', 'incendiary sensation', 'intense internal glow', 'unrelenting heat', 'blazing warmth',
        'molten interior heat', 'near boiling point', 'incapacitating heat', 'relentless feverishness', 'sizzling body temp', 'flaming sensation', 'constant burning feeling', 'heat wave inside me', 'sweating due to internal heat',
        'red-hot core', 'smoldering embers of warmth', 'furnace-like core', 'pulsating heat', 'unremitting temperature rise', 'searing body condition', 'fire coursing through veins', 'endlessly hot', 'elevated reading on the thermometer',
        'no relief from heat', 'intense internal burning', 'volcanic warmth', 'torched from inside', 'superheated body', 'radical temperature spike', 'roasting sensation', 'tropical internal climate', 'heat-induced misery',
        'stoked internal fires', 'hothouse conditions inside', 'stifling fever fire', 'blazing internal inferno', 'relentless temperature climb', 'fever wave'
    ],
    'cough': [
        'Persistent cough', 'hacking cough', 'dry cough', 'wet cough', 'productive cough (with phlegm)', 'barking cough', 'non-productive cough', 'chronic cough',
        'coughing up mucus/sputum/blood', 'irritating cough', 'scratchy cough', 'whooping cough-like sound', 'continuous throat clearing', 'raspy hacking', 'chesty cough',
        'rattling cough', 'deep-chested cough', 'shallow annoying cough', 'tickling cough', 'lingering throat hack', 'spasm-like coughs', 'throaty expulsions',
        'worrisome coughing fits', 'repetitive cough bursts', 'phlegmy hacking', 'bronchial coughing', 'stubborn cough', 'dry, tickling cough', 'persistent throat tickle',
        'strangling cough', 'wheezing cough', 'loud barking cough', 'cracking cough', 'sputum-laden cough', 'cough with gagging', 'spasmodic cough', 'stubborn dry cough',
        'overwhelming coughing sensation', 'sharp, dry cough', 'cough with sharp throat pain', 'violent coughing fits', 'painful coughing episodes', 'coughing after exertion',
        'chronic phlegm cough', 'intense wheezing cough', 'grating cough', 'wet chesty cough', 'gurgling cough'
    ],
    'sore throat': [
        'throat pain', 'scratchy throat', 'painful throat', 'burning throat', 'irritated throat', 'swollen throat', 'inflamed throat', 'throat discomfort', 'throat scratchiness',
        'raw throat', 'tight throat', 'feeling of something stuck in throat', 'hoarse throat', 'swollen tonsils', 'throat inflammation', 'red throat', 'sore and swollen throat',
        'gritty throat', 'tender throat', 'raspy throat', 'dry throat', 'throat burning sensation', 'feeling of throat swelling', 'pain on swallowing', 'raw feeling in throat',
        'difficulty swallowing', 'sore feeling when talking', 'throat soreness', 'painful swallowing', 'constant throat irritation', 'throat muscle soreness', 'tight feeling in throat',
        'throat dryness', 'itchy throat', 'burning sensation in throat', 'scratching feeling in throat', 'tenderness in throat', 'chronic throat discomfort', 'raspiness in voice',
        'feeling like throat is closing', 'constant need to clear throat', 'sore throat with hoarseness', 'dry cough with sore throat', 'sharp throat pain'
    ],
    'stomach ache': [
        'stomach pain', 'abdominal pain', 'belly ache', 'intestinal discomfort', 'cramps', 'stomach cramps', 'gastric discomfort', 'bloated feeling', 'nauseous stomach pain',
        'sharp stomach pain', 'stomach tenderness', 'sharp abdominal cramps', 'stomach upset', 'abdominal tenderness', 'intestinal bloating', 'tummy pain', 'swollen belly',
        'feeling of fullness', 'feeling heavy in stomach', 'digestive pain', 'stomach spasms', 'soreness in abdomen', 'gas in stomach', 'bloating', 'nausea and stomach ache',
        'gastric pain', 'pain after eating', 'belly discomfort', 'gurgling stomach', 'stomach churning', 'sharp abdominal pain', 'dull abdominal pain', 'stomach bloating',
        'abdominal tightness', 'cramping feeling', 'aching belly', 'painful digestion', 'nausea and bloating', 'gassy feeling', 'pain under ribs', 'discomfort after meals',
        'uncomfortable stomach', 'intestinal cramps', 'sharp pain in lower abdomen', 'feeling of indigestion', 'pain around stomach area'
    ],
    'fatigue': [
        'tiredness', 'extreme tiredness', 'exhaustion', 'weariness', 'fatigued feeling', 'lack of energy', 'physical depletion', 'mental fatigue', 'chronic tiredness',
        'drained', 'feeling wiped out', 'feeling run down', 'low energy', 'total exhaustion', 'severe fatigue', 'feeling sluggish', 'morning fatigue', 'fatigue after exertion',
        'weakness', 'debilitating tiredness', 'drowsiness', 'chronic fatigue syndrome', 'feeling lethargic', 'mental sluggishness', 'physical tiredness', 'difficulty keeping eyes open',
        'lack of vitality', 'energy depletion', 'feeling drained', 'listlessness', 'burned out', 'exhausted state', 'feeling zoned out', 'tired all the time', 'fatigued state',
        'mental exhaustion', 'constant tiredness', 'feeling sleepy', 'no motivation', 'fatigued muscles', 'endless tiredness', 'exhaustion after minimal effort', 'lethargic movements',
        'lacking strength', 'body fatigue', 'complete exhaustion', 'feeling disconnected'
    ],
    'nausea': [
        'feeling nauseous', 'upset stomach', 'queasy', 'stomach turning', 'sick feeling', 'feeling like vomiting', 'gagging sensation', 'discomfort in stomach', 'unsettled stomach',
        'vomit-like sensation', 'stomach churn', 'sick to stomach', 'nauseous feeling', 'spinning stomach', 'intense nausea', 'gagging feeling', 'feeling on the verge of throwing up',
        'uneasy stomach', 'feeling faint', 'upset belly', 'dizzy stomach', 'nauseous and dizzy', 'headache and nausea', 'intense queasiness', 'morning sickness feeling',
        'stomach discomfort', 'feeling faint with nausea', 'stomach churn', 'constant nausea', 'puking feeling', 'nausea after eating', 'feeling like you could throw up',
        'stomach upset with nausea', 'unsettled feeling in stomach', 'feeling lightheaded with nausea', 'nausea with dizziness', 'craving nausea', 'nausea from food', 'stomach unease',
        'sick feeling after meals', 'swirling stomach', 'nauseous waves', 'gag reflex activated', 'gurgling stomach with nausea'
    ],
    'dizziness': [
        'Lightheadedness', 'feeling faint', 'woozy sensation', 'spinning feeling', 'off-balance', 'unsteady', 'dizzy spells', 'giddy feeling', 'vertiginous sensation',
        'wobbly feeling', 'swaying in mind', 'head swimming', 'feeling as if room is turning', 'disoriented equilibrium', 'teetering sense', 'tipsy sensation without alcohol',
        'floating head', 'unstable ground feeling', 'swirling environment', 'sense of being on a boat', 'nauseating spin', 'loss of spatial orientation', 'drifting balance',
        'feeling like I might topple', 'wavy floor sensation', 'heady unsteadiness', 'murky equilibrium', 'airy head sensation', 'constant near-tip-over feeling', 'mental wobble',
        'feathery balance', 'gravity shifting under feet', 'dizziness waves', 'rocking sensation', 'seasick feeling on land', 'fuzzy-headed instability', 'inner ear imbalance feeling',
        'wavy-field-of-view sensation', 'lurching environment', 'faltering steadiness', 'delicately balanced but slipping', 'rubbery legs feeling', 'giddy swirl in head',
        'tilting world', 'swaying sensation', 'imbalance feeling', 'shaky equilibrium', 'floating dizziness', 'spinning sensation', 'feeling off-kilter'
    ],
    'shortness of breath': [
        'Shortness of breath', 'breathlessness', 'difficulty breathing', 'feeling air hunger', 'rapid breathing', 'fast breathing', 'shallow breathing', 'gasping for air',
        'labored breathing', 'struggling to breathe', 'tightness in chest while inhaling', 'feeling suffocated', 'can’t catch my breath', 'panting heavily', 'air feeling thin',
        'lungs working overtime', 'chest feels restricted', 'fighting for each breath', 'wheezing for air', 'strained respiration', 'feeling smothered', 'desperate for oxygen',
        'winded easily', 'constant puffing', 'breathing feels blocked', 'inhaling with effort', 'forced breathing', 'constant need to gulp air', 'sensation of drowning in open air',
        'chest heaviness on breathing', 'incomplete lung expansion', 'inadequate airflow', 'lungs not filling properly', 'needing to breathe harder', 'stuck in half-breath',
        'breath cut short', 'huffing and puffing', 'shallow panting', 'frantic search for air', 'hyperventilating feeling', 'feeling as if air is too thick', 'minimal air exchange',
        'muscle effort just to breathe', 'chest oppression', 'suffocating sensation even in open space', 'feeling strangled by lack of air', 'restrictive breathing pattern',
        'breathing feels like pushing through a straw', 'air-starved lungs', 'cannot take a deep breath', 'strained oxygen intake', 'feeling like each breath is a struggle',
        'never fully satisfied inhalation', 'gasping between words', 'needy breathing pattern', 'barely pulling in enough air', 'lungs working at half capacity', 'respiratory distress',
        'continuous short-windedness', 'feeling I can’t fully inflate lungs'
    ],
    'chest pain': [
        'chest discomfort', 'sharp chest pain', 'tightness in chest', 'aching chest', 'burning chest', 'soreness in chest', 'chest pressure', 'tight chest', 'chest heaviness',
        'stabbing chest pain', 'gripping chest pain', 'pain across chest', 'chest muscle soreness', 'sharp stabbing in chest', 'chronic chest pain', 'dull chest pain', 'mild chest ache',
        'sharp twinges in chest', 'pleuritic chest pain', 'pain in left side of chest', 'pain in right side of chest', 'pain with deep breath', 'sharp pain when coughing',
        'crushing chest pain', 'sensitive chest area', 'burning sensation in chest', 'tightness when breathing', 'sharp chest discomfort', 'radiating chest pain', 'underlying chest discomfort',
        'aching on chest movement', 'sore chest muscles', 'pain in the rib cage', 'pressure across chest', 'stabbing pain with movement', 'cramp-like chest pain', 'tension in chest',
        'stinging pain in chest', 'gripping sensation in chest', 'pain under the breastbone', 'breastbone discomfort', 'sharpness in the heart region', 'feeling of chest constriction'
    ],
    'muscle pain': [
        'muscle ache', 'muscle soreness', 'muscle strain', 'muscle discomfort', 'muscle stiffness', 'muscle tension', 'muscle fatigue', 'muscle injury', 'muscle cramps',
        'muscle spasm', 'muscle pulling', 'muscle tears', 'muscle tightness', 'muscle throbbing', 'aching muscles', 'sore muscles', 'tender muscles', 'painful muscles',
        'muscle inflammation', 'deep muscle pain', 'sharp muscle pain', 'pulling sensation in muscles', 'muscle tenderness', 'delayed onset muscle soreness (DOMS)', 'straining muscle',
        'muscle weakness', 'fatigued muscles', 'muscle stiffness after exercise', 'muscle burning', 'swollen muscles', 'muscle discomfort on movement', 'muscle ache after exertion',
        'overused muscles', 'fatigue-related muscle pain', 'chronic muscle pain', 'localized muscle pain', 'muscle strain from overuse', 'aching from tension in muscles',
        'muscle soreness from heavy lifting', 'muscle discomfort from exercise', 'muscle pain after activity', 'muscle distress', 'inflamed muscle tissue', 'muscle spasm after effort',
        'tensed muscles', 'muscle overextension', 'pain in the calves', 'pain in the upper arms', 'pain in the back muscles'
    ],
    'insomnia': [
        'difficulty sleeping', 'trouble sleeping', 'sleeplessness', 'restlessness at night', 'inability to fall asleep', 'waking up during the night', 'frequent wake-ups',
        'early morning wakefulness', 'poor sleep quality', 'sleep deprivation', 'sleep disturbance', 'trouble staying asleep', 'sleep interruptions', 'unable to sleep through the night',
        'insufficient sleep', 'lack of sleep', 'unrefreshing sleep', 'tossing and turning', 'unsettled sleep', 'sleep issues', 'chronic insomnia', 'difficulty achieving deep sleep',
        'waking up too early', 'difficulty with sleep onset', 'difficulty getting comfortable at night', 'sleep anxiety', 'sleeping problems', 'frequent nighttime awakenings', 'irregular sleep cycle',
        'poor sleep habits', 'nighttime restlessness', 'waking in the middle of the night', 'sleep deprivation symptoms', 'daytime sleepiness from poor sleep', 'sleep fragmentation',
        'restless sleep', 'persistent insomnia', 'sleep troubles', 'light sleeping', 'short sleep duration', 'restorative sleep deprivation', 'fatigue from sleeplessness',
        'waking up exhausted', 'sleep cycle disruption', 'sleep onset difficulty', 'insomnia due to stress', 'mental hyperactivity preventing sleep'
    ],
    'rash': [
        'skin rash', 'redness on skin', 'skin irritation', 'skin inflammation', 'skin breakout', 'itchy rash', 'hives', 'blotchy skin', 'skin eruption', 'skin lesions',
        'red bumps on skin', 'inflamed skin', 'patchy rash', 'discolored skin', 'raised rash', 'painful rash', 'rash with blisters', 'dry rash', 'moist rash', 'allergic rash',
        'eczema', 'psoriasis patches', 'contact dermatitis', 'hives breakout', 'heat rash', 'prickly heat', 'scaly rash', 'rash on face', 'body rash', 'rashes on arms',
        'welts on skin', 'itchy patches on skin', 'skin redness', 'chronic skin rash', 'dry, scaly rash', 'blistering rash', 'swollen rash', 'rash that won’t heal', 'rash with swelling',
        'inflamed, sore rash', 'rash with pus', 'pimple-like rash', 'rash caused by allergic reaction', 'skin irritation with swelling', 'flaky rash', 'raw skin from rash',
        'horrible itching rash', 'rashes from medication', 'painful itching on skin', 'burning sensation from rash'
    ],
    'congestion': [
        'nasal congestion', 'blocked nose', 'stuffy nose', 'clogged nasal passages', 'nasal obstruction', 'sinus congestion', 'sinus blockage', 'stuffy sinuses', 'pressure in sinuses',
        'nasal blockage', 'difficulty breathing through nose', 'swollen nasal passages', 'congested sinuses', 'nose congestion', 'nasal stuffiness', 'head congestion',
        'swelling of nasal tissues', 'difficulty breathing through nostrils', 'sinus pressure', 'stuffy feeling in head', 'congestion in sinus cavities', 'nasal stuffy feeling',
        'inflamed nasal passages', 'feeling of a blocked nose', 'swollen nostrils', 'nasal airway blockage', 'heavy feeling in head from congestion', 'sinus drainage blockage',
        'clogged airways', 'full nose', 'stuffy head', 'excess mucus in nose', 'thick mucus in nostrils', 'nasal obstruction from mucus', 'inability to breathe through nose',
        'nasal phlegm buildup', 'blocked airways', 'increased mucus production', 'congested nasal lining', 'swelling in nasal cavity', 'unpleasant nose feeling from congestion',
        'nasal fullness', 'pressure behind the eyes from congestion', 'nasal sinus blockage', 'nasal breathing difficulties'
    ],
    'runny nose': [
        'nasal discharge', 'drippy nose', 'clear runny nose', 'watery nose', 'excessive mucus secretion', 'nose dripping', 'watery nasal discharge', 'runny mucus from nose',
        'frequent nose blowing', 'excessive snot', 'thin nasal discharge', 'clear mucus', 'constant nose drip', 'streaming nose', 'watery runny nose', 'mucus dripping down from nose',
        'nose running uncontrollably', 'sticky nasal discharge', 'clear discharge from nostrils', 'frequent nasal wiping', 'constant nasal leaks', 'draining sinuses',
        'runny nose due to allergies', 'constant nasal secretions', 'wet nose', 'nose discharge', 'sinus leakage', 'flowing nose', 'uncontrolled nasal discharge',
        'persistent runny nose', 'dripping from nostrils', 'clogged but dripping nose', 'excessive mucus from nostrils', 'sniffling from a runny nose', 'constant nasal drip',
        'dripping sinuses', 'runny nose caused by cold', 'mucus continuously dripping', 'snotty nose', 'stuffy nose with runny discharge', 'chronic runny nose', 'dripping all day long'
    ],
    'sneezing': [
        'Sneezing fits', 'frequent sneezing', 'sneezing spells', 'sneezing bouts', 'sneezing attacks', 'sneezing episodes', 'uncontrollable sneezing', 'explosive sneezes',
        'repetitive sneezes', 'unstoppable nasal explosions', 'constant “Achoo!”', 'serial sneezing', 'sneeze after sneeze', 'chain-sneezing', 'nasal expulsions',
        'nasal reflex outbursts', 'convulsive sneezing', 'rapid-fire sneezes', 'machine-gun sneezing', 'persistent nasal expulsions', 'surprise sneezes', 'itching sneeze reflex',
        'tickling in nose triggering sneezes', 'staccato sneezing', 'sneeze cascades', 'recurrent sneezing', 'violent sneezing', 'spontaneous sneezes', 'sudden sneezing',
        'blasting sneezes', 'paroxysmal sneezing', 'intense sneezing', 'frequent sneezing attacks', 'sneezing with watery eyes', 'sudden fit of sneezing', 'uncontrollable nasal reflex',
        'hayfever sneezing', 'sneeze bursts', 'non-stop sneezing', 'gasping after sneezing', 'nasal reflex reactions', 'irritated sneezing', 'allergic sneezing', 'multiple sneeze cycles'
    ],
    'swollen lymph nodes': [
        'swollen glands', 'lymph node swelling', 'enlarged lymph nodes', 'swelling in neck', 'lumps in neck', 'tender lymph nodes', 'painful lymph nodes', 'swelling near jaw',
        'lymphatic swelling', 'lymph node enlargement', 'swollen glands under arms', 'underarm lymph node swelling', 'swollen neck glands', 'increased lymph node size',
        'lymphatic system swelling', 'lumps under the skin', 'swollen lymphatic glands', 'painful lumps in neck', 'inflamed lymph nodes', 'lymph node tenderness', 'neck swelling',
        'uncomfortable lumps in neck', 'tender neck lumps', 'swollen lymph glands in groin', 'swollen lymph nodes in armpit', 'painful swelling in neck', 'inflamed glands',
        'lymph node tenderness under jaw', 'enlarged glands in the throat', 'neck lumps', 'swollen lymph nodes behind ears', 'tender swollen glands', 'neck lymphatic swelling',
        'swelling in the throat', 'pain in swollen glands', 'pain from swollen lymph nodes', 'inflamed and tender lymph nodes', 'lymphatic swelling with pain'
    ],
    'joint pain': [
        'arthritis', 'joint ache', 'joint discomfort', 'joint inflammation', 'joint stiffness', 'joint tenderness', 'pain in joints', 'arthritic pain', 'swollen joint', 'joint soreness',
        'joint irritation', 'musculoskeletal pain', 'painful joints', 'joint stiffness', 'grating joint feeling', 'aching joints', 'joint tightness', 'joint swelling', 'rheumatoid pain', 'stiff joints',
        'painful knees', 'painful shoulders', 'pain in elbows', 'wrist joint pain', 'ankle joint pain', 'hip joint pain', 'persistent joint pain', 'severe joint pain', 'uncomfortable joint pressure',
        'popping joints', 'clicking joints', 'cracking joints', 'sore knees', 'joint inflammation in fingers', 'inflamed joints', 'stiffened knee joints', 'swollen ankles', 'excessive joint pain',
        'joint tenderness', 'joint soreness from strain', 'arthralgia', 'aching knees', 'hip pain', 'painful wrists', 'sharp joint pain', 'stabbing joint pain', 'chronic joint ache', 'inflamed elbow joints',
        'chronic knee pain', 'joint damage', 'strained joint', 'degenerative joint disease', 'discomfort in joints', 'dull joint ache', 'acute joint pain', 'swollen hands', 'weakening joint flexibility',
        'muscle and joint discomfort', 'continuous joint pain', 'painful back joints', 'arthritic inflammation'
    ],
   'diarrhea': [
        'loose stools', 'frequent bowel movements', 'watery stools', 'runny stools', 'loose bowels', 'urgent need to defecate', 'watery bowel movements', 'explosive diarrhea',
        'stomach upset with diarrhea', 'frequent trips to the bathroom', 'diarrhea with cramping', 'abnormal stool consistency', 'watery feces', 'fecal urgency', 'loose bowel movement',
        'urgent diarrhea', 'persistent diarrhea', 'morning diarrhea', 'stomach flu diarrhea', 'digestive distress', 'frequent liquid stools', 'runny bowel movements', 'intense bowel movements',
        'diarrheal episode', 'loose stool rush', 'urgent diarrhea attack', 'acute diarrhea', 'chronic diarrhea', 'pale watery stools', 'stomach churn with diarrhea', 'intestinal upset',
        'frequent bowel clearing', 'fluid-filled stools', 'non-stop diarrhea', 'gut infection diarrhea', 'dehydrating diarrhea', 'uncontrolled liquid stools', 'loose stool frequency',
        'constantly running to the bathroom', 'liquid-filled intestines', 'intense gastrointestinal upset', 'abnormally frequent bowel movements', 'severe bowel looseness', 'bowel irregularity',
        'involuntary liquid stools', 'gassy diarrhea', 'splashy diarrhea', 'digestive upset causing liquid stools', 'diarrhea with abdominal pain'
    ],
    'vomiting': [
        'throwing up', 'puking', 'stomach upset', 'retching', 'emesis', 'nausea with vomiting', 'forcefully throwing up', 'heaving', 'vomiting episodes', 'sick stomach',
        'gagging', 'expelling stomach contents', 'stomach expulsion', 'violent vomiting', 'repeated vomiting', 'uncontrollable vomiting', 'upchucking', 'spitting up', 'retching reflex',
        'forceful expulsion of food', 'involuntary stomach release', 'emetic response', 'feeling of needing to vomit', 'gag reflex triggering', 'chronic vomiting', 'severe vomiting',
        'nausea-induced vomiting', 'unpleasant stomach eruption', 'stomach contents expelled forcefully', 'gastrointestinal purge', 'expulsion of gastric contents', 'violent heaving',
        'nauseated vomiting', 'vomit-induced gagging', 'stomach-purging sensation', 'retching uncontrollably', 'throwing up after eating', 'puking episodes', 'sick and throwing up',
        'puking from irritation', 'regurgitating food', 'empty stomach vomiting', 'morning sickness vomiting', 'nausea attacks with vomiting', 'emesis due to motion sickness', 'heaving up'
    ],
    'wheezing': [
        'Wheezy breathing', 'whistling sound in chest', 'squeaky breath sounds', 'airway constriction noise', 'raspy inhalation/exhalation', 'high-pitched airway noise', 'strained breathing sounds',
        'pipe-like whistling in lungs', 'shrill breathing tone', 'whistle-like wheeze', 'tight-sounding inhalation', 'air squeaking through narrow passages', 'wheezy rasp', 'reedy airway noise',
        'restricted airflow whistling', 'hissing breath', 'airy shrillness', 'squealing lung sounds', 'sinusoidal airway noise', 'faint whistling undertone', 'discordant breathing note',
        'squeaky bronchial sound', 'flute-like chest murmur', 'feeble airy gasp', 'throttled breathing noise', 'subtle wind-through-a-pipe sound', 'frictional wheeze', 'squealing inhalation',
        'musical chest sound', 'constricted airway melody', 'tiny whistle at exhale', 'squeaking lung pipes', 'scratchy internal whistle', 'hush-like squeal inside chest', 'low whistling hum',
        'piping chest note', 'shrilling lung friction', 'wind instrument-like breath', 'wheezy hum inside', 'squeaky balloon-like noise', 'shrill bronchial constriction', 'pinched airway sounds',
        'skinny-tube breathing noise', 'fragile airway tone', 'forced narrow-bore breath', 'tight-lung whistle', 'faint flute-like exhale', 'windy squeak in lungs', 'diminished airway diameter sound',
        'toy whistle breathing', 'compressed bronchial passage tune'
    ],
    'ear ache': [
        'ear pain', 'pain in the ear', 'ear discomfort', 'ear irritation', 'painful ear', 'throbbing ear ache', 'sharp ear pain', 'dull ear pain', 'stabbing pain in ear', 'ringing ear pain',
        'pressure in ear', 'ear sensitivity', 'intense ear discomfort', 'itchy ear', 'swollen ear', 'ear tenderness', 'ear pulsations', 'persistent ear pain', 'ear infection pain',
        'ear tenderness', 'pain behind ear', 'soreness in ear', 'ear pressure', 'ear inflammation', 'ear ache from cold', 'stuffy ear pain', 'pain in ear canal', 'ear ache when swallowing',
        'painful inner ear', 'hearing sensitivity with pain', 'fluid in ear causing pain', 'acute ear pain', 'chronic ear ache', 'pain after water exposure', 'ear infection causing pain',
        'tender ear lobes', 'painful eardrum', 'painful earful feeling', 'pounding ear pain', 'sharp stabbing ear ache', 'swollen ear canal', 'eardrum sensitivity', 'sharp pressure sensation in ear',
        'soreness in ear cavity', 'clogged ear with pain', 'throbbing sensation in ear', 'ear ache during sleep'
    ],
    'back pain': [
        'lower back pain', 'upper back pain', 'spinal pain', 'pain in the back', 'back is paining', 'achy back', 'sharp back pain', 'dull back pain', 'severe back pain', 'chronic back pain',
        'stiff back', 'muscle soreness in back', 'pressure in lower back', 'pain between shoulder blades', 'sharp pain in spine', 'pain in back muscles', 'backache from lifting',
        'back discomfort', 'spinal discomfort', 'pain in lumbar region', 'back injury', 'radiating back pain', 'tight back muscles', 'spinal stiffness', 'lower back strain', 'back pain after exercise',
        'muscle strain in the back', 'burning sensation in back', 'intense back pain', 'nagging back pain', 'sharp stabbing pain in lower back', 'back tension', 'sore spine',
        'pinched nerve in back', 'back spasms', 'pain when bending', 'pain while standing up', 'pressure in upper back', 'burning pain in the back', 'pain in the sacral region',
        'pain with movement', 'back pain when sitting', 'lower back discomfort', 'muscular back pain', 'upper spinal discomfort', 'radiating pain down the back'
    ],
    'cold': [
        'Common cold', 'head cold', 'mild viral infection', 'slight sniffles', 'catching a cold', 'seasonal cold', 'chest cold', 'light upper respiratory infection', 'mild sniffle bug',
        'standard cold virus', 'low-grade nasal virus', 'mild runny-nose ailment', 'basic rhinovirus', 'everyday cold symptoms', 'short-term sniffles', 'routine winter bug', 'easy viral cold',
        'minor head stuffiness illness', 'typical seasonal illness', 'cold symptoms', 'stuffy nose cold', 'mild sore throat with cold', 'cold with slight fever', 'cough with cold',
        'runny nose cold', 'sneezing with cold', 'mild chest congestion', 'low-grade cold infection', 'itchy throat cold', 'general cold symptoms', 'nasal congestion from cold',
        'watery eyes with cold', 'mild head congestion', 'cold-related fatigue', 'chilly viral infection', 'upper respiratory cold', 'typical cold symptoms', 'stuffy feeling from cold',
        'cough and cold', 'runny nose from cold', 'frequent sneezing cold', 'cold-related chills', 'feeling chilled from cold', 'aching muscles with cold', 'minor fever with cold',
        'slight cold discomfort', 'cold-induced sore throat'
    ],
    'sweat': [
        'sweat', 'sweating', 'excessive sweating', 'unusual sweating', 'profuse sweating', 'drenched in sweat', 'perspiring heavily', 'sweating buckets', 'clammy sweating', 'dripping perspiration',
        'bead-like sweat on skin', 'moisture streaming down face', 'uncontrollable sweating', 'soaked in sweat', 'overactive sweat glands', 'sweaty and damp skin', 'sweat-soaked clothes',
        'constant perspiration', 'sticky sweat', 'salty perspiration', 'glistening with sweat', 'sweat trickling down spine', 'nervous sweating', 'stress-induced sweat', 'drenching perspiration',
        'sweat-laden body', 'humid feeling', 'slick skin', 'warm moisture on skin', 'sweat beads forming everywhere', 'bodily moisture overload', 'persistent dampness', 'sweaty palms and forehead',
        'rivers of sweat', 'sweat dripping off hairline', 'sweat-soaked sheets', 'nocturnal sweating', 'smelly perspiration', 'standing in a pool of sweat', 'sweat forming under arms', 'shiny perspiring face',
        'sweat running down temples', 'sweat-induced chafing', 'slick and slippery feeling', 'sweating like in a steam room', 'permanent dampness', 'sweat stains on clothing'
    ],
    'swelling': [
        'swollen area', 'inflammation', 'edema', 'swelling of body part', 'fluid retention', 'swollen body part', 'inflamed tissue', 'swollen limbs', 'puffiness', 'bloated feeling',
        'swollen joints', 'swollen ankle', 'swollen hands', 'swollen feet', 'localized swelling', 'bloating', 'swollen skin', 'swelling in legs', 'swelling due to injury', 'swollen belly',
        'swollen face', 'swollen knees', 'edematous swelling', 'painful swelling', 'swollen extremities', 'swelling from infection', 'swelling from trauma', 'swelling after surgery',
        'swelling of the face', 'swelling under the skin', 'swollen throat', 'swelling with discomfort', 'puffy hands', 'swelling after a fall', 'generalized swelling', 'swelling in eyes',
        'swelling from arthritis', 'swelling around wounds', 'enlarged tissue area', 'swelling from allergic reaction', 'swelling in body cavity', 'swelling around the joints'
    ],
    'tremors': [
        'shaking', 'shivering', 'twitching', 'involuntary movements', 'nervous shaking', 'muscle tremors', 'rhythmic shaking', 'trembling hands', 'uncontrolled muscle movement',
        'shaking limbs', 'twitchy fingers', 'uncontrolled tremor', 'flickering motion', 'trembling body', 'shaky movements', 'muscle spasms', 'jerking', 'shivering body', 'shaky hands',
        'shaking from cold', 'nervous tremors', 'trembling sensation', 'shuddering', 'uncontrollable shaking', 'flickering muscles', 'twitching eyes', 'nervous jerks', 'shaky fingers',
        'twitching limbs', 'muscle jerks', 'nervous body shakes', 'subtle muscle tremors', 'involuntary shaking', 'feeling of tremors', 'trembling body parts', 'sporadic body shaking',
        'hand shaking', 'shaky voice', 'rhythmic tremors', 'shivering fingers', 'body quivering', 'body shudders', 'shaking from anxiety'
    ],
    'chills': [
        'Shivering', 'trembling with cold', 'goosebumps', 'feeling cold inside', 'uncontrollable shaking', 'teeth chattering', 'feeling frosty', 'quivering limbs', 'body shaking from cold',
        'icy tremors', 'frigid vibrations', 'quaking with chill', 'hair standing on end', 'trembling internally', 'spasmodic shivers', 'cold-induced tremble', 'chilled to the bone',
        'freezing sensation', 'vibrating with cold', 'small uncontrollable shakes', 'persistent shuddering', 'subtle shivers', 'prickly gooseflesh', 'frost-like feeling', 'quivery muscles',
        'rattled by chill', 'shudders running down spine', 'uncontrollable cold tremors', 'shaky fingers and toes', 'rattling teeth', 'jittering from cold', 'frigid trembles',
        'cold-induced shaking', 'body frozen in chills', 'deep chills', 'numbing cold'
    ],
    'dry skin': [
        'skin dryness', 'rough skin', 'scaly skin', 'flaky skin', 'cracked skin', 'itchy dry skin', 'chapped skin', 'parched skin', 'dry patches', 'tight skin',
        'dry flaky patches', 'irritated dry skin', 'dehydrated skin', 'unhealthy skin texture', 'dull dry skin', 'dryness around elbows', 'rough patches on skin', 'rough and irritated skin',
        'dry skin flakes', 'flaky arms and legs', 'dry skin feeling', 'peeling skin', 'scaly patches', 'dryness on face', 'brittle skin', 'skin with visible dryness', 'dry skin redness',
        'skin looking parched', 'skin dehydration', 'skin feeling tight', 'extremely dry skin', 'skin patches drying out', 'rough hands', 'cracked hands', 'flaky fingers', 'dry lips',
        'dryness on feet', 'cracked heels', 'flaky scalp', 'dry patches around mouth', 'dry skin sensitivity', 'rough elbows and knees', 'desquamation', 'skin rough to touch', 'dry skin irritation',
        'dryness causing redness', 'thin and cracked skin'
    ],
    'eye pain': [
        'ocular pain', 'eye discomfort', 'pain in the eye', 'eye ache', 'sore eye', 'sharp pain in the eye', 'pain around the eyes', 'painful vision', 'pain behind the eye',
        'irritation in the eye', 'burning sensation in the eye', 'dry eye pain', 'stabbing eye pain', 'eye strain', 'pressure in the eye', 'throbbing in the eye',
        'sensitive eyes', 'eye tenderness', 'aching in the eye', 'eye inflammation', 'pulsing pain in the eye', 'intense eye discomfort', 'distorted vision from pain', 'foreign body sensation in the eye',
        'sharp eye ache', 'vision-related pain', 'severe eye pain', 'sharp stabbing pain in the eye', 'pain in the eyeball', 'tired eye pain', 'swollen eye discomfort', 'throbbing behind the eyes',
        'pain from light sensitivity', 'pain after reading', 'pain when blinking', 'gritty feeling in the eyes', 'intense eye pressure', 'pain around the eyelids', 'blurry vision with pain', 'puffy eyes with pain',
        'pain near the cornea', 'stinging pain in the eye', 'pain with redness in the eye', 'ocular discomfort', 'persistent eye pain', 'painful feeling when moving eyes', 'pressure sensation in the eyes',
        'pain from eye strain', 'pain with dry eyes'
    ],
    'ear pain': [
        'ear ache', 'pain in the ear', 'ear discomfort', 'sharp ear pain', 'throbbing ear ache', 'ear irritation', 'pressure in the ear', 'sharp pain in the ear', 'stabbing ear pain',
        'pain in the eardrum', 'intense ear ache', 'pain around the ear', 'aching ear', 'pain in the ear canal', 'tinnitus-related ear pain', 'ear pressure', 'burning pain in the ear',
        'ear tenderness', 'feeling of fullness in the ear', 'swollen ear', 'pain behind the ear', 'ear congestion', 'pain with hearing loss', 'fluid-filled ear pain', 'pain from ear infection',
        'dull ear pain', 'ear pain with headache', 'pain from wax buildup', 'ear ache with cold', 'sharp sensation in the ear', 'sore ear', 'ringing ear pain', 'pain in the outer ear',
        'pain from sinus pressure', 'pain in inner ear', 'pain from ear trauma', 'pain from water exposure', 'sensitive ear', 'painful ear canal', 'deep ear pain', 'pain in the ear after flying',
        'uncomfortable ear ache', 'hearing discomfort', 'ear pain after loud noise', 'sensitive eardrum'
    ],
    'nose pain': [
        'pain in the nose', 'nasal pain', 'sinus pain', 'stabbing pain in the nose', 'sharp nasal discomfort', 'painful sinus area', 'throbbing pain in the nose', 'blocked nose pain',
        'pain from sinusitis', 'nasal congestion pain', 'pain from cold in the nose', 'swollen nose', 'pressure in the sinuses', 'pain from a cold', 'tenderness in the nose', 'painful nostrils',
        'pain from nasal polyps', 'pain around the nostrils', 'nosebleed-related pain', 'pain with nasal drip', 'stuffy nose pain', 'pain due to allergies', 'burning sensation in the nose',
        'painful nasal congestion', 'aching nose', 'chronic nasal pain', 'pain when breathing through nose', 'nose pressure', 'pain at the bridge of the nose', 'pain in nasal cavity', 'itchy nose with pain',
        'pain from external nose injury', 'pain from sniffles', 'swelling in the nasal area', 'facial pain near the nose', 'pain from sinus congestion', 'painful nasal passages', 'pain in the septum',
        'pain from sniffing'
    ],
    'throat pain': [
        'sore throat', 'pain in the throat', 'scratchy throat', 'throat discomfort', 'painful swallowing', 'irritated throat', 'dry throat pain', 'burning sensation in the throat', 'throat tenderness',
        'throat scratchiness', 'swollen throat', 'pain from tonsillitis', 'strep throat pain', 'pain from a cold', 'inflammation of the throat', 'sore throat with fever', 'pain in the tonsils',
        'pain from acid reflux', 'hoarse throat', 'tight throat pain', 'painful voice box', 'pain in the larynx', 'pain with coughing', 'pain with dry mouth', 'pain with difficulty swallowing',
        'pain in the pharynx', 'pain with swollen glands', 'throat congestion', 'pain after excessive talking', 'pain from dry air', 'pain from smoking', 'throat dryness', 'severe throat discomfort',
        'pain after eating', 'pain from post-nasal drip', 'sore and swollen throat', 'pain from throat infection', 'swollen tonsils with pain', 'tightness in the throat', 'pain from sore mouth',
        'stabbing throat pain', 'pain when swallowing food', 'burning throat pain'
    ],
    'jaw pain': [
        'pain in the jaw', 'jaw discomfort', 'jaw ache', 'pain in the temporomandibular joint', 'TMJ pain', 'painful jaw muscles', 'pain from jaw clenching', 'tooth-related jaw pain',
        'sharp jaw pain', 'throbbing jaw pain', 'jaw tension', 'muscle pain in the jaw', 'pain from grinding teeth', 'jaw stiffness', 'pain in the lower jaw', 'pain from jaw injury', 'pain near the jawline',
        'pain around the mouth area', 'pain when chewing', 'discomfort in the jaw', 'jaw lock', 'jaw popping pain', 'pain around the ear and jaw', 'pain from dental issues', 'pain with jaw movement',
        'swollen jaw area', 'pain in the temporomandibular joint', 'facial pain near the jaw', 'pain from misaligned teeth', 'jaw pain from stress', 'jaw swelling', 'pain when yawning',
        'sharp pain in jaw joint', 'stiffness in jaw', 'dull aching jaw pain', 'pain from jaw trauma', 'pain from wisdom teeth', 'jaw clicking', 'pain in the side of the jaw', 'pain from jaw misalignment',
        'pain after jaw surgery', 'pain during biting', 'jaw discomfort while sleeping'
    ],
    'tooth pain': [
        'toothache', 'dental pain', 'pain in the tooth', 'sharp tooth pain', 'throbbing tooth pain', 'pain from cavity', 'pain from tooth infection', 'pain in the gums', 'sensitive teeth pain',
        'pain from a dental abscess', 'pain when chewing', 'pain from tooth decay', 'pain with tooth sensitivity', 'pain after dental work', 'pain in the tooth root', 'pain from tooth fracture',
        'pain from gum disease', 'tooth pressure', 'pain after eating', 'pain when brushing teeth', 'pain from wisdom teeth', 'pain in the molars', 'pain from misaligned teeth', 'dull tooth pain',
        'pain from tooth eruption', 'pain from a cracked tooth', 'pain with swollen gums', 'constant toothache', 'sharp shooting tooth pain', 'pain from tooth trauma', 'dental discomfort',
        'pain in the tooth nerve', 'pain from filling', 'gum-related tooth pain', 'pain from chipped tooth', 'pain from teeth grinding', 'pain from dental infection', 'pain from plaque buildup',
        'tooth pressure with pain', 'pain from oral sores', 'pain in upper teeth', 'severe tooth pain', 'pain in front teeth'
    ],
    'chest pain': [
        'pain in the chest', 'chest discomfort', 'tightness in chest', 'pressure in the chest', 'sharp chest pain', 'tight chest feeling', 'stabbing chest pain', 'burning chest pain', 'aching chest',
        'chest heaviness', 'pain in the breastbone', 'pain radiating from chest', 'dull chest pain', 'pain from heartburn', 'pain from acid reflux', 'pain in the ribs', 'pain in the upper chest',
        'sharp stabbing pain in chest', 'chest tightness', 'pain under the sternum', 'pain when breathing deeply', 'feeling of chest pressure', 'pain from pulmonary issues', 'heart-related chest pain',
        'sharp pain in the breastbone', 'radiating chest discomfort', 'pain when moving', 'pain from costochondritis', 'pain from muscle strain in chest', 'deep chest discomfort', 'pain from anxiety',
        'dull aching chest pain', 'pain in the upper left chest', 'pain when lying down', 'sore chest', 'pain from trauma to chest', 'persistent chest pain', 'discomfort after exercise',
        'pain in the center of the chest', 'pain from chest cold', 'pain in the chest while breathing', 'sore chest area', 'pain in the left side of the chest', 'pain from coughing', 'pain from deep breathing'
    ],
    'abdominal pain': [
        'stomach ache', 'belly pain', 'pain in the abdomen', 'stomach discomfort', 'gastric pain', 'sharp stomach pain', 'dull abdominal pain', 'cramping in the abdomen', 'bloating with pain',
        'gas pain in the abdomen', 'stabbing pain in the belly', 'abdominal cramps', 'sharp pain in the stomach area', 'pain from indigestion', 'pain after eating', 'nauseating abdominal pain',
        'pain from gas buildup', 'pressure in the stomach', 'pain from constipation', 'burning sensation in the stomach', 'distended abdomen', 'pain from ulcers', 'pain from bloating', 'pain from food intolerance',
        'sore stomach', 'pain from intestinal issues', 'gastrointestinal pain', 'tenderness in the stomach', 'pain near the navel', 'pain from diarrhea', 'stomach flu pain', 'pain in the lower abdomen',
        'pain from stress', 'feeling of fullness with pain', 'pain in the upper abdomen', 'stomach cramping', 'sharp abdominal cramps', 'nausea with stomach pain', 'abdominal swelling with pain',
        'chronic stomach pain', 'pain with digestive issues', 'pain from food poisoning', 'pain from gallbladder issues', 'pain from acid reflux'
    ],
    'knee pain': [
        'knee discomfort', 'pain in the knee', 'joint pain in the knee', 'knee ache', 'sharp knee pain', 'throbbing knee pain', 'stabbing pain in the knee', 'pain in the knee joint',
        'pain from knee injury', 'pain from knee strain', 'knee swelling', 'pain when bending knee', 'pain while walking', 'pain after exercise', 'pain from knee overuse', 'pain with knee movement',
        'pain in the kneecap', 'pain on the outer knee', 'pain on the inner knee', 'pain from arthritis', 'pain from ligament injury', 'pain from torn meniscus', 'sharp pain in the knee cap',
        'knee joint inflammation', 'pain when climbing stairs', 'pain with swelling', 'pain from running', 'pain from twisting knee', 'pain when standing up', 'knee tenderness', 'pain from patella dislocation',
        'pain with knee instability', 'pain from bursitis', 'pain with osteoarthritis', 'pain from cartilage damage', 'pain after knee surgery', 'pain in the back of the knee'
    ],
    'foot pain': [
        'pain in the foot', 'plantar pain', 'foot discomfort', 'foot ache', 'pain in the heel', 'sharp foot pain', 'throbbing foot pain', 'pain from foot injury', 'pain in the arch',
        'pain from flat feet', 'pain from bunions', 'pain in the toes', 'pain from corns', 'pain from calluses', 'pain from foot fractures', 'pain from wearing tight shoes', 'pain with walking',
        'pain from arthritis in foot', 'swollen foot', 'pain in the sole', 'pain when standing', 'pain from sprained ankle', 'pain from tendinitis', 'sharp pain in the foot arch', 'pain in foot joints',
        'pain in the ball of the foot', 'numbness in foot with pain', 'heel pain', 'pain from Morton’s neuroma', 'pain in foot after exercise', 'pain from overuse', 'pain after running', 'foot cramping pain',
        'pain after standing for long periods', 'pain in the toes after walking', 'sharp heel pain', 'foot pain from nerve issues', 'pain from diabetic neuropathy', 'foot pain from swelling', 'pain after wearing heels'
    ],
    'ankle pain': [
        'ankle discomfort', 'pain in the ankle', 'twisted ankle pain', 'pain from sprained ankle', 'swollen ankle', 'sharp ankle pain', 'throbbing pain in the ankle', 'pain when walking',
        'pain after ankle injury', 'pain from overuse', 'pain after exercise', 'pain with ankle movement', 'pain with swelling', 'pain from torn ligament', 'pain in the outer ankle', 'pain in the inner ankle',
        'pain in the ankle joint', 'pain from ligament strain', 'pain from ankle fracture', 'ankle tenderness', 'pain with ankle instability', 'pain when standing', 'sharp pain in the ankle',
        'pain in ankle tendon', 'pain after running', 'pain from ankle arthritis', 'pain with twisting', 'pain in ankle after jumping', 'pain in the Achilles tendon', 'stabbing pain in ankle',
        'pain with ankle sprain', 'ankle bruising', 'pain when walking on uneven surfaces', 'pain when bending the foot', 'pain in the heel of the ankle', 'pain during sports activities', 'pain when stretching ankle'
    ],
     'wrist pain': [
        'pain in the wrist', 'wrist discomfort', 'carpal pain', 'wrist ache', 'pain in the wrist joint', 'wrist inflammation', 'swollen wrist', 'stiff wrist pain', 'pain from repetitive strain',
        'tenderness in the wrist', 'sharp wrist pain', 'throbbing wrist pain', 'pain after wrist injury', 'pain from wrist overuse', 'wrist sprain pain', 'pain in the wrist tendons', 'wrist joint stiffness',
        'pain with wrist movement', 'pain during wrist flexion', 'pain in wrist ligaments', 'carpal tunnel syndrome pain', 'pain from arthritis in the wrist', 'pain with wrist rotation', 'pain in the wrist after typing',
        'wrist discomfort from injury', 'pain around wrist bones', 'dull wrist pain', 'pain when lifting objects', 'pain in the wrist after exertion', 'pain from wrist fractures', 'wrist tendonitis pain',
        'pain after hand movements', 'pain in the wrist when gripping', 'pain with wrist extension', 'stiff wrist from overuse', 'sharp sensation in the wrist', 'pain after extended typing', 'pain with wrist bending',
        'swollen joints in the wrist', 'wrist pain with numbness', 'pain around wrist bones after activity', 'pain in the carpal region', 'wrist discomfort with tingling sensation', 'pain after sports activity'
    ],
    'hand pain': [
        'hand discomfort', 'pain in the hand', 'aching hand', 'sharp hand pain', 'throbbing hand pain', 'pain in hand joints', 'pain in the palm of the hand', 'pain in the fingers', 'pain in the knuckles',
        'pain from hand injury', 'swollen hand', 'pain with hand movement', 'hand strain pain', 'numbness in the hand', 'pain after gripping', 'pain in the thumb', 'pain from arthritis in the hand',
        'pain from repetitive motions', 'pain from carpal tunnel syndrome', 'pain from hand overuse', 'hand joint pain', 'pain from hand sprain', 'pain from tendonitis in the hand', 'pain in the wrist with hand use',
        'sharp pain when holding objects', 'burning pain in the hand', 'painful hand cramps', 'dull hand pain', 'pain from hand fracture', 'pain from swelling in the hand', 'joint stiffness in the hand',
        'pain from typing', 'pain after using the hand excessively', 'pain when stretching the hand', 'pain in hand from trauma', 'pain when writing', 'pain with hand dexterity', 'pain in the hand after exercise',
        'muscle pain in the hand', 'pain from cold in the hand', 'pain after lifting objects'
    ],
    'arm pain': [
        'pain in the arm', 'upper limb pain', 'arm discomfort', 'sharp arm pain', 'throbbing arm pain', 'pain in the elbow', 'pain in the shoulder', 'pain in the forearm', 'pain in the biceps',
        'pain from arm injury', 'pain from repetitive arm movement', 'pain from tendonitis in the arm', 'muscle pain in the arm', 'nerve pain in the arm', 'pain from elbow strain', 'pain in the upper arm muscles',
        'pain from arm sprain', 'pain in the wrist and arm', 'stiffness in the arm', 'swollen arm', 'pain when moving the arm', 'burning pain in the arm', 'aching in the arm', 'arm cramping',
        'pain from lifting with the arm', 'pain when raising the arm', 'pain from overuse of the arm', 'pain from arm fracture', 'pain in the arm muscles after exercise', 'pain from muscle strain in the arm',
        'pain from joint inflammation', 'sharp pain in the arm muscles', 'pain in the elbow joint', 'pain in the shoulder joint', 'dull arm pain', 'pain in the forearm when lifting', 'shooting arm pain',
        'nerve-like pain in the arm'
    ],
    'leg pain': [
        'pain in the leg', 'lower limb pain', 'leg discomfort', 'muscle pain in the leg', 'pain in the thigh', 'pain in the calf', 'pain in the knee', 'pain in the shin', 'pain from leg injury',
        'sharp leg pain', 'throbbing leg pain', 'aching leg pain', 'pain in the leg muscles', 'pain in the leg joints', 'pain when walking', 'pain from leg cramps', 'pain after leg exercise',
        'pain after running', 'pain from overuse', 'pain in the hamstring', 'pain from leg sprain', 'muscle soreness in the leg', 'pain in the calf after activity', 'pain from leg fractures',
        'burning pain in the leg', 'pain from restless legs', 'pain when standing', 'pain in the thigh after sitting', 'pain in the foot and leg', 'pain with leg movement', 'pain from sciatica',
        'leg pain from sitting too long', 'pain when bending the leg', 'pain in the shin muscles', 'swollen leg', 'pain from arthritis in the leg', 'dull pain in the leg', 'sharp pain in the lower leg',
        'pain when walking on uneven ground', 'pain in the lower back and leg'
    ],
    'confusion': [
        'disorientation', 'muddled thinking', 'mental fog', 'trouble thinking clearly', 'brain fog', 'cognitive cloudiness', 'puzzled state', 'jumbled thoughts', 'incoherent reasoning', 'tangled mental process',
        'unclear comprehension', 'befuddled mind', 'scrambled logic', 'perplexed state', 'hazy understanding', 'blurred mental picture', 'fuzzy reasoning', 'perplexity', 'baffled intellect',
        'uncertain grasp', 'foggy mental landscape', 'clouded judgment', 'unclear headspace', 'mixed-up thoughts', 'lack of mental clarity', 'distorted perspective', 'murky understanding',
        'minds in knots', 'head scrambled eggs feeling', 'no clear thread of thought', 'haphazard reasoning', 'bewildered stance', 'lost mental bearings', 'mental haze', 'unclear mental signals',
        'vague cognitive process', 'mental static', 'mentally adrift', 'diluted focus', 'no sharpness in mind', 'blinking confusion', 'unsure mental footing', 'perplexed awareness',
        'reduced mental acuity', 'messy mental white noise'
    ],
    'hip pain': [
        'pain in the hip', 'hip discomfort', 'hip joint pain', 'pain from hip arthritis', 'sharp hip pain', 'throbbing hip pain', 'pain in the hip joint', 'pain in the hip area', 'pain from hip injury',
        'hip inflammation', 'pain from hip strain', 'pain from bursitis in the hip', 'pain when moving the hip', 'pain from overuse of the hip', 'pain in the groin area', 'pain during walking',
        'pain from hip fracture', 'pain after standing for a long time', 'pain from hip surgery', 'pain when lying on the hip', 'pain with hip rotation', 'pain in the side of the hip',
        'pain in the front of the hip', 'pain from sciatica', 'sharp hip joint pain', 'pain from hip flexor strain', 'deep hip pain', 'pain from hip dislocation', 'aching hip pain',
        'pain from muscle strain around the hip'
    ],
    'back pain': [
        'lower back pain', 'upper back pain', 'spinal pain', 'pain in the back', 'back discomfort', 'sharp back pain', 'muscle pain in the back', 'pain in the lumbar region', 'pain in the thoracic region',
        'pain in the back muscles', 'dull back pain', 'throbbing back pain', 'pain when bending over', 'pain when standing', 'pain from sciatica', 'pain from a herniated disc', 'pain from poor posture',
        'pain from back injury', 'sharp pain in the lower back', 'pain from spinal stenosis', 'pain in the upper back', 'pain in the back after lifting', 'pain from overuse of back muscles',
        'chronic back pain', 'pain from spinal degeneration', 'aching back muscles', 'back pain with tingling', 'pain from back strain', 'pain when sitting for too long', 'swollen back muscles',
        'pain from muscle spasms in the back', 'pain with back movement', 'sharp shooting pain in the back'
    ],
    'muscle pain': [
        'muscle soreness', 'muscle ache', 'muscle tenderness', 'muscle fatigue', 'muscle strain', 'muscle injury', 'aching muscles', 'muscle cramps', 'tender muscles', 'sore muscles',
        'muscle inflammation', 'muscle tightness', 'muscle discomfort', 'muscle spasms', 'pain from overworked muscles', 'muscle stiffness', 'muscle pull', 'muscle weakness', 'muscle stiffness from exercise',
        'muscle soreness after workout', 'muscle pain from exercise', 'deep muscle pain', 'pain from muscle overload', 'chronic muscle pain', 'dull muscle ache', 'sharp muscle pain', 'muscle knots',
        'muscle tension', 'pain from stretching muscles', 'tired muscles', 'pain from muscle injury during sports', 'fatigued muscles', 'swollen muscles', 'pain from repetitive muscle movement',
        'muscle ache after physical activity', 'muscle pain from lifting weights'
    ],
    'memory loss': [
        'forgetfulness', 'difficulty recalling', 'poor memory', 'memory lapses', 'amnestic episodes', 'short-term memory issues', 'difficulty remembering recent events', 'blanking out on details',
        'slip of the mind', 'fuzzy recollections', 'failing memory', 'losing track of thoughts', 'can’t recall names', 'vacant mental storage', 'holes in memory', 'patchy recollection',
        'vanishing details from mind', 'gaps in remembrance', 'fleeting mental notes', 'mental blanks', 'fragmented memory', 'elusive past events', 'stuttering memory', 'hazy recall',
        'details fading away', 'mental erasures', 'unstable memory bank', 'shaky recollections', 'selective forgetfulness', 'mental blackouts', 'fuzzy mental snapshots', 'misplacing thoughts',
        'memory glitches', 'jumbled recall', 'inability to summon certain facts', 'feeling brain-drained', 'memory going dark', 'fragments of information missing', 'scattering of remembered info',
        'ghost-like recollections', 'losing the thread of events', 'names escaping me', 'scrambled memory patterns', 'drifting mental records', 'unreliable mental archives', 'evaporation of recent info',
        'dimming recollection', 'disintegrating memory', 'thinning retention', 'leaky mental container', 'short-circuited memory', 'mental fade-outs', 'mental sputtering', 'forgetting simple things',
        'intangible memories slipping away', 'memory weakening over time', 'groping for details', 'elusive truths once known', 'mental book pages going blank', 'unstable mental files',
        'dulled memory edges', 'uncertain memory foothold', 'eroded recollections', 'falling out of my mind', 'scattering mental fragments', 'temporary amnesia-like moments', 'no access to recent thoughts',
        'memory wires disconnected', 'stuttering recollection attempts', 'defragmented mental records', 'shaky mental camera', 'fading mental impressions', 'mind like a sieve', 'losing info instantly',
        'rattled mental library', 'concept slip-through', 'flickering data in mind', 'barren mental shelves', 'no retrieval of recent facts', 'thinking it’s on the tip of my tongue but never surfacing',
        'losing track of recent conversations', 'difficulty holding new info', 'memory short-circuits frequently', 'mental vacancy', 'ephemeral recollections', 'passing mental clouds with no retention',
        'drifting away from details', 'no anchor to past events'
    ],
    'hearing loss': ['damaging hearing', 'loss in hearing'],
   'hallucination': [
        'visual disturbance', 'illusion', 'seeing things', 'auditory hallucinations', 'sensory distortion', 'false perception', 'psychotic episode', 'delusion', 'perceptual misinterpretation', 'false vision',
        'seeing unreal things', 'hearing voices', 'mind tricks', 'imagined sights', 'fictitious perception', 'phantom sensations', 'visual or auditory illusion', 'perceptual disorder', 'seeing illusions',
        'false images', 'confused perceptions', 'distorted reality', 'seeing non-existent objects', 'hallucinatory experience', 'visual hallucinations', 'mental mirages', 'mind-created visions', 'cognitive disorientation',
        'illusionary sights', 'out of body perception', 'distorted vision', 'unreal sensory inputs', 'dream-like experience', 'delirium', 'brain-generated images', 'unseen figures', 'fantasy perception',
        'mind-induced voices', 'mind-generated images', 'psychosis-related perception', 'perception delusions', 'hallucinated sounds', 'out of touch with reality', 'auditory delusion', 'seeing the impossible',
        'false sensory perception', 'misinterpreted reality', 'altered state of perception', 'phantasmagoria', 'delusional thoughts', 'imaginative perception', 'delirious hallucinations', 'unfounded sensations',
        'falsified sensory experience', 'experiencing the non-existent'
    ],
    'vomiting': ['throwing up', 'puking', 'stomach upset'],
    'ear ache': ['ear pain', 'otalgia', 'ear discomfort', 'pain in the ear', 'ear pressure'],
    'hearing loss' : ['loss of hearing'],
    'taste change' : []'taste distortion'],
    'allergies' : []'allergy'],
    'weight gain': ['increase in weight', 'gain in body mass'],

   }


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
    """
    From a single clause, find all intensity words and pick the highest.
    Also handle multi-word intensity words like "really bad".
    """
    text_lower = text.lower()
    found_intensity = None
    found_value = 0

    # Check multi-word intensities first
    for phrase, val in intensity_words.items():
        # phrase could be multi-word
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
    if fuzzy_result and fuzzy_result[1] > 80:
        return fuzzy_result[0]

    # Attempt SBERT embeddings
    user_embedding = model.encode(normalized_input, convert_to_tensor=True)
    cos_scores = util.cos_sim(user_embedding, symptom_embeddings)
    max_score = torch.max(cos_scores).item()
    if max_score > 0.7:
        best_match_idx = torch.argmax(cos_scores)
        return symptom_list[best_match_idx]

    return None

def remove_redundant_symptoms(symptoms):
    """
    Remove any symptom that is a substring of a longer symptom.
    """
    # Sort symptoms by length in descending order
    sorted_symptoms = sorted(symptoms, key=len, reverse=True)
    filtered = []
    for sym in sorted_symptoms:
        if not any(sym in existing_sym for existing_sym in filtered):
            filtered.append(sym)
    return filtered

def detect_symptoms_in_clause(clause):
    """
    For a given clause:
    1. Check synonyms directly
    2. Check body part + symptom keyword combination
    3. Fallback to fuzzy or SBERT
    4. Remove any general symptom keywords that are part of a more specific symptom
    Returns a list of matched symptoms.
    """
    results = []
    normalized_input = normalize_text(clause)

    # Synonym match
    synonym_match = map_synonym(clause)
    if synonym_match:
        results.append(synonym_match)

    # Body part + keyword
    kw_found = extract_symptom_keywords_clause(normalized_input)
    bp_found = extract_body_parts_clause(normalized_input)
    if kw_found and bp_found:
        # For each body part and each keyword found, attempt a combo
        for bp in bp_found:
            for kw in kw_found:
                combined_symptom = f"{bp} {kw}"
                if combined_symptom in symptom_list:
                    results.append(combined_symptom)
                else:
                    # Try fuzzy / SBERT on combined
                    combined_res = try_all_methods(normalize_text(combined_symptom))
                    if combined_res:
                        results.append(combined_res)

    # Fallback to general symptom detection
    if not results:
        final_res = try_all_methods(normalized_input)
        if final_res:
            results.append(final_res)
    return list(set(results))
    # Remove redundant symptoms that are substrings of longer symptoms
    filtered_results = remove_redundant_symptoms(results)
    return list(set(filtered_results))  # unique symptoms

def detect_symptoms_and_intensity(user_input):
    """
    Steps:
    1. Split input into clauses.
    2. For each clause, detect intensity and symptoms.
    3. Map the highest intensity in that clause to all symptoms found in that clause.
    """
    clauses = re.split(r'[.,;]|\band\b', user_input, flags=re.IGNORECASE)
    clauses = [c.strip() for c in clauses if c.strip()]

    final_results = []
    for clause in clauses:
        # Extract intensity from clause
        intensity_word, intensity_value = extract_intensities_in_clause(clause)
        symptoms = detect_symptoms_in_clause(clause)

        # Assign the intensity to each symptom found in this clause
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
