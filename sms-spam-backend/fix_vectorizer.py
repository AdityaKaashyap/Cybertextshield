import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess(text):
    """Preprocess text same as training"""
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text.lower())
    return text.split()

# Load your original dataset
path = "E:/Cybertextshield/sms-spam-backend/spam_detector/final_spam.csv"

df = pd.read_csv(path)
df = df[df['Labels'].isin(['ham', 'spam'])].copy()

# Get messages
messages = df['Message'].tolist()

# Create vectorizer without tokenizer parameter (use default preprocessing)
vectorizer = TfidfVectorizer(max_features=500, lowercase=True, token_pattern=r'[a-zA-Z0-9]+')

# Fit the vectorizer on your messages
vectorizer.fit(messages)

# Save the vectorizer
with open('vectorizer_fixed.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Fixed vectorizer saved as 'vectorizer_fixed.pkl'")

# Test loading
with open('vectorizer_fixed.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)
    
print("Vectorizer loaded successfully!")
print(f"Vocabulary size: {len(loaded_vectorizer.vocabulary_)}")

# Test transform
test_message = "Hello this is a test message"
features = loaded_vectorizer.transform([test_message])
print(f"Features shape: {features.shape}")