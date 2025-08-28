import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data (you can replace this with real examples)
sample_texts = [
    "This is a human-written sentence.",
    "AI-generated texts may be harder to detect.",
    "This text was created by a real person.",
    "Machine learning models generate coherent text.",
    "This is another human-written example."
]

# Initialize and train the vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(sample_texts)

# Save the vectorizer
current_directory = os.path.dirname(os.path.abspath(__file__))
joblib.dump(tfidf_vectorizer, os.path.join(current_directory, 'tfidf_vectorizer.joblib'))

print("✅ tfidf_vectorizer.joblib has been saved.")
