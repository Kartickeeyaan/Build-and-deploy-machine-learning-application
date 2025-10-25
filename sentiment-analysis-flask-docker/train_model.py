# train_model.py
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk

# Download NLTK data (required for stopwords)
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer

# ========================
# STEP 1: Load Dataset
# ========================
print("\n[STEP 1] Loading dataset...")
# Change the filename to match your dataset
df = pd.read_csv('data/IMDB Dataset.csv')  # Or 'data/sample_reviews.csv'

print(f"Dataset loaded successfully!")
print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 3 rows:")
print(df.head(3))

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# ========================
# STEP 2: Text Preprocessing
# ========================
print("\n[STEP 2] Starting text preprocessing...")

# Define stopwords
stop_words = set(stopwords.words('english'))

_word_tokenizer = TreebankWordTokenizer()


def clean_text(text):
    """
    Clean and preprocess text data
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization (use Treebank tokenizer to avoid requiring punkt)
    tokens = _word_tokenizer.tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back to string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

# Apply cleaning to all reviews
print("Cleaning text data (this may take a few minutes)...")
df['cleaned_review'] = df['review'].apply(clean_text)

print(f"\nExample of cleaned text:")
print(f"Original: {df['review'][0][:100]}...")
print(f"Cleaned: {df['cleaned_review'][0][:100]}...")

# ========================
# STEP 3: Prepare Labels
# ========================
print("\n[STEP 3] Encoding labels...")

# Convert sentiment to binary (1 for positive, 0 for negative)
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print(f"Label distribution:")
print(df['label'].value_counts())

# ========================
# STEP 4: Split Data
# ========================
print("\n[STEP 4] Splitting data into train and test sets...")

X = df['cleaned_review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Ensure balanced split
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ========================
# STEP 5: TF-IDF Vectorization
# ========================
print("\n[STEP 5] Applying TF-IDF vectorization...")

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,  # Use top 5000 features
    ngram_range=(1, 2),  # Use unigrams and bigrams
    min_df=2  # Ignore terms that appear in less than 2 documents
)

# Fit and transform training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform test data
X_test_tfidf = tfidf.transform(X_test)

print(f"TF-IDF shape: {X_train_tfidf.shape}")
print(f"Number of features: {len(tfidf.get_feature_names_out())}")

# ========================
# STEP 6: Train Model
# ========================
print("\n[STEP 6] Training Logistic Regression model...")

# Initialize and train model
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='liblinear'
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")

# ========================
# STEP 7: Evaluate Model
# ========================
print("\n[STEP 7] Evaluating model performance...")

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ========================
# STEP 8: Save Model and Vectorizer
# ========================
print("\n[STEP 8] Saving model and vectorizer...")

# Save the trained model
with open('models/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved: models/sentiment_model.pkl")

# Save the TF-IDF vectorizer
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("✓ Vectorizer saved: models/tfidf_vectorizer.pkl")

# ========================
# STEP 9: Test Predictions
# ========================
print("\n[STEP 9] Testing predictions with sample texts...")

# Test samples
test_samples = [
    "This movie is absolutely fantastic! I loved it.",
    "Terrible film. Complete waste of time.",
    "An okay movie, nothing special."
]

for sample in test_samples:
    # Clean the text
    cleaned = clean_text(sample)
    
    # Vectorize
    vectorized = tfidf.transform([cleaned])
    
    # Predict
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = max(probability) * 100
    
    print(f"\nText: {sample}")
    print(f"Prediction: {sentiment} (Confidence: {confidence:.2f}%)")

print("\n" + "="*50)
print("MODEL TRAINING COMPLETE!")
print("="*50)
