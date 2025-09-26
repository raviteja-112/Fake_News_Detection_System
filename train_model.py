import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""

    # Remove HTML entities and special characters
    text = re.sub(r'&[^\s;]+;', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    # Lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_data(df):
    """Preprocess the entire dataset"""
    print("Preprocessing data...")

    # Combine title and text
    df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

    # Apply preprocessing
    df['processed_text'] = df['combined_text'].apply(preprocess_text)

    # Remove empty texts
    df = df[df['processed_text'].str.len() > 0]

    return df

def train_model():
    """Train the fake news detection model"""
    print("Loading data...")
    df = pd.read_csv('News.csv')

    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['class'].value_counts()}")

    # Preprocess data
    df = preprocess_data(df)

    # Prepare features and target
    X = df['processed_text']
    y = df['class']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Vectorize text
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    print("Training model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Calculate detailed metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    accuracy = accuracy_score(y_test, y_pred)
    precision_fake = precision_score(y_test, y_pred, pos_label=0)
    precision_real = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'precision_fake': precision_fake,
        'precision_real': precision_real,
        'recall': recall,
        'f1_score': f1
    }
    joblib.dump(metrics, 'model_metrics.pkl')

    # Save model and vectorizer
    print("Saving model and vectorizer...")
    joblib.dump(model, 'fake_news_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

    print("Model training completed!")

    return df

if __name__ == "__main__":
    df = train_model()
