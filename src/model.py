import pandas as pd
import re
import nltk
import ollama

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# NLP SETUP , downloading necessary resources
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# =========================================
# Cleaning function
def clean_text(text):
    text = str(text).lower()

    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\b(productpurchased|ticketid|customername|email|com)\b', '', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    words = [word for word in words if len(word) > 2]

    return " ".join(words)


# LOADING & PREPARING DATA + TRAINING MODEL
try:
    df = pd.read_csv("cleaned_tickets.csv")

    df['Ticket Subject'] = df['Ticket Subject'].fillna('')
    df['Ticket Description'] = df['Ticket Description'].fillna('')

    df['issue_text'] = df['Ticket Subject'] + " " + df['Ticket Description']
    df['cleaned_text'] = df['issue_text'].apply(clean_text)

    def fix_labels(label):
        label = str(label).lower().strip()

        if "refund" in label:
            return "refund"
        elif "billing" in label or "payment" in label:
            return "billing"
        elif "technical" in label or "bug" in label or "error" in label:
            return "technical"
        elif "account" in label or "login" in label or "access" in label:
            return "account"
        elif "product" in label or "inquiry" in label:
            return "product"

        return "other"

    df['Ticket Type'] = df['Ticket Type'].apply(fix_labels)
    df = df[df['Ticket Type'] != "other"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['Ticket Type']

    model = LogisticRegression(max_iter=300)
    model.fit(X, y)

    print("✅ Model trained successfully")

except Exception as e:
    print(f"❌ Model training failed: {e}")
    model = None
    vectorizer = None


# PREDICTION function

def predict_issue(text):
    text_lower = text.lower()

    # Rule-based override
    if any(word in text_lower for word in ["refund", "money back"]):
        return "refund issue"
    elif any(word in text_lower for word in ["payment", "billing", "deducted"]):
        return "billing issue"
    elif any(word in text_lower for word in ["login", "account", "access"]):
        return "account issue"
    elif any(word in text_lower for word in ["crash", "bug", "error"]):
        return "technical issue"

    if model is None or vectorizer is None:
        return "Model not available"

    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)

    return prediction[0]



# AI SUMMARY function
def generate_summary(text):
    try:
        response = ollama.chat(
            model='llama3',
            messages=[
                {
                    'role': 'user',
                    'content': f"Summarize this customer complaint in one short line:\n{text}"
                }
            ]
        )
        return response['message']['content']

    except Exception as e:
        return f"Error generating summary: {e}"



# AI INSIGHTS function

def generate_insights(data):
    try:
        sample = " ".join(data.astype(str).head(5))

        response = ollama.chat(
            model='llama3',
            messages=[
                {
                    'role': 'user',
                    'content': f"""
You are a business analyst.

Give:
1. Top 3 common issues
2. 1 recommendation

Complaints:
{sample}
"""
                }
            ]
        )

        return response['message']['content']

    except Exception as e:
        return f"Error generating insights: {e}"