# AI-Powered Customer Complaint Intelligence System

## Demo

Watch the project in action:

[▶️ Watch Demo Video](https://1drv.ms/v/c/59195917a148f4f8/IQAKq0m-c2_jQbxAumKyxby-AeeJnGJjSmnlRNQUHnJL5sQ?e=bqnSLa)

An end-to-end system that uses Machine Learning and Generative AI to automatically classify, summarize, and extract insights from customer support complaints, reducing manual effort and helping teams act faster on support data.

---

## Problem

Customer support teams receive thousands of unstructured complaints daily. Manually reviewing them is slow, inconsistent, and makes it hard to spot recurring issues before they escalate.

Key pain points:
- No easy way to identify patterns across large volumes of tickets
- Repetitive issues go unnoticed without structured categorization
- Decision-making is delayed because insights aren't surfaced automatically

---

## Solution

This system combines traditional NLP with a local LLM to handle the full pipeline , from raw complaint text to actionable insight , with a simple web UI anyone on the team can use.

**What it does:**
- Classifies each complaint into a category (billing, technical, refund, etc.)
- Generates a one-line AI summary of each complaint
- Visualizes the distribution of complaint types across a dataset
- Produces dataset-level insights powered by Llama3

---

## Why This Project Matters

- Helps companies identify recurring customer issues faster  
- Reduces manual analysis effort  
- Enables data-driven decision making  

--- 

## Approach

### 1. Text Preprocessing (NLTK)
Raw complaint text is cleaned before any model sees it , lowercasing, stopword removal, and lemmatization.

### 2. Classification + Summarization (TF-IDF + Llama3)
Each complaint is classified into a category and summarized in one line by Llama3.


<img width="1513" height="843" alt="Complaint_analysis" src="https://github.com/user-attachments/assets/d9fde56c-7a13-43dd-a0f3-e193e16fc83e" />


### 3. AI Insights and Recommendations
The model identifies the top recurring issues across the entire dataset. Based on the patterns found, Llama3 generates actionable business recommendations.

<img width="883" height="811" alt="Insights and Recommendations" src="https://github.com/user-attachments/assets/1fa0c4c3-10bf-453c-8494-ad9e7dc22c6a" />

<img width="877" height="834" alt="Insights and Recommendations (2)" src="https://github.com/user-attachments/assets/145f5c99-255d-4ad9-a2f5-d9d7b0d0a125" />

### 5. Analytics Dashboard
A visual dashboard summarizing ticket volume, resolution time, ratings, and trends.

<img width="1318" height="746" alt="Customer Support Analysis" src="https://github.com/user-attachments/assets/245b1e14-390e-4224-bb69-28270c64e769" />

---

## Features

| Feature | Description |
|---|---|
| Complaint Classification | Categorizes tickets into billing, technical, refund, etc. |
| AI Summary | One-line LLM-generated summary per complaint |
| Dataset Visualization | Bar chart of ticket type distribution |
| AI Insights | Dataset-level patterns surfaced by Llama3 |
| Simple UI | Streamlit app — no setup needed for end users |

---

## Example

**Input complaint:**
> "I have been charged twice for the same subscription and nobody from support has responded in 5 days."

**Output:**
- **Category:** Billing
- **Summary:** Customer was double-charged for a subscription and has not received a support response after 5 days.
- **Insight (dataset level):** Billing complaints account for 34% of tickets this week, with response delay being the most common sub-issue.

---
## Business Impact

This project goes beyond basic classification and summarization by helping translate customer complaints into actionable insights.

- Enables faster identification of recurring issues across large volumes of tickets  
- Helps prioritize high-impact problems that affect customer experience  
- Reduces manual effort required to analyze unstructured complaint data  
- Supports data-driven decision making for improving products and services  

In a real-world setting, such a system can assist support and operations teams in quickly understanding customer pain points and taking corrective actions.

--- 

## Tech Stack

- **Language:** Python 3.10+
- **NLP:** NLTK (tokenization, stopwords, lemmatization)
- **ML:** scikit-learn (TF-IDF, Logistic Regression)
- **LLM:** Llama3 via [Ollama](https://ollama.com/)
- **UI:** Streamlit
- **Visualization:** Matplotlib / Seaborn

---

## Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally

### 1. Clone the repository
```bash
git clone https://github.com/Raghav20-cyber/AI-Customer-Complaint-Intelligence.git
cd AI-Customer-Complaint-Intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull the Llama3 model
```bash
ollama pull llama3
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## Project Structure
```
├── app.py                  # Streamlit UI
├── classifier.py           # TF-IDF + Logistic Regression model
├── summarizer.py           # Llama3 summarization via Ollama
├── preprocessor.py         # NLTK text cleaning pipeline
├── insights.py             # Dataset-level insight generation
├── data/
│   └── cleaned_tickets.csv      # Sample dataset
├── requirements.txt
└── README.md
```

---

## Limitations & Future Work

- Currently runs Llama3 locally , cloud LLM support (OpenAI, Gemini) can be added
- Classification accuracy depends on training data quality and size
- Planned: confidence scores per classification, multi-label support, export to CSV
