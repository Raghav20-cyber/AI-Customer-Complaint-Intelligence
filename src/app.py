import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import generate_insights, predict_issue, generate_summary


# TITLE
st.title(" AI-Powered Customer Complaint Intelligence System")
st.write("Upload dataset or analyze customer complaints")


#  FILE UPLOAD
uploaded_file = st.file_uploader("Kindly Upload your dataset", type=["csv"])

df = None  # default

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset loaded successfully ✅")
    except Exception as e:
        st.error(f"Error loading file: {e}")


# CUSTOMER COMPLAINT ANALYSIS
st.subheader(" Customer Complaint Analysis")

user_input = st.text_area("Enter Customer Complaint:")

if st.button("Analyze Complaint"):
    if user_input:
        category = predict_issue(user_input)
        summary = generate_summary(user_input)

        st.subheader(" Results")
        st.write("**Category:**", category)
        st.write("**Summary:**", summary)
    else:
        st.warning("Please enter a complaint")


#  DATASET INSIGHTS
st.subheader(" Dataset Insights")

if df is not None:
    st.write("Preview of Data:")
    st.write(df.head())

    # Chart                
    if 'Ticket Type' in df.columns:
        st.subheader(" Ticket Type Distribution")

        fig, ax = plt.subplots()
        df['Ticket Type'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Column 'Ticket Type' not found in dataset")

    #  AI Insights      
    if st.button("Generate Insights"):
        with st.spinner("Generating AI insights..."):
            insights = generate_insights(df['cleaned_text'])
        st.success(insights)

else:
    st.warning("Please upload a dataset to see insights")