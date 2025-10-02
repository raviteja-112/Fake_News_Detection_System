import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from groq_analyzer import GroqNewsAnalyzer, hybrid_predict

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

# Load dataset for visualizations
@st.cache_data
def load_data():
    df = pd.read_csv('News.csv')
    return df

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

def predict_news(text, model, vectorizer):
    """Predict if news is fake or real"""
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0]

    return prediction, probability

def main():
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="wide"
    )

    st.title("📰 Fake News Detection System")
    st.markdown("---")

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Choose a page",
        ["News Prediction", "Dataset Visualization"]
    )

    if page == "News Prediction":
        show_prediction_page()
    else:
        show_visualization_page()

def show_prediction_page():
    st.header("🔍 News Article Analysis")

    st.markdown("""
    Paste a news article below to check if it's likely to be fake or real news.
    The system uses a hybrid ML + LLM approach for enhanced accuracy and explanations.
    """)

    # Text input
    news_text = st.text_area(
        "Enter news article text:",
        height=300,
        placeholder="Paste your news article here..."
    )

    # Analysis options
    use_llm = st.checkbox("Enable LLM Analysis (slower but more accurate)", value=True)

    if st.button("Analyze News", type="primary"):
        if news_text.strip():
            with st.spinner("Analyzing..."):
                model, vectorizer = load_model()

                if use_llm:
                    # Initialize Groq analyzer
                    try:
                        groq_analyzer = GroqNewsAnalyzer()
                        result = hybrid_predict(news_text, model, vectorizer, groq_analyzer, confidence_threshold=0.7)
                    except Exception as e:
                        st.warning(f"LLM analysis failed: {e}. Using ML only.")
                        result = hybrid_predict(news_text, model, vectorizer, groq_analyzer=None)
                else:
                    result = hybrid_predict(news_text, model, vectorizer, groq_analyzer=None)

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    if result["final_prediction"] == 0:
                        st.error("🚨 **FAKE NEWS DETECTED**")
                        st.markdown(f"**Confidence:** {result['final_confidence']*100:.1f}%")
                    else:
                        st.success("✅ **REAL NEWS**")
                        st.markdown(f"**Confidence:** {result['final_confidence']*100:.1f}%")

                    # Show which model was used
                    if result["used_llm"]:
                        st.info("🤖 **Hybrid Analysis Used** (ML + LLM)")
                    else:
                        st.info("⚡ **Fast ML Analysis Used**")

                with col2:
                    st.markdown("### Prediction Details")

                    # Calculate probabilities for each class
                    if result["final_prediction"] == 0:  # FAKE prediction
                        fake_probability = result['final_confidence']
                        real_probability = 1 - result['final_confidence']
                    else:  # REAL prediction
                        real_probability = result['final_confidence']
                        fake_probability = 1 - result['final_confidence']

                    st.markdown(f"**Fake Probability:** {fake_probability*100:.1f}%")
                    st.markdown(f"**Real Probability:** {real_probability*100:.1f}%")

                    # Progress bars - show likelihood for each class
                    st.markdown("**Fake News Likelihood:**")
                    st.progress(float(fake_probability))

                    st.markdown("**Real News Likelihood:**")
                    st.progress(float(real_probability))

                # Show LLM explanation if available
                if result["used_llm"] and result["llm_analysis"]:
                    st.markdown("---")
                    st.subheader("🤖 LLM Analysis & Explanation")

                    llm = result["llm_analysis"]
                    st.markdown(f"**LLM Prediction:** {'FAKE' if llm['prediction'] == 0 else 'REAL'}")
                    st.markdown(f"**LLM Confidence:** {llm['confidence']*100:.1f}%")

                    with st.expander("📝 Detailed Explanation", expanded=True):
                        st.write(llm["explanation"])

                    if llm["key_indicators"]:
                        st.markdown("**Key Indicators:**")
                        st.write(llm["key_indicators"])

                # Show ML details
                with st.expander("⚙️ Technical Details"):
                    st.markdown(f"**ML Prediction:** {'FAKE' if result['ml_prediction'] == 0 else 'REAL'}")
                    st.markdown(f"**ML Confidence:** {result['ml_confidence']*100:.1f}%")
                    if result["used_llm"]:
                        st.markdown("**Analysis Method:** Hybrid (ML + LLM)")
                    else:
                        st.markdown("**Analysis Method:** ML Only")

        else:
            st.warning("Please enter some text to analyze.")

def show_visualization_page():
    st.header("📊 Dataset Analysis & Visualization")

    df = load_data()

    st.markdown(f"**Dataset Overview:** {len(df)} articles")

    # Class distribution
    st.subheader("Class Distribution")
    class_counts = df['class'].value_counts()
    class_counts.index = class_counts.index.map({0: 'Fake', 1: 'Real'})

    fig, ax = plt.subplots()
    ax.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
           colors=['#ff6b6b', '#4ecdc4'])
    ax.set_title("Fake vs Real News Distribution")
    st.pyplot(fig)

    # Model Performance Metrics
    st.subheader("🤖 Model Performance Metrics")

    try:
        # Load saved metrics
        metrics = joblib.load('model_metrics.pkl')

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Test Accuracy", f"{metrics['accuracy']*100:.1f}%")
            st.metric("Precision (Fake)", f"{metrics['precision_fake']*100:.1f}%")

        with col2:
            st.metric("Precision (Real)", f"{metrics['precision_real']*100:.1f}%")
            st.metric("Recall", f"{metrics['recall']*100:.1f}%")

        with col3:
            st.metric("F1-Score", f"{metrics['f1_score']*100:.1f}%")

    except:
        st.warning("Model metrics not available. Please run training script first.")

    st.markdown("""
    **Model Details:**
    - Algorithm: Logistic Regression
    - Features: TF-IDF (5000 features, unigrams + bigrams)
    - Training Data: 36,000 articles
    - Test Data: 9,000 articles
    """)

    # Date analysis
    st.subheader("Publication Timeline")
    df['date'] = df['date'].str.strip()  # Remove trailing spaces
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month_year'] = df['date'].dt.to_period('M')

    monthly_counts = df.groupby(['month_year', 'class']).size().unstack().fillna(0)
    # Ensure both columns exist
    if 0 not in monthly_counts.columns:
        monthly_counts[0] = 0
    if 1 not in monthly_counts.columns:
        monthly_counts[1] = 0
    monthly_counts = monthly_counts[[1, 0]]  # Real first, then Fake
    monthly_counts.columns = ['Real', 'Fake']

    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_counts.plot(kind='line', marker='o', ax=ax)
    ax.set_title("Monthly Publication Trends")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Articles")
    # Legend matches the column order: Real first, then Fake
    ax.legend(['Real News', 'Fake News'])
    plt.xticks(rotation=45)
    st.pyplot(fig)



if __name__ == "__main__":
    main()
