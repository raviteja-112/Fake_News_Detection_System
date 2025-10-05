import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from fake_news_predictor import predict_news

# Set page configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        border: 3px solid;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .fake-news {
        background: linear-gradient(135deg, #ffe6e6 0%, #ffcccc 100%);
        border-color: #ff4444;
        color: #cc0000;
    }
    .fake-news h3 {
        color: #cc0000 !important;
    }
    .real-news {
        background: linear-gradient(135deg, #e6ffe6 0%, #ccffcc 100%);
        border-color: #44aa44;
        color: #006600;
    }
    .real-news h3 {
        color: #006600 !important;
    }
    .confidence-meter {
        width: 100%;
        height: 24px;
        border-radius: 12px;
        background-color: #f0f0f0;
        margin: 0.5rem 0;
        border: 1px solid #ddd;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 12px;
        transition: width 0.5s ease;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar navigation
    st.sidebar.title("📰 Fake News Detection")
    page = st.sidebar.radio("Navigate", ["News Prediction", "Dataset Visualization"])

    if page == "News Prediction":
        show_prediction_page()
    else:
        show_visualization_page()

def show_prediction_page():
    st.markdown('<h1 class="main-header">📰 Fake News Detection System</h1>', unsafe_allow_html=True)
    st.markdown("### Analyze news articles for authenticity using advanced AI")

    # News input
    st.subheader("📝 Enter News Article")
    news_text = st.text_area(
        "Paste the news article text here:",
        height=200,
        placeholder="Enter the full news article text to analyze..."
    )

    # Analysis button
    if st.button("🔍 Analyze News", type="primary", use_container_width=True):
        if not news_text.strip():
            st.error("Please enter some news text to analyze.")
            return

        with st.spinner("Analyzing news article..."):
            # Add a small delay for better UX
            time.sleep(0.5)

            try:
                # Get prediction
                result = predict_news(news_text)

                # Display results
                display_prediction_results(result, news_text)

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("Make sure your model files are in the correct location.")

def display_prediction_results(result, news_text):
    """Display the prediction results in a nice format"""

    prediction_class = result['prediction_label']
    confidence = result['confidence']
    fake_prob = result['fake_probability']
    real_prob = result['real_probability']

    # Main prediction box
    box_class = "fake-news" if prediction_class == "Fake" else "real-news"
    st.markdown(f"""
    <div class="prediction-box {box_class}">
        <h3 style="margin-top: 0;">📰 Prediction: {prediction_class} News</h3>
        <p><strong>Confidence:</strong> {confidence:.1%}</p>
    </div>
    """, unsafe_allow_html=True)

    # Detailed breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Probability Breakdown")
        st.metric("Fake News Probability", f"{fake_prob:.1%}")
        st.metric("Real News Probability", f"{real_prob:.1%}")

        # Progress bars
        st.progress(fake_prob, text="Fake Probability")
        st.progress(real_prob, text="Real Probability")

    with col2:
        st.subheader("📈 Confidence Analysis")
        if confidence > 0.8:
            st.success("High confidence prediction")
        elif confidence > 0.6:
            st.warning("Moderate confidence - review carefully")
        else:
            st.error("Low confidence - additional verification recommended")

        # Confidence meter
        st.markdown(f"""
        <div class="confidence-meter">
            <div class="confidence-fill" style="width: {confidence*100}%; background-color: {'#ff6b6b' if prediction_class == 'Fake' else '#4ecdc4'};"></div>
        </div>
        <p style="text-align: center; margin-top: 0.5rem;">Overall Confidence: {confidence:.1%}</p>
        """, unsafe_allow_html=True)

    # Technical details (expandable)
    with st.expander("🔧 Technical Details"):
        st.write("**Model:** DistilBERT (Fine-tuned)")
        st.write("**Architecture:** Transformer-based Classification")
        st.write("**Input Length:** 512 tokens (truncated if longer)")
        st.write("**Device:** CPU/GPU (automatic detection)")

        if len(result['processed_text']) < len(news_text):
            st.info("Note: Text was truncated for analysis (512 token limit)")

def show_visualization_page():
    st.markdown('<h1 class="main-header">📊 Dataset & Model Analytics</h1>', unsafe_allow_html=True)

    try:
        # Load dataset for visualization
        df = pd.read_csv('News.csv')

        # Basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Articles", len(df))
        with col2:
            fake_count = (df['class'] == 0).sum()
            st.metric("Fake News", fake_count)
        with col3:
            real_count = (df['class'] == 1).sum()
            st.metric("Real News", real_count)

        # Class distribution pie chart
        st.subheader("📈 Class Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        class_counts = df['class'].value_counts()
        labels = ['Fake News' if idx == 0 else 'Real News' for idx in class_counts.index]
        colors = ['#ff6b6b', '#4ecdc4']

        ax.pie(class_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        # Model performance metrics (DistilBERT fine-tuned)
        st.subheader("🎯 Model Performance (DistilBERT Fine-tuned)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test Accuracy", "99.23%")
            st.metric("Precision (Fake)", "99.98%")
        with col2:
            st.metric("F1-Score", "99.23%")
            st.metric("Precision (Real)", "98.44%")
        with col3:
            st.metric("Recall (Fake)", "99.00%")
            st.metric("Recall (Real)", "99.47%")

        st.success("Model trained on 44,919 articles with stratified 80/20 train/test split")

    except FileNotFoundError:
        st.error("News.csv not found. Please ensure the dataset file is in the project directory.")
    except Exception as e:
        st.error(f"Error loading visualizations: {str(e)}")

if __name__ == "__main__":
    main()
