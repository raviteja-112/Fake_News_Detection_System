#!/usr/bin/env python3
"""
Test script for the hybrid fake news detection system
"""

import joblib
from groq_analyzer import GroqNewsAnalyzer, hybrid_predict

def test_hybrid_system():
    """Test the hybrid prediction system"""

    print("Loading models...")
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Test news articles
    test_articles = [
        # Fake news example
        "BREAKING: Scientists Discover That Eating Chocolate Cures All Diseases! A new study from Harvard University reveals that consuming just one chocolate bar per day can eliminate cancer, heart disease, and even COVID-19. The research, which was conducted on a sample of 5 people over 2 weeks, shows miraculous results that will change medicine forever.",

        # Real news example (simulated)
        "The Federal Reserve announced today that it will maintain current interest rates following its latest policy meeting. Chairman Jerome Powell stated that inflation remains a concern but economic growth is stabilizing. The decision affects mortgage rates and borrowing costs across the United States.",

        # Ambiguous/unclear example
        "Local community center hosts annual food drive. Volunteers collected over 500 pounds of non-perishable items for families in need. The event was organized by the neighborhood association and supported by local businesses."
    ]

    print("\n" + "="*80)
    print("TESTING HYBRID FAKE NEWS DETECTION SYSTEM")
    print("="*80)

    for i, article in enumerate(test_articles, 1):
        print(f"\n📰 TEST ARTICLE {i}")
        print("-" * 50)
        print(article[:200] + "..." if len(article) > 200 else article)
        print("-" * 50)

        # Test with ML only
        print("\n⚡ ML-ONLY ANALYSIS:")
        result_ml = hybrid_predict(article, model, vectorizer, groq_analyzer=None)
        ml_pred = "FAKE" if result_ml["ml_prediction"] == 0 else "REAL"
        print(f"Prediction: {ml_pred}")
        print(".2f")

        # Test with hybrid (ML + LLM)
        print("\n🤖 HYBRID ANALYSIS (ML + LLM):")
        try:
            groq_analyzer = GroqNewsAnalyzer()
            result_hybrid = hybrid_predict(article, model, vectorizer, groq_analyzer, confidence_threshold=0.8)

            hybrid_pred = "FAKE" if result_hybrid["final_prediction"] == 0 else "REAL"
            print(f"Final Prediction: {hybrid_pred}")
            print(".2f")
            print(f"Used LLM: {result_hybrid['used_llm']}")

            if result_hybrid["used_llm"] and result_hybrid["llm_analysis"]:
                llm = result_hybrid["llm_analysis"]
                print(f"LLM Prediction: {'FAKE' if llm['prediction'] == 0 else 'REAL'}")
                print(".2f")
                print(f"Explanation: {llm['explanation'][:150]}...")

        except Exception as e:
            print(f"Hybrid analysis failed: {e}")
            print("Falling back to ML-only result.")

        print("\n" + "="*80)

if __name__ == "__main__":
    test_hybrid_system()
