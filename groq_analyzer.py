import os
from groq import Groq
import re
api_key = os.getenv("api_key")
class GroqNewsAnalyzer:
    def __init__(self):
        # Initialize Groq client
        # Note: You'll need to set GROQ_API_KEY environment variable
        self.client = Groq(api_key=api_key)
        self.model = "gemma2-9b-it"

    def analyze_news(self, news_text, ml_prediction=None, ml_confidence=None):
        """
        Analyze news article using Groq LLM
        Returns: dict with prediction, confidence, explanation
        """

        # Create comprehensive prompt
        prompt = self._create_analysis_prompt(news_text, ml_prediction, ml_confidence)

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=1024,
                top_p=1,
                stream=False
            )

            response_text = completion.choices[0].message.content

            # Parse the response
            return self._parse_llm_response(response_text)

        except Exception as e:
            print(f"Groq API error: {e}")
            return {
                "prediction": ml_prediction if ml_prediction is not None else 1,
                "confidence": 0.5,
                "explanation": "Analysis unavailable due to API error. Using ML prediction only.",
                "error": str(e)
            }

    def _create_analysis_prompt(self, news_text, ml_prediction=None, ml_confidence=None):
        """Create detailed prompt for fake news analysis"""

        base_prompt = f"""
You are an expert fact-checker and journalist analyzing news articles for misinformation.

Analyze this news article and determine if it's likely FAKE or REAL news. Consider:
- Factual accuracy and verifiability
- Sensational language or emotional manipulation
- Source credibility indicators
- Logical consistency
- Writing style and professionalism
- Political bias or agenda
- Timeliness and context

Article to analyze:
"{news_text[:2000]}"  # Limit text length

Provide your analysis in this exact format:
PREDICTION: [FAKE or REAL]
CONFIDENCE: [0-100]% (how certain are you?)
EXPLANATION: [2-3 sentences explaining your reasoning]
KEY_INDICATORS: [list 2-3 specific reasons]
"""

        if ml_prediction is not None and ml_confidence is not None:
            ml_label = "FAKE" if ml_prediction == 0 else "REAL"
            base_prompt += f"\n\nFor reference, our ML model predicts: {ml_label} with {ml_confidence:.1f}% confidence."

        return base_prompt

    def _parse_llm_response(self, response_text):
        """Parse the LLM response into structured data"""

        try:
            # Extract prediction
            pred_match = re.search(r'PREDICTION:\s*(FAKE|REAL)', response_text, re.IGNORECASE)
            prediction = 0 if pred_match and pred_match.group(1).upper() == 'FAKE' else 1

            # Extract confidence
            conf_match = re.search(r'CONFIDENCE:\s*(\d+)%', response_text, re.IGNORECASE)
            confidence = int(conf_match.group(1)) / 100 if conf_match else 0.5

            # Extract explanation
            exp_match = re.search(r'EXPLANATION:\s*(.+?)(?=KEY_INDICATORS:|$)', response_text, re.DOTALL)
            explanation = exp_match.group(1).strip() if exp_match else "No explanation provided."

            # Extract key indicators
            ind_match = re.search(r'KEY_INDICATORS:\s*(.+)', response_text, re.DOTALL)
            key_indicators = ind_match.group(1).strip() if ind_match else ""

            return {
                "prediction": prediction,
                "confidence": confidence,
                "explanation": explanation,
                "key_indicators": key_indicators
            }

        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return {
                "prediction": 1,  # Default to real
                "confidence": 0.5,
                "explanation": "Unable to parse analysis. Please try again.",
                "key_indicators": ""
            }

def hybrid_predict(news_text, ml_model, vectorizer, groq_analyzer=None, confidence_threshold=0.7):
    """
    Hybrid prediction combining ML and LLM
    """

    # Get ML prediction first
    processed_text = preprocess_text(news_text)
    vectorized_text = vectorizer.transform([processed_text])
    ml_prediction = ml_model.predict(vectorized_text)[0]
    ml_probabilities = ml_model.predict_proba(vectorized_text)[0]
    ml_confidence = max(ml_probabilities)

    result = {
        "ml_prediction": ml_prediction,
        "ml_confidence": ml_confidence,
        "final_prediction": ml_prediction,
        "final_confidence": ml_confidence,
        "used_llm": False,
        "llm_analysis": None
    }

    # Use LLM if ML confidence is low or if Groq analyzer is available
    if groq_analyzer and (ml_confidence < confidence_threshold):
        print(f"ML confidence {ml_confidence:.2f} < {confidence_threshold}, using LLM analysis...")

        llm_result = groq_analyzer.analyze_news(
            news_text,
            ml_prediction=ml_prediction,
            ml_confidence=ml_confidence
        )

        result["llm_analysis"] = llm_result
        result["used_llm"] = True

        # Combine predictions - give more weight to LLM when used
        # Since ML model seems biased towards fake news
        ml_weight = ml_confidence * 0.6  # Reduce ML weight
        llm_weight = llm_result["confidence"] * 0.9  # Increase LLM weight

        if llm_weight > ml_weight:
            result["final_prediction"] = llm_result["prediction"]
            result["final_confidence"] = llm_result["confidence"]
        else:
            result["final_prediction"] = ml_prediction
            result["final_confidence"] = ml_confidence

    return result

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

# Import pandas for preprocess_text (avoiding circular import)
import pandas as pd
