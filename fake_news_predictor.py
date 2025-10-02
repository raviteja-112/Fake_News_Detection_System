import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re
import pandas as pd
import os

class FakeNewsPredictor:
    def __init__(self, model_path='./distilbert_fake_news_model', tokenizer_path='./distilbert_fake_news_tokenizer'):
        """
        Initialize the DistilBERT-based fake news predictor
        
        Args:
            model_path (str): Path to the trained model directory
            tokenizer_path (str): Path to the tokenizer directory
            
        Raises:
            FileNotFoundError: If model or tokenizer files are not found
            RuntimeError: If model loading fails
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on device: {self.device}")

        try:
            # Validate paths exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model directory not found: {model_path}")
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_path}")

            # Load tokenizer and model
            self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()

            print("Model loaded successfully!")
            
        except FileNotFoundError as e:
            error_msg = f"Model files not found: {str(e)}\n\nPlease ensure:\n" \
                       f"1. Model directory exists at: {model_path}\n" \
                       f"2. Tokenizer directory exists at: {tokenizer_path}\n" \
                       f"3. Files are downloaded from training or available locally"
            raise FileNotFoundError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}\n\nThis could be due to:\n" \
                       f"- Corrupted model files\n" \
                       f"- Incompatible PyTorch/transformers versions\n" \
                       f"- Missing dependencies"
            raise RuntimeError(error_msg) from e

    def preprocess_text(self, text):
        """
        Preprocess text - same cleaning as training
        """
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

    def predict(self, news_text, max_length=512):
        """
        Predict if news is fake or real

        Args:
            news_text (str): The news article text to analyze
            max_length (int): Maximum sequence length for tokenizer (default: 512)

        Returns:
            dict: Prediction results with confidence and probabilities

        Raises:
            ValueError: If input validation fails
        """
        # Input validation
        if not isinstance(news_text, str):
            raise ValueError(f"news_text must be a string, got {type(news_text)}")
        
        if not news_text.strip():
            raise ValueError("news_text cannot be empty or whitespace-only")
            
        if len(news_text.strip()) < 10:
            raise ValueError("news_text must be at least 10 characters long for meaningful analysis")
            
        if not isinstance(max_length, int) or max_length < 32 or max_length > 512:
            raise ValueError("max_length must be an integer between 32 and 512")

        # Preprocess the text
        processed_text = self.preprocess_text(news_text)
        
        if len(processed_text) == 0:
            raise ValueError("Text preprocessing resulted in empty content - please provide meaningful text")

        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        # Extract probabilities
        fake_prob = probabilities[0][0].item()
        real_prob = probabilities[0][1].item()

        # Confidence is the higher probability
        confidence = max(fake_prob, real_prob)

        return {
            'prediction': predicted_class,  # 0 = Fake, 1 = Real
            'prediction_label': 'Fake' if predicted_class == 0 else 'Real',
            'confidence': confidence,
            'fake_probability': fake_prob,
            'real_probability': real_prob,
            'processed_text': processed_text
        }

    def predict_batch(self, news_texts, max_length=512):
        """
        Predict multiple news articles in batch for better performance

        Args:
            news_texts (list): List of news article texts to analyze
            max_length (int): Maximum sequence length for tokenizer (default: 512)

        Returns:
            list: List of prediction results dictionaries

        Raises:
            ValueError: If input validation fails
        """
        # Input validation
        if not isinstance(news_texts, list):
            raise ValueError(f"news_texts must be a list, got {type(news_texts)}")
            
        if not news_texts:
            return []
            
        if len(news_texts) > 100:
            raise ValueError("Batch size cannot exceed 100 articles for performance reasons")
            
        if not isinstance(max_length, int) or max_length < 32 or max_length > 512:
            raise ValueError("max_length must be an integer between 32 and 512")

        # Validate each text in the batch
        valid_texts = []
        invalid_indices = []
        
        for i, text in enumerate(news_texts):
            if not isinstance(text, str):
                invalid_indices.append((i, f"must be a string, got {type(text)}"))
                continue
                
            if not text.strip():
                invalid_indices.append((i, "cannot be empty or whitespace-only"))
                continue
                
            if len(text.strip()) < 10:
                invalid_indices.append((i, "must be at least 10 characters long"))
                continue
                
            valid_texts.append(text)

        if invalid_indices:
            error_details = "\n".join([f"  Index {idx}: {reason}" for idx, reason in invalid_indices])
            raise ValueError(f"Invalid texts in batch:\n{error_details}")

        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in valid_texts]
        
        # Check for empty processed texts
        empty_indices = [i for i, text in enumerate(processed_texts) if len(text) == 0]
        if empty_indices:
            raise ValueError(f"Text preprocessing resulted in empty content for indices: {empty_indices}")

        # Tokenize in batch
        inputs = self.tokenizer(
            processed_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions in batch
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1).tolist()

        results = []
        for i, (predicted_class, prob) in enumerate(zip(predicted_classes, probabilities)):
            fake_prob = prob[0].item()
            real_prob = prob[1].item()
            confidence = max(fake_prob, real_prob)

            results.append({
                'prediction': predicted_class,
                'prediction_label': 'Fake' if predicted_class == 0 else 'Real',
                'confidence': confidence,
                'fake_probability': fake_prob,
                'real_probability': real_prob,
                'processed_text': processed_texts[i],
                'original_text': valid_texts[i]
            })

        return results

# Global predictor instance
_predictor = None

def get_predictor():
    """Get or create the global predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = FakeNewsPredictor()
    return _predictor

def predict_news(news_text):
    """
    Convenience function for predicting news authenticity

    Args:
        news_text (str): News article text

    Returns:
        dict: Prediction results
    """
    predictor = get_predictor()
    return predictor.predict(news_text)

def predict_news_batch(news_texts):
    """
    Convenience function for batch prediction of multiple news articles

    Args:
        news_texts (list): List of news article texts

    Returns:
        list: List of prediction results dictionaries
    """
    predictor = get_predictor()
    return predictor.predict_batch(news_texts)

# Model warm-up on import to avoid first-call latency
try:
    _warmup_predictor = get_predictor()
    # Perform a warm-up prediction to initialize all components
    _warmup_result = _warmup_predictor.predict("Model warm-up text for initialization")
    print("Model warm-up completed successfully")
except Exception as e:
    print(f"Model warm-up failed: {str(e)}")
