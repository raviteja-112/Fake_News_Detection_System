# DistilBERT Fake News Detection System

A high-performance, transformer-based fake news detection system built with DistilBERT, providing accurate and explainable news authenticity analysis through local processing without external API dependencies.

## Features

- **Real-time Analysis**: Instant classification of news articles
- **High Accuracy**: Trained on diverse datasets for robust performance  
- **Easy Integration**: Simple API for integration with other applications
- **Batch Processing**: Process multiple articles at once for efficiency
- **Local Processing**: No external API calls required
- **Explainable Results**: Detailed probability breakdowns and confidence scores

## Quick Start

```python
from fake_news_predictor import predict_news

# Analyze single news article
result = predict_news("Your news article text here")
print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence']:.1%}")

# Analyze multiple articles
batch_results = predict_news_batch(["Article 1", "Article 2", "Article 3"])
```

## Installation

```bash
pip install -r requirements.txt
```

## Performance

- **Single Article**: ~800ms on CPU, ~150ms on GPU
- **Batch Processing**: ~50ms per article (batch_size=8)
- **Accuracy**: 99.23% on test set
- **Precision**: 99.98% for fake news detection

## API Reference

### `predict_news(text: str) -> dict`
Analyze a single news article. Returns a dictionary with prediction results including:
- `prediction`: 0 for fake, 1 for real
- `prediction_label`: Human-readable label
- `confidence`: Confidence score (0-1)
- `probabilities`: Individual class probabilities

### `predict_news_batch(texts: list) -> list`
Analyze multiple news articles in batch. Returns a list of result dictionaries.

## Model Details

The system uses a fine-tuned DistilBERT model with the following specifications:
- **Model**: `distilbert-base-uncased`
- **Parameters**: 66 million
- **Max Sequence Length**: 512 tokens
- **Training Data**: 45,000+ labeled news articles
- **Accuracy**: 99.23% on test set

## TODO List

Track progress on the development tasks:

- [x] Update model performance metrics in app.py
- [x] Add proper error handling in fake_news_predictor.py
- [x] Add batch prediction support
- [x] Implement model warm-up
- [x] Add input validation
- [x] Update README.md if needed

**Current Progress: 6/6 items completed (100%)**

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
