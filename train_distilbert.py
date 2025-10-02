# Disable wandb at the earliest point
import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from torch.utils.data import Dataset
import re

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

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

def compute_metrics(eval_pred):
    """Compute accuracy and F1 for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)

    from sklearn.metrics import precision_score, recall_score, f1_score
    precision_fake = precision_score(labels, predictions, pos_label=0)
    precision_real = precision_score(labels, predictions, pos_label=1)
    f1 = f1_score(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'precision_fake': precision_fake,
        'precision_real': precision_real,
        'f1': f1
    }

class CustomEarlyStoppingCallback(TrainerCallback):
    def __init__(self, target_accuracy=0.99):
        self.target_accuracy = target_accuracy

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if metrics.get('eval_accuracy', 0) >= self.target_accuracy:
            control.should_training_stop = True

def main():
    # Enable GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv('/content/drive/MyDrive/News.csv')

    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['class'].value_counts()}")

    # Combine title and text
    df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

    # Apply preprocessing
    df['processed_text'] = df['combined_text'].apply(preprocess_text)

    # Remove empty texts
    df = df[df['processed_text'].str.len() > 0]

    # Prepare features and target
    X = df['processed_text'].tolist()
    y = df['class'].tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Load tokenizer and model
    print("Loading DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    ).to(device)

    # Create datasets
    train_dataset = NewsDataset(X_train, y_train, tokenizer)
    test_dataset = NewsDataset(X_test, y_test, tokenizer)

    # Training arguments optimized for local run
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        gradient_accumulation_steps=1,
        disable_tqdm=False,
        report_to="none"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            CustomEarlyStoppingCallback(target_accuracy=0.99)
        ]
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Evaluate on test set
    print("Evaluating on test set...")
    eval_results = trainer.evaluate()
    print("\nTest Results:")
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")

    # Get predictions for detailed report
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, preds, target_names=['Fake', 'Real']))

    # Save model and tokenizer
    print("Saving model and tokenizer...")
    model.save_pretrained('./distilbert_fake_news_model')
    tokenizer.save_pretrained('./distilbert_fake_news_tokenizer')

    print("Training completed!")
    print("Download the 'distilbert_fake_news_model' and 'distilbert_fake_news_tokenizer' directories")

    # Release GPU memory
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()