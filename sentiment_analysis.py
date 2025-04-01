from transformers import pipeline
import pandas as pd
import numpy as np
from typing import List, Dict, Union

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analysis pipeline."""
        self.analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict containing sentiment label and confidence score
        """
        result = self.analyzer(text)[0]
        return {
            'sentiment': result['label'],
            'confidence': result['score']
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Analyze sentiment of multiple texts.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List of dictionaries containing sentiment analysis results
        """
        results = self.analyzer(texts)
        return [
            {
                'sentiment': result['label'],
                'confidence': result['score']
            }
            for result in results
        ]
    
    def analyze_reviews(self, reviews_df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Analyze sentiment of reviews in a DataFrame.
        
        Args:
            reviews_df (pd.DataFrame): DataFrame containing reviews
            text_column (str): Name of the column containing review text
            
        Returns:
            DataFrame with added sentiment analysis columns
        """
        results = self.analyze_batch(reviews_df[text_column].tolist())
        
        # Add sentiment analysis results to DataFrame
        reviews_df['sentiment'] = [r['sentiment'] for r in results]
        reviews_df['sentiment_confidence'] = [r['confidence'] for r in results]
        
        return reviews_df

def main():
    """Example usage of the SentimentAnalyzer class."""
    # Example data
    sample_reviews = [
        "Great product, exactly what I was looking for!",
        "Disappointed with the quality, would not recommend.",
        "Fast shipping and good customer service.",
        "Product arrived damaged, very unhappy."
    ]
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Analyze single review
    print("Single review analysis:")
    result = analyzer.analyze_text(sample_reviews[0])
    print(f"Review: {sample_reviews[0]}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    # Analyze batch of reviews
    print("\nBatch analysis:")
    results = analyzer.analyze_batch(sample_reviews)
    for review, result in zip(sample_reviews, results):
        print(f"\nReview: {review}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2f}")

if __name__ == "__main__":
    main() 