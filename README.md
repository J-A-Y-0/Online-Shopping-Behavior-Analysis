# Online Shopping Behavior Analysis

This project analyzes online shopping behavior patterns using machine learning and data visualization techniques. It focuses on understanding customer purchase intentions and identifying key factors that influence online shopping decisions.

## Dataset
The dataset used in this analysis is the "Online Shoppers Purchasing Intention" dataset from the UCI Machine Learning Repository. It contains 12,330 user sessions with the following features:

- **Behavioral Metrics**: PageValues, BounceRates, ExitRates
- **Technical Information**: OperatingSystems, Browser, TrafficType
- **User Characteristics**: VisitorType, Region, Weekend
- **Temporal Features**: Month, SpecialDay
- **Target Variable**: Revenue (Boolean indicating purchase completion)

Dataset Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)

## Author
- **Jay (Weike) Yu** ([@J-A-Y-0](https://github.com/J-A-Y-0))

## Project Structure

```
online_shopping_analysis/
├── data/
│   └── online_shoppers_intention.csv    # Dataset
├── src/
│   ├── data_utils.py                    # Data loading and preprocessing
│   ├── model_evaluation.py              # Model training and evaluation
│   ├── viz_utils.py                     # Visualization functions
│   ├── generate_plots.py                # Script to generate all visualizations
│   └── sentiment_analysis.py            # Sentiment analysis functionality
├── plots/                               # Generated visualizations
│   ├── *.png                           # Static visualizations
│   └── *.html                          # Interactive Plotly visualizations
└── reports/                            # Analysis reports and results
    ├── analysis_report.html            # Main analysis report
    ├── feature_importance.csv          # Feature importance scores
    ├── anomaly_stats.csv               # Anomaly detection results
    └── customer_segments.csv           # Customer segmentation data
```

## Features

1. **Data Analysis**
   - Customer behavior analysis
   - Traffic source analysis
   - Page behavior analysis
   - User journey analysis

2. **Visualizations**
   - Interactive Plotly visualizations
   - Static matplotlib/seaborn plots
   - Feature importance analysis
   - Anomaly detection visualizations

3. **Model Evaluation**
   - Purchase prediction
   - Customer segmentation
   - Feature importance analysis
   - Anomaly detection

4. **AI Features**
   - Sentiment analysis of customer reviews using DistilBERT
   - Batch processing of multiple reviews
   - Confidence scoring for sentiment predictions
   - Integration with existing data analysis pipeline

## Key Findings

1. **Customer Segments**
   - Identified 5 distinct customer segments
   - High-value segment (15.3%) shows 61.6% conversion rate
   - Medium-value segment (8.7%) achieves 32.8% conversion rate
   - Low-value segments (75.9%) have conversion rates below 5%

2. **User Journeys**
   - Most common path: Direct product page visits (5,299 sessions)
   - Best performing path: Admin->Info->Product (24.3% conversion rate)
   - Multi-page journeys show 3x higher conversion rates than direct visits

3. **Feature Impact**
   - PageValues is the strongest predictor (27.1% importance)
   - Product-related duration and pages are key indicators
   - Technical factors (OS, browser) have moderate impact

4. **Anomaly Detection**
   - 10% of sessions identified as anomalous
   - Extreme values in product-related metrics
   - Both very short and very long sessions flagged

5. **Strategic Insights**
   - Focus on high-value customer retention
   - Optimize multi-page user journeys
   - Enhance product page engagement
   - Monitor and investigate anomalous patterns

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Generate all visualizations:
```bash
python src/generate_plots.py
```

2. Use sentiment analysis:
```python
from src.sentiment_analysis import SentimentAnalyzer

# Initialize the analyzer
analyzer = SentimentAnalyzer()

# Analyze a single review
result = analyzer.analyze_text("Great product, exactly what I was looking for!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2f}")

# Analyze multiple reviews from a DataFrame
import pandas as pd
reviews_df = pd.DataFrame({
    'review_text': [
        "Great product, exactly what I was looking for!",
        "Disappointed with the quality, would not recommend.",
        "Fast shipping and good customer service."
    ]
})
analyzed_df = analyzer.analyze_reviews(reviews_df, 'review_text')
```

3. View the generated visualizations:
   - Static plots: Check the `plots/` directory for PNG files
   - Interactive plots: Open HTML files in `plots/` directory in a web browser

4. View the analysis report:
   - Open `reports/analysis_report.html` in a web browser
   - The report includes:
     - Key findings and insights
     - Detailed analysis of customer segments
     - User journey patterns
     - Feature importance analysis
     - Anomaly detection results
     - Strategic recommendations

## Analysis Report Contents

1. **User Journey Analysis**
   - Most common paths through the website
   - Conversion rates by journey type
   - Optimization recommendations

2. **Customer Segmentation**
   - High-value customers (61.6% conversion rate)
   - Medium-value customers (32.8% conversion rate)
   - Low-value/browsing customers (<5% conversion rate)

3. **Feature Importance**
   - PageValues (27.1% importance)
   - Segment classification
   - Product-related metrics

4. **Anomaly Detection**
   - Unusual shopping patterns
   - Technical anomalies
   - Risk management insights

## Visualization Types

1. **Conversion Analysis**
   - Conversion rates by visitor type
   - Conversion rates by traffic source
   - Conversion rates by operating system
   - Conversion rates by browser

2. **Behavior Analysis**
   - Page behavior patterns
   - User behavior patterns
   - Traffic analysis
   - Temporal patterns

3. **Technical Analysis**
   - Feature importance
   - Correlation matrix
   - Anomaly detection

## Dependencies

Core packages:
- pandas>=1.3.0
- numpy>=1.21.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- scikit-learn>=0.24.0
- plotly>=5.1.0

AI/ML packages:
- transformers>=4.30.0
- torch>=2.0.0
- xgboost>=1.4.0
- lightgbm>=3.2.0
- imbalanced-learn>=0.8.0

## License

MIT License