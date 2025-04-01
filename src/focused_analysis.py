import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest, RandomForestClassifier

def run_focused_analysis(data_path='data/online_shoppers_intention.csv'):
    """Run the four selected advanced analyses"""
    # Load data
    df = pd.read_csv(data_path)
    
    # 1. User Journey Analysis
    journey_patterns = analyze_user_journeys(df)
    
    # 2. Advanced Customer Segmentation
    segments = perform_customer_segmentation(df)
    
    # 3. Anomaly Detection
    anomalies = detect_anomalies(df)
    
    # 4. Feature Importance Analysis
    importance = analyze_feature_importance(df)
    
    return journey_patterns, segments, anomalies, importance

def analyze_user_journeys(df):
    """Analyze user journey patterns"""
    # Create journey sequences
    df['PageSequence'] = df.apply(lambda x: [
        'Admin' if x['Administrative'] > 0 else '',
        'Info' if x['Informational'] > 0 else '',
        'Product' if x['ProductRelated'] > 0 else ''
    ], axis=1)
    
    df['PageSequence'] = df['PageSequence'].apply(lambda x: '->'.join([i for i in x if i]))
    
    # Analyze sequences
    journey_stats = df.groupby('PageSequence').agg({
        'Revenue': ['count', 'mean'],
        'PageValues': 'mean'
    }).round(3)
    journey_stats.columns = ['Count', 'ConversionRate', 'AvgPageValue']
    
    # Visualize top journey patterns
    plt.figure(figsize=(12, 6))
    top_journeys = journey_stats.sort_values('Count', ascending=True).tail(10)
    sns.barplot(data=top_journeys.reset_index(), x='Count', y='PageSequence')
    plt.title('Top 10 User Journey Patterns')
    plt.tight_layout()
    plt.savefig('plots/top_journeys.png')
    plt.close()
    
    return journey_stats

def perform_customer_segmentation(df, n_components=5):
    """Perform advanced customer segmentation"""
    # Select features for clustering
    features = ['ProductRelated', 'ProductRelated_Duration', 'BounceRates', 
               'ExitRates', 'PageValues', 'SpecialDay']
    X = df[features].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    df['Segment'] = gmm.fit_predict(X_scaled)
    
    # Analyze segments
    segment_stats = df.groupby('Segment').agg({
        'Revenue': ['count', 'mean'],
        'PageValues': 'mean',
        'BounceRates': 'mean'
    })
    
    # Visualize segments
    plt.figure(figsize=(12, 6))
    for i in range(n_components):
        plt.scatter(X_scaled[df['Segment'] == i, 0], 
                   X_scaled[df['Segment'] == i, 1], 
                   label=f'Segment {i}')
    plt.title('Customer Segments')
    plt.xlabel('Standardized Product Related Pages')
    plt.ylabel('Standardized Product Related Duration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/customer_segments.png')
    plt.close()
    
    return segment_stats

def detect_anomalies(df):
    """Detect anomalous behavior patterns"""
    # Select features for anomaly detection
    features = ['ProductRelated', 'ProductRelated_Duration', 'BounceRates', 
               'ExitRates', 'PageValues']
    X = df[features].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df['Is_Anomaly'] = iso_forest.fit_predict(X_scaled)
    
    # Analyze anomalies
    anomaly_stats = df[df['Is_Anomaly'] == -1].describe()
    
    # Visualize anomalies
    plt.figure(figsize=(12, 6))
    plt.scatter(X_scaled[df['Is_Anomaly'] == 1, 0], 
               X_scaled[df['Is_Anomaly'] == 1, 1], 
               c='blue', label='Normal')
    plt.scatter(X_scaled[df['Is_Anomaly'] == -1, 0], 
               X_scaled[df['Is_Anomaly'] == -1, 1], 
               c='red', label='Anomaly')
    plt.title('Anomaly Detection Results')
    plt.xlabel('Standardized Product Related Pages')
    plt.ylabel('Standardized Product Related Duration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/anomalies.png')
    plt.close()
    
    return anomaly_stats

def analyze_feature_importance(df):
    """Analyze feature importance using Random Forest"""
    # Prepare features
    X = df.select_dtypes(include=['float64', 'int64']).copy()
    y = df['Revenue']
    
    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()
    
    return feature_importance

if __name__ == "__main__":
    # Run all analyses
    journey_patterns, segments, anomalies, importance = run_focused_analysis()
    
    # Save results to CSV files
    journey_patterns.to_csv('reports/journey_patterns.csv')
    segments.to_csv('reports/customer_segments.csv')
    anomalies.to_csv('reports/anomaly_stats.csv')
    importance.to_csv('reports/feature_importance.csv') 