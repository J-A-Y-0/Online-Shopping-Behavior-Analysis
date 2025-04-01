import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set plotting style
sns.set_theme(style='whitegrid', palette='husl')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14

def analyze_page_behavior(df):
    """Detailed analysis of page behavior metrics"""
    print("\n1. Page Behavior Analysis")
    print("-" * 50)
    
    # Calculate average metrics by revenue
    behavior_metrics = df.groupby('Revenue').agg({
        'BounceRates': 'mean',
        'ExitRates': 'mean',
        'PageValues': 'mean',
        'ProductRelated': 'mean',
        'ProductRelated_Duration': 'mean'
    }).round(3)
    
    print("\nAverage Metrics by Purchase Decision:")
    print(behavior_metrics)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Bounce Rates vs Page Values
    sns.scatterplot(data=df, x='BounceRates', y='PageValues', 
                   hue='Revenue', ax=axes[0,0])
    axes[0,0].set_title('Bounce Rates vs Page Values')
    
    # Product Related Duration vs Page Values
    sns.scatterplot(data=df, x='ProductRelated_Duration', y='PageValues',
                   hue='Revenue', ax=axes[0,1])
    axes[0,1].set_title('Product Duration vs Page Values')
    
    # Exit Rates Distribution
    sns.boxplot(data=df, x='Revenue', y='ExitRates', ax=axes[1,0])
    axes[1,0].set_title('Exit Rates by Purchase Decision')
    
    # Product Related Pages Distribution
    sns.boxplot(data=df, x='Revenue', y='ProductRelated', ax=axes[1,1])
    axes[1,1].set_title('Product Pages by Purchase Decision')
    
    plt.tight_layout()
    plt.savefig('plots/page_behavior_analysis.png')
    plt.close()

def analyze_traffic_sources(df):
    """Analysis of traffic sources and their impact"""
    print("\n2. Traffic Source Analysis")
    print("-" * 50)
    
    # Calculate conversion rates by traffic type
    traffic_conv = df.groupby('TrafficType').agg({
        'Revenue': ['count', 'mean'],
        'PageValues': 'mean',
        'BounceRates': 'mean'
    })
    traffic_conv.columns = ['Count', 'ConversionRate', 'AvgPageValue', 'AvgBounceRate']
    traffic_conv['ConversionRate'] = traffic_conv['ConversionRate'] * 100
    
    print("\nTraffic Source Performance:")
    print(traffic_conv.sort_values('ConversionRate', ascending=False))
    
    # Visualize traffic source performance
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    # Conversion rates by traffic type
    traffic_conv['ConversionRate'].plot(kind='bar', ax=axes[0])
    axes[0].set_title('Conversion Rates by Traffic Type')
    axes[0].set_ylabel('Conversion Rate (%)')
    
    # Page values by traffic type
    traffic_conv['AvgPageValue'].plot(kind='bar', ax=axes[1])
    axes[1].set_title('Average Page Values by Traffic Type')
    axes[1].set_ylabel('Average Page Value')
    
    plt.tight_layout()
    plt.savefig('plots/traffic_source_analysis.png')
    plt.close()

def analyze_special_days(df):
    """Analysis of special day impact"""
    print("\n3. Special Day Analysis")
    print("-" * 50)
    
    # Create special day categories
    df['SpecialDayCategory'] = pd.cut(df['SpecialDay'], 
                                     bins=[-np.inf, 0, 0.2, 0.4, 0.6, 0.8, np.inf],
                                     labels=['No Special Day', 'Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Calculate metrics by special day category
    special_day_metrics = df.groupby('SpecialDayCategory').agg({
        'Revenue': ['count', 'mean'],
        'PageValues': 'mean',
        'BounceRates': 'mean'
    })
    special_day_metrics.columns = ['Count', 'ConversionRate', 'AvgPageValue', 'AvgBounceRate']
    special_day_metrics['ConversionRate'] = special_day_metrics['ConversionRate'] * 100
    
    print("\nMetrics by Special Day Category:")
    print(special_day_metrics)
    
    # Visualize special day impact
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    special_day_metrics['ConversionRate'].plot(kind='bar', ax=axes[0])
    axes[0].set_title('Conversion Rate by Special Day Category')
    axes[0].set_ylabel('Conversion Rate (%)')
    
    special_day_metrics['AvgPageValue'].plot(kind='bar', ax=axes[1])
    axes[1].set_title('Average Page Value by Special Day Category')
    axes[1].set_ylabel('Average Page Value')
    
    plt.tight_layout()
    plt.savefig('plots/special_day_analysis.png')
    plt.close()

def analyze_technical_factors(df):
    """Analysis of operating systems and browsers"""
    print("\n4. Technical Factors Analysis")
    print("-" * 50)
    
    # Calculate conversion rates by OS and Browser
    tech_metrics = df.groupby(['OperatingSystems', 'Browser'])['Revenue'].agg(['count', 'mean'])
    tech_metrics['mean'] = tech_metrics['mean'] * 100
    
    print("\nConversion Rates by OS and Browser:")
    print(tech_metrics.sort_values('mean', ascending=False))
    
    # Visualize technical factors
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # OS conversion rates
    os_conv = df.groupby('OperatingSystems')['Revenue'].mean() * 100
    os_conv.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Conversion Rate by Operating System')
    axes[0].set_ylabel('Conversion Rate (%)')
    
    # Browser conversion rates
    browser_conv = df.groupby('Browser')['Revenue'].mean() * 100
    browser_conv.plot(kind='bar', ax=axes[1])
    axes[1].set_title('Conversion Rate by Browser')
    axes[1].set_ylabel('Conversion Rate (%)')
    
    plt.tight_layout()
    plt.savefig('plots/technical_factors_analysis.png')
    plt.close()

def perform_clustering(df):
    """Perform customer segmentation using K-means clustering"""
    print("\n5. Customer Segmentation Analysis")
    print("-" * 50)
    
    # Select features for clustering
    cluster_features = ['ProductRelated', 'ProductRelated_Duration', 'BounceRates', 
                       'ExitRates', 'PageValues', 'SpecialDay']
    
    # Prepare data for clustering
    X = df[cluster_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal number of clusters
    silhouette_scores = []
    K = range(2, 8)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
    
    optimal_k = K[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters: {optimal_k}")
    
    # Perform clustering with optimal K
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze clusters
    cluster_stats = df.groupby('Cluster').agg({
        'Revenue': ['count', 'mean'],
        'PageValues': 'mean',
        'ProductRelated': 'mean',
        'BounceRates': 'mean'
    })
    cluster_stats.columns = ['Size', 'ConversionRate', 'AvgPageValue', 'AvgProductPages', 'AvgBounceRate']
    cluster_stats['ConversionRate'] = cluster_stats['ConversionRate'] * 100
    
    print("\nCluster Characteristics:")
    print(cluster_stats)
    
    # Visualize clusters
    fig = plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                         c=df['Cluster'], cmap='viridis')
    plt.xlabel('Standardized Product Related Pages')
    plt.ylabel('Standardized Product Related Duration')
    plt.title('Customer Segments Visualization')
    plt.colorbar(scatter)
    plt.savefig('plots/customer_segments.png')
    plt.close()
    
    return df['Cluster']

def main():
    """Main analysis function"""
    # Load data
    df = pd.read_csv('data/online_shoppers_intention.csv')
    
    # Perform analyses
    analyze_page_behavior(df)
    analyze_traffic_sources(df)
    analyze_special_days(df)
    analyze_technical_factors(df)
    clusters = perform_clustering(df)
    
    print("\nAdvanced analysis complete! Check the 'plots' directory for visualizations.")

if __name__ == "__main__":
    main() 