import pandas as pd
from viz_utils import (
    plot_conversion_by_category,
    plot_temporal_patterns,
    plot_correlation_matrix,
    plot_page_behavior_analysis,
    plot_user_behavior_patterns,
    plot_traffic_analysis
)

def main():
    """Generate all visualizations using viz_utils functions"""
    print("Loading data...")
    df = pd.read_csv('data/online_shoppers_intention.csv')
    
    print("Generating visualizations...")
    
    # Generate all plots
    categories = ['VisitorType', 'TrafficType', 'OperatingSystems', 'Browser']
    for category in categories:
        print(f"Creating conversion plot for {category}...")
        fig = plot_conversion_by_category(df, category)
        fig.savefig(f'plots/conversion_by_{category.lower()}.png')
    
    print("Creating temporal patterns plot...")
    fig = plot_temporal_patterns(df)
    fig.write_html('plots/temporal_patterns.html')
    
    print("Creating correlation matrix...")
    fig = plot_correlation_matrix(df)
    fig.savefig('plots/correlation_matrix.png')
    
    print("Creating page behavior analysis...")
    fig = plot_page_behavior_analysis(df)
    fig.write_html('plots/page_behavior_analysis.html')
    
    print("Creating user behavior patterns...")
    fig = plot_user_behavior_patterns(df)
    fig.write_html('plots/user_behavior_patterns.html')
    
    print("Creating traffic analysis...")
    fig = plot_traffic_analysis(df)
    fig.write_html('plots/traffic_analysis.html')
    
    print("All visualizations generated successfully!")

if __name__ == "__main__":
    main() 