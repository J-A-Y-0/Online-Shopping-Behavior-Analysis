import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_conversion_by_category(df, category_col, title=None):
    """
    Plot conversion rates by a categorical variable
    """
    conversion_rates = df.groupby(category_col)['Revenue'].agg(['count', 'mean']).reset_index()
    conversion_rates['mean'] = conversion_rates['mean'] * 100
    
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(data=conversion_rates, x=category_col, y='mean')
    plt.title(title or f'Conversion Rate by {category_col}')
    plt.ylabel('Conversion Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_temporal_patterns(df):
    """
    Plot temporal patterns in the data
    """
    # Monthly patterns
    monthly_metrics = df.groupby('Month').agg({
        'Revenue': 'mean',
        'PageValues': 'mean',
        'BounceRates': 'mean'
    }).reset_index()
    
    # Order months correctly
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_metrics['Month'] = pd.Categorical(monthly_metrics['Month'], categories=month_order, ordered=True)
    monthly_metrics = monthly_metrics.sort_values('Month')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add conversion rate
    fig.add_trace(
        go.Scatter(x=monthly_metrics['Month'], y=monthly_metrics['Revenue']*100,
                  name="Conversion Rate (%)", line=dict(color='blue')),
        secondary_y=False
    )
    
    # Add page values
    fig.add_trace(
        go.Scatter(x=monthly_metrics['Month'], y=monthly_metrics['PageValues'],
                  name="Page Values", line=dict(color='red')),
        secondary_y=True
    )
    
    fig.update_layout(
        title='Monthly Trends: Conversion Rate and Page Values',
        xaxis_title='Month',
        yaxis_title='Conversion Rate (%)',
        yaxis2_title='Page Values'
    )
    
    return fig

def plot_correlation_matrix(df, figsize=(12, 8)):
    """
    Plot correlation matrix heatmap
    """
    # Calculate correlations
    correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    return plt.gcf()

def plot_page_behavior_analysis(df):
    """
    Create visualizations for page behavior analysis
    """
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=('Page Values by Revenue',
                                     'Bounce Rates by Revenue',
                                     'Exit Rates by Revenue',
                                     'Page Values vs. Bounce Rates'))
    
    # Box plots for Page Values, Bounce Rates, and Exit Rates
    fig.add_trace(
        go.Box(x=df['Revenue'].map({0: 'No Purchase', 1: 'Purchase'}),
               y=df['PageValues'],
               name='Page Values'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Box(x=df['Revenue'].map({0: 'No Purchase', 1: 'Purchase'}),
               y=df['BounceRates'],
               name='Bounce Rates'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Box(x=df['Revenue'].map({0: 'No Purchase', 1: 'Purchase'}),
               y=df['ExitRates'],
               name='Exit Rates'),
        row=2, col=1
    )
    
    # Scatter plot of Page Values vs. Bounce Rates
    fig.add_trace(
        go.Scatter(x=df['BounceRates'],
                  y=df['PageValues'],
                  mode='markers',
                  marker=dict(color=df['Revenue'],
                            colorscale='Viridis',
                            showscale=True),
                  name='Page Values vs. Bounce Rates'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Page Behavior Analysis")
    return fig

def plot_user_behavior_patterns(df):
    """
    Create visualizations for user behavior patterns
    """
    # Calculate average metrics by visitor type
    visitor_metrics = df.groupby('VisitorType').agg({
        'Revenue': 'mean',
        'PageValues': 'mean',
        'BounceRates': 'mean',
        'ExitRates': 'mean'
    }).reset_index()
    
    # Create subplots
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=('Conversion Rate by Visitor Type',
                                     'Page Values by Visitor Type',
                                     'Bounce Rates by Visitor Type',
                                     'Exit Rates by Visitor Type'))
    
    # Add traces for each metric
    metrics = ['Revenue', 'PageValues', 'BounceRates', 'ExitRates']
    positions = [(1,1), (1,2), (2,1), (2,2)]
    
    for metric, pos in zip(metrics, positions):
        fig.add_trace(
            go.Bar(x=visitor_metrics['VisitorType'],
                  y=visitor_metrics[metric] * (100 if metric == 'Revenue' else 1),
                  name=metric),
            row=pos[0], col=pos[1]
        )
    
    fig.update_layout(height=800, title_text="User Behavior Patterns by Visitor Type")
    return fig

def plot_traffic_analysis(df):
    """
    Create visualizations for traffic analysis
    """
    # Calculate metrics by traffic type
    traffic_metrics = df.groupby('TrafficType').agg({
        'Revenue': 'mean',
        'PageValues': 'mean',
        'BounceRates': 'mean'
    }).reset_index()
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Bar(x=traffic_metrics['TrafficType'],
               y=traffic_metrics['Revenue']*100,
               name="Conversion Rate (%)"),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=traffic_metrics['TrafficType'],
                  y=traffic_metrics['PageValues'],
                  name="Page Values",
                  line=dict(color='red')),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title='Traffic Analysis: Conversion Rate and Page Values by Traffic Type',
        xaxis_title='Traffic Type',
        barmode='group'
    )
    
    fig.update_yaxes(title_text="Conversion Rate (%)", secondary_y=False)
    fig.update_yaxes(title_text="Page Values", secondary_y=True)
    
    return fig 