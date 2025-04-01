import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

def select_features(X, y, k=10):
    """Select top k features based on ANOVA F-value"""
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Get feature scores
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    }).sort_values('Score', ascending=False)
    
    print("\nTop features by F-score:")
    print(feature_scores.head(k))
    
    return X_selected, selected_features, feature_scores

def remove_low_variance_features(X, threshold=0.01):
    """Remove features with low variance"""
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Get feature variances
    variances = pd.DataFrame({
        'Feature': X.columns,
        'Variance': selector.variances_
    }).sort_values('Variance', ascending=False)
    
    print("\nFeatures removed due to low variance:")
    print(variances[variances['Variance'] < threshold])
    
    return X_selected, selected_features, variances

def evaluate_model_performance(model, X_train, X_test, y_train, y_test):
    """Comprehensive model evaluation"""
    # Train the model
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    
    # Plot ROC curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Plot Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('plots/model_performance.png')
    plt.close()
    
    # Print cross-validation results
    print("\nCross-validation ROC-AUC scores:")
    print(f"Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'cv_scores': cv_scores
    }

def analyze_feature_importance(model, feature_names):
    """Analyze feature importance from the model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("Model doesn't provide feature importance information")
        return None
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(15), x='Importance', y='Feature')
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()
    
    return feature_importance

def generate_recommendations(model, feature_importance, cluster_info=None):
    """Generate personalized recommendations based on model insights"""
    recommendations = []
    
    # Add recommendations based on feature importance
    top_features = feature_importance.head(5)['Feature'].tolist()
    recommendations.append("Key factors influencing purchase decisions:")
    for feature in top_features:
        recommendations.append(f"- Focus on optimizing {feature}")
    
    # Add cluster-based recommendations if available
    if cluster_info is not None:
        recommendations.append("\nCustomer segment recommendations:")
        for cluster in cluster_info.index:
            conv_rate = cluster_info.loc[cluster, 'ConversionRate']
            if conv_rate > 50:
                recommendations.append(f"- Cluster {cluster}: High-value segment, focus on retention")
            elif conv_rate > 20:
                recommendations.append(f"- Cluster {cluster}: Medium-value segment, focus on upselling")
            else:
                recommendations.append(f"- Cluster {cluster}: Low-value segment, focus on engagement")
    
    return recommendations

def main():
    """Main function for model evaluation"""
    # Load the preprocessed data
    df = pd.read_csv('data/online_shoppers_intention.csv')
    
    # Prepare features and target
    X = df.select_dtypes(include=['float64', 'int64']).copy()
    y = df['Revenue']
    
    # Feature selection
    X_selected, selected_features, feature_scores = select_features(X, y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate models (example with Random Forest)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    
    # Evaluate model
    performance_metrics = evaluate_model_performance(
        model, X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(model, selected_features)
    
    # Generate recommendations
    recommendations = generate_recommendations(model, feature_importance)
    
    # Save recommendations
    with open('reports/model_recommendations.txt', 'w') as f:
        f.write('\n'.join(recommendations))
    
    print("\nAnalysis complete! Check the 'plots' and 'reports' directories for results.")

if __name__ == "__main__":
    main() 