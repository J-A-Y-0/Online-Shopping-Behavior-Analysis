import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_describe_data(file_path):
    """
    Load the dataset and provide basic description
    """
    df = pd.read_csv(file_path)
    
    print("Dataset Shape:", df.shape)
    print("\nFeature Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    return df

def prepare_features(df):
    """
    Prepare features for modeling
    """
    # Create new features
    df['TotalDuration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']
    df['TotalPages'] = df['Administrative'] + df['Informational'] + df['ProductRelated']
    df['AvgDurationPerPage'] = df['TotalDuration'] / df['TotalPages']
    df['AvgDurationPerPage'] = df['AvgDurationPerPage'].fillna(0)
    
    # Convert Month to numerical (cyclic encoding)
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df['Month_Num'] = df['Month'].map(lambda x: month_order.index(x))
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month_Num']/12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month_Num']/12)
    
    return df

def plot_feature_distributions(df, target_col='Revenue'):
    """
    Plot distribution of numerical features by target variable
    """
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = numerical_cols[numerical_cols != target_col]
    
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.ravel()
    
    for idx, col in enumerate(numerical_cols):
        sns.histplot(data=df, x=col, hue=target_col, multiple="stack", ax=axes[idx])
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for idx in range(len(numerical_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig

def prepare_data_for_modeling(df, target_col='Revenue', test_size=0.2, random_state=42):
    """
    Prepare data for modeling including train-test split and SMOTE
    """
    # Separate features and target
    X = df.drop([target_col, 'Month'], axis=1)
    y = df[target_col]
    
    # Split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    return (X_train_balanced, X_test_scaled, y_train_balanced, y_test, scaler)

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance and create visualizations
    """
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Print classification report
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.show()
    
    return {
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred),
        'roc_auc': roc_auc
    }

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance for tree-based models
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(top_n), importances[indices][:top_n])
        plt.xticks(range(top_n), [feature_names[i] for i in indices][:top_n], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
    else:
        print("This model doesn't support feature importance visualization") 