import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

def get_latest_model():
    """Get the latest model from the models directory"""
    try:
        model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
        if not model_files:
            return None
        
        # Get the most recent model
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join('models', x)))
        return os.path.join('models', latest_model)
    except Exception as e:
        logging.error(f"Error finding latest model: {str(e)}")
        return None

def load_model():
    """Load the trained model and scaler with error handling"""
    try:
        # Try to load the latest model
        model_path = get_latest_model()
        if model_path and os.path.exists(model_path):
            logging.info(f"Loading model from {model_path}")
            model = joblib.load(model_path)
            scaler = joblib.load('models/scaler.joblib')
            return model, scaler
        
        logging.warning("No existing model found. Training new model...")
        return train_new_model()
    
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        st.error("Error loading model. Please try again later.")
        return None, None

def train_new_model():
    """Train a new model if none exists"""
    try:
        if not os.path.exists('data/online_shoppers_intention.csv'):
            raise FileNotFoundError("Training data not found")
        
        df = pd.read_csv('data/online_shoppers_intention.csv')
        X = df.select_dtypes(include=['float64', 'int64']).copy()
        y = df['Revenue']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_scaled, y)
        
        # Save model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'models/random_forest_{timestamp}.joblib'
        scaler_path = 'models/scaler.joblib'
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logging.info(f"New model trained and saved to {model_path}")
        return model, scaler
    
    except Exception as e:
        logging.error(f"Error training new model: {str(e)}")
        st.error("Error training model. Please check the data and try again.")
        return None, None

def validate_input(data):
    """Validate input data"""
    try:
        # Check value ranges
        if any(v < 0 for v in data):
            return False, "All values must be non-negative"
        
        # Check bounce and exit rates
        if data[6] > 1 or data[7] > 1:
            return False, "Bounce and Exit rates must be between 0 and 1"
        
        # Check special day
        if data[9] > 1:
            return False, "Special day must be between 0 and 1"
        
        return True, ""
    except Exception as e:
        return False, str(e)

def main():
    st.set_page_config(
        page_title="Online Purchase Intent Predictor",
        page_icon="🛒",
        layout="wide"
    )
    
    st.title('Online Purchase Intent Predictor')
    st.write("""
    This app predicts the likelihood of a purchase based on user session data.
    Enter the session details below to get a prediction.
    """)
    
    # Load model
    model, scaler = load_model()
    if model is None or scaler is None:
        st.error("Could not load or train model. Please check the logs.")
        return
    
    # Create input form
    st.header('Session Information')
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            administrative = st.number_input('Administrative Pages', min_value=0, value=0)
            administrative_duration = st.number_input('Administrative Duration (s)', min_value=0.0, value=0.0)
            informational = st.number_input('Informational Pages', min_value=0, value=0)
            informational_duration = st.number_input('Informational Duration (s)', min_value=0.0, value=0.0)
            product_related = st.number_input('Product Related Pages', min_value=0, value=0)
        
        with col2:
            product_related_duration = st.number_input('Product Related Duration (s)', min_value=0.0, value=0.0)
            bounce_rate = st.number_input('Bounce Rate', min_value=0.0, max_value=1.0, value=0.0)
            exit_rate = st.number_input('Exit Rate', min_value=0.0, max_value=1.0, value=0.0)
            page_value = st.number_input('Page Value', min_value=0.0, value=0.0)
            special_day = st.number_input('Special Day', min_value=0.0, max_value=1.0, value=0.0)
        
        submitted = st.form_submit_button("Predict Purchase Intent")
    
    if submitted:
        # Prepare input data
        input_data = np.array([
            administrative, administrative_duration,
            informational, informational_duration,
            product_related, product_related_duration,
            bounce_rate, exit_rate, page_value, special_day
        ])
        
        # Validate input
        is_valid, error_message = validate_input(input_data)
        if not is_valid:
            st.error(f"Invalid input: {error_message}")
            return
        
        try:
            # Scale input
            input_scaled = scaler.transform(input_data.reshape(1, -1))
            
            # Get prediction and probability
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            # Display results
            st.header('Prediction Results')
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction:
                    st.success('Purchase Likely! 🎉')
                else:
                    st.warning('Purchase Unlikely 😕')
            
            with col2:
                st.metric(
                    label="Purchase Probability",
                    value=f"{probability:.1%}"
                )
            
            # Display feature importance
            st.header('Feature Importance')
            feature_names = [
                'Administrative', 'Administrative Duration',
                'Informational', 'Informational Duration',
                'Product Related', 'Product Related Duration',
                'Bounce Rate', 'Exit Rate', 'Page Value', 'Special Day'
            ]
            
            importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.bar_chart(importances.set_index('Feature'))
            
            # Log prediction
            logging.info(f"Prediction made: {prediction} with probability {probability:.2f}")
            
        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            st.error("Error making prediction. Please try again.")

if __name__ == '__main__':
    main() 