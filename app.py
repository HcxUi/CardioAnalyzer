import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

import data_preprocessing as dp
import data_visualization as dv
import model_training as mt
import model_evaluation as me
import utils
import database

# Set page configuration with increased performance
st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS to speed up interactions
st.markdown("""
<style>
    button {
        transition: all 0.1s ease;
    }
    .stButton>button {
        transition-duration: 0.1s;
    }
    div.stButton > button:first-child {
        font-weight: bold;
        height: 2.5em;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data with optimized caching
@st.cache_data(ttl=3600, show_spinner="Loading dataset...")
def load_data():
    data = pd.read_csv('attached_assets/cardio_train.csv', sep=';')
    return data

def main():
    # Title and introduction
    st.title("Cardiovascular Disease Prediction System")
    st.markdown("""
    This application analyzes patient data to predict cardiovascular disease risk using multiple machine learning algorithms.
    Upload your dataset or use the sample dataset to explore the analysis.
    """)
    
    # For Netlify function handler
    return {
        'statusCode': 200,
        'body': st.get_component_value(),
        'headers': {
            'Content-Type': 'text/html'
        }
    }

# This allows the app to run both locally and on Streamlit Cloud
if __name__ == "__main__":
    main()

# Precompute commonly used model features for faster access
@st.cache_data(ttl=3600)
def get_model_features():
    return ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
            'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']

try:
    data = load_data()
    st.success("Dataset loaded successfully!")
    
    # Create essential derived features regardless of navigation
    if 'age_years' not in data.columns:
        data['age_years'] = dp.convert_age_to_years(data['age'])
    
    if 'bmi' not in data.columns:
        data['bmi'] = dp.calculate_bmi(data['weight'], data['height'])
        
    if 'bmi_category' not in data.columns:
        data['bmi_category'] = dp.categorize_bmi(data['bmi'])
        
    if 'bp_category' not in data.columns:
        data['bp_category'] = dp.categorize_blood_pressure(data['ap_hi'], data['ap_lo'])
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Data Preprocessing", "Data Visualization", 
                                "Feature Correlation", "Model Training", "Model Comparison", 
                                "Prediction", "Prediction History"])

# Data Overview
if page == "Data Overview":
    st.header("Data Overview")
    
    st.subheader("First Few Rows of the Dataset")
    st.dataframe(data.head())
    
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Number of Samples: {data.shape[0]}")
        st.write(f"Number of Features: {data.shape[1] - 1}")
    with col2:
        st.write(f"Target Distribution:")
        fig = px.pie(values=data['cardio'].value_counts().values, 
                    names=data['cardio'].value_counts().index.map({0: 'No Disease', 1: 'Disease'}),
                    title='Cardiovascular Disease Distribution')
        st.plotly_chart(fig)
    
    st.subheader("Statistical Summary")
    st.dataframe(data.describe())
    
    with st.expander("Data Dictionary"):
        st.markdown("""
        - **id**: Unique identifier
        - **age**: Age in days
        - **gender**: Gender (1: female, 2: male)
        - **height**: Height in cm
        - **weight**: Weight in kg
        - **ap_hi**: Systolic blood pressure
        - **ap_lo**: Diastolic blood pressure
        - **cholesterol**: Cholesterol (1: normal, 2: above normal, 3: well above normal)
        - **gluc**: Glucose (1: normal, 2: above normal, 3: well above normal)
        - **smoke**: Smoking status (0: non-smoker, 1: smoker)
        - **alco**: Alcohol intake (0: no alcohol, 1: alcohol consumption)
        - **active**: Physical activity (0: inactive, 1: active)
        - **cardio**: Target variable - presence of cardiovascular disease (0: no, 1: yes)
        """)

# Data Preprocessing
elif page == "Data Preprocessing":
    st.header("Data Preprocessing")
    
    # Create tabs for different preprocessing steps
    preprocessing_tab = st.tabs(["Feature Engineering", "Missing Values", "Outliers", "Feature Scaling"])
    
    with preprocessing_tab[0]:
        st.subheader("Feature Engineering")
        
        st.write("Converting age from days to years:")
        data['age_years'] = dp.convert_age_to_years(data['age'])
        st.dataframe(data[['age', 'age_years']].head())
        
        st.write("Creating BMI feature:")
        data['bmi'] = dp.calculate_bmi(data['weight'], data['height'])
        st.dataframe(data[['height', 'weight', 'bmi']].head())
        
        # Show BMI category distribution
        data['bmi_category'] = dp.categorize_bmi(data['bmi'])
        fig = px.histogram(data, x='bmi_category', color='cardio', 
                         barmode='group', title='BMI Categories vs Cardiovascular Disease')
        st.plotly_chart(fig)
        
        # Blood pressure categories
        data['bp_category'] = dp.categorize_blood_pressure(data['ap_hi'], data['ap_lo'])
        fig = px.histogram(data, x='bp_category', color='cardio', 
                         barmode='group', title='Blood Pressure Categories vs Cardiovascular Disease')
        st.plotly_chart(fig)
    
    with preprocessing_tab[1]:
        st.subheader("Missing Values Analysis")
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            st.write("Missing values in each column:")
            st.dataframe(missing_values[missing_values > 0])
            
            st.write("Handling missing values...")
            data = dp.handle_missing_values(data)
            st.success("Missing values handled!")
        else:
            st.success("No missing values found in the dataset!")
    
    with preprocessing_tab[2]:
        st.subheader("Outlier Detection and Treatment")
        
        # Display outliers before handling
        numerical_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
        
        with st.expander("View Outlier Visualization"):
            for col in numerical_cols:
                fig = px.box(data, y=col, title=f'Box Plot for {col}')
                st.plotly_chart(fig)
        
        # Handle outliers
        st.write("Detecting and handling outliers...")
        data_cleaned, outliers_removed = dp.handle_outliers(data, numerical_cols)
        
        st.write(f"Number of outliers removed: {outliers_removed}")
        
        # Display data after outlier handling
        with st.expander("View data after outlier handling"):
            st.dataframe(data_cleaned.head())
            
            for col in numerical_cols:
                fig = px.box(data_cleaned, y=col, title=f'Box Plot for {col} (After Outlier Handling)')
                st.plotly_chart(fig)
        
        # Assign cleaned data back to data
        data = data_cleaned
    
    with preprocessing_tab[3]:
        st.subheader("Feature Scaling")
        
        # Select features for scaling
        features_to_scale = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
        
        # Scale the features
        st.write("Standardizing numerical features...")
        X_scaled, scaler = dp.scale_features(data[features_to_scale])
        
        # Show before and after scaling
        comparison_data = pd.DataFrame({
            'Original': data[features_to_scale[0]],
            'Scaled': X_scaled[:, 0]
        })
        
        fig = px.histogram(comparison_data, barmode='overlay', 
                         title=f'Before vs After Scaling ({features_to_scale[0]})')
        st.plotly_chart(fig)
        
        # Create a dataframe with the scaled features
        scaled_data = pd.DataFrame(X_scaled, columns=features_to_scale)
        
        # Keep track of the scaled data
        data_scaled = data.copy()
        data_scaled[features_to_scale] = scaled_data
        
        st.success("Features scaled successfully!")
        with st.expander("View scaled data"):
            st.dataframe(data_scaled[features_to_scale].head())

# Data Visualization
elif page == "Data Visualization":
    st.header("Data Visualization and Analysis")
    
    # Create tabs for different visualization categories
    viz_tabs = st.tabs(["Demographics", "Health Indicators", "Lifestyle Factors", "Multivariate Analysis"])
    
    with viz_tabs[0]:
        st.subheader("Demographic Analysis")
        
        # Age distribution
        fig = dv.plot_age_distribution(data)
        st.plotly_chart(fig)
        
        # Gender distribution
        fig = dv.plot_gender_distribution(data)
        st.plotly_chart(fig)
        
        # Age vs Gender with Disease
        fig = dv.plot_age_gender_disease(data)
        st.plotly_chart(fig)
    
    with viz_tabs[1]:
        st.subheader("Health Indicators Analysis")
        
        # Blood pressure distribution
        fig = dv.plot_blood_pressure_distribution(data)
        st.plotly_chart(fig)
        
        # Cholesterol and Glucose
        fig = dv.plot_cholesterol_glucose(data)
        st.plotly_chart(fig)
        
        # BMI distribution
        fig = dv.plot_bmi_distribution(data)
        st.plotly_chart(fig)
    
    with viz_tabs[2]:
        st.subheader("Lifestyle Factors Analysis")
        
        # Smoking, alcohol and physical activity
        fig = dv.plot_lifestyle_factors(data)
        st.plotly_chart(fig)
        
        # Combination of lifestyle factors
        fig = dv.plot_lifestyle_combination(data)
        st.plotly_chart(fig)
    
    with viz_tabs[3]:
        st.subheader("Multivariate Analysis")
        
        # Scatter plot matrix of key variables
        fig = dv.plot_scatter_matrix(data)
        st.plotly_chart(fig)
        
        # 3D visualization
        fig = dv.plot_3d_visualization(data)
        st.plotly_chart(fig)

# Feature Correlation
elif page == "Feature Correlation":
    st.header("Feature Correlation Analysis")
    
    # Select features for correlation
    numeric_features = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 
                      'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
    
    # Calculate correlation matrix
    corr_matrix = data[numeric_features].corr()
    
    # Plot correlation heatmap
    fig = px.imshow(corr_matrix, text_auto=True, 
                   title='Correlation Matrix of Features',
                   color_continuous_scale='RdBu_r')
    st.plotly_chart(fig)
    
    # Feature importance analysis
    st.subheader("Feature Importance Analysis")
    
    # Prepare data for feature importance
    X = data[numeric_features].drop(columns=['cardio'])
    y = data['cardio']
    
    # Get feature importance from random forest
    feature_importance = utils.get_feature_importance(X, y)
    
    # Plot feature importance
    fig = px.bar(x=feature_importance, y=X.columns, 
               title='Feature Importance', 
               labels={'x': 'Importance', 'y': 'Features'},
               orientation='h')
    st.plotly_chart(fig)
    
    # Pair plot for top features
    st.subheader("Relationships Between Top Features")
    
    # Get top 4 features
    top_features = list(X.columns[np.argsort(feature_importance)[-4:]])
    top_features.append('cardio')
    
    # Create pair plot
    fig = dv.plot_top_features_pair(data, top_features)
    st.plotly_chart(fig)

# Model Training
elif page == "Model Training":
    st.header("Model Training")
    
    # Prepare data for modeling
    st.subheader("Data Preparation for Modeling")
    
    # Select features for modeling
    model_features = get_model_features()
    
    # Create train-test split
    X = data[model_features]
    y = data['cardio']
    
    test_size = st.slider("Select test size percentage:", 10, 40, 20) / 100
    random_state = st.slider("Select random state:", 0, 100, 42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                     random_state=random_state)
    
    st.write(f"Training set size: {X_train.shape[0]} samples")
    st.write(f"Testing set size: {X_test.shape[0]} samples")
    
    # Model selection and hyperparameters
    st.subheader("Model Selection and Training")
    
    # User selects which models to train
    models_to_train = st.multiselect(
        "Select models to train:",
        ["Logistic Regression", "K-Nearest Neighbors", "Decision Tree", 
         "Random Forest", "Support Vector Machine"],
        ["Logistic Regression", "Random Forest"]
    )
    
    # Train models button
    if st.button("Train Selected Models"):
        with st.spinner("Training models..."):
            # Dictionary to store trained models and their performance
            trained_models = {}
            
            # Progress bar
            progress_bar = st.progress(0)
            
            for i, model_name in enumerate(models_to_train):
                # Update progress bar
                progress = (i) / len(models_to_train)
                progress_bar.progress(progress)
                
                st.write(f"Training {model_name}...")
                
                # Train the model
                if model_name == "Logistic Regression":
                    model, metrics = mt.train_logistic_regression(X_train, y_train, X_test, y_test)
                elif model_name == "K-Nearest Neighbors":
                    model, metrics = mt.train_knn(X_train, y_train, X_test, y_test)
                elif model_name == "Decision Tree":
                    model, metrics = mt.train_decision_tree(X_train, y_train, X_test, y_test)
                elif model_name == "Random Forest":
                    model, metrics = mt.train_random_forest(X_train, y_train, X_test, y_test)
                elif model_name == "Support Vector Machine":
                    model, metrics = mt.train_svm(X_train, y_train, X_test, y_test)
                
                # Store the model and metrics
                trained_models[model_name] = {
                    'model': model,
                    'metrics': metrics
                }
                
                # Display model metrics
                st.subheader(f"{model_name} Performance")
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                    'Value': [metrics['accuracy'], metrics['precision'], 
                             metrics['recall'], metrics['f1']]
                })
                st.dataframe(metrics_df)
                
                # Plot confusion matrix
                fig = me.plot_confusion_matrix(metrics['confusion_matrix'], model_name)
                st.plotly_chart(fig)
                
                # Plot ROC curve if applicable
                if 'roc_auc' in metrics:
                    fig = me.plot_roc_curve(metrics['fpr'], metrics['tpr'], 
                                          metrics['roc_auc'], model_name)
                    st.plotly_chart(fig)
            
            # Complete progress bar
            progress_bar.progress(1.0)
            
            # Save trained models to session state
            st.session_state.trained_models = trained_models
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            
            st.success("Model training completed!")

# Model Comparison
elif page == "Model Comparison":
    st.header("Model Comparison")
    
    # Check if models have been trained
    if 'trained_models' not in st.session_state:
        st.warning("Please train models first in the Model Training section.")
        st.stop()
    
    # Get trained models from session state
    trained_models = st.session_state.trained_models
    
    # Compare model performance
    st.subheader("Performance Comparison")
    
    # Create a dataframe with model metrics
    model_metrics = []
    for model_name, model_data in trained_models.items():
        metrics = model_data['metrics']
        model_metrics.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1'],
            'ROC AUC': metrics.get('roc_auc', None)
        })
    
    metrics_df = pd.DataFrame(model_metrics)
    st.dataframe(metrics_df)
    
    # Plot comparison bar chart
    fig = me.plot_model_comparison(metrics_df)
    st.plotly_chart(fig)
    
    # Find the best model
    metric_choice = st.selectbox(
        "Select metric for best model selection:",
        ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        index=0
    )
    
    # Get the best model
    best_model_name = metrics_df.loc[metrics_df[metric_choice].idxmax(), 'Model']
    best_model = trained_models[best_model_name]['model']
    
    st.success(f"The best model based on {metric_choice} is: {best_model_name}")
    
    # Save best model to session state
    st.session_state.best_model = best_model
    st.session_state.best_model_name = best_model_name
    
    # Feature importance of the best model (if applicable)
    try:
        if hasattr(best_model, 'feature_importances_'):
            st.subheader(f"Feature Importance of {best_model_name}")
            
            # Get feature names and importance
            feature_names = st.session_state.X_test.columns
            feature_importance = best_model.feature_importances_
            
            # Sort feature importance
            sorted_idx = np.argsort(feature_importance)
            
            # Plot feature importance
            fig = px.bar(
                x=feature_importance[sorted_idx],
                y=[feature_names[i] for i in sorted_idx],
                title=f'Feature Importance ({best_model_name})',
                labels={'x': 'Importance', 'y': 'Feature'},
                orientation='h'
            )
            st.plotly_chart(fig)
    except:
        st.info(f"Feature importance is not available for {best_model_name}.")
    
    # Cross-validation analysis
    st.subheader("Cross-Validation Analysis")
    
    # Check if cv_results are in session state
    if 'cv_results' not in st.session_state:
        # Perform cross-validation
        with st.spinner("Performing cross-validation..."):
            cv_results = me.perform_cross_validation(
                trained_models, 
                data[model_features], 
                data['cardio']
            )
            st.session_state.cv_results = cv_results
    else:
        cv_results = st.session_state.cv_results
    
    # Display cross-validation results
    cv_df = pd.DataFrame({
        'Model': list(cv_results.keys()),
        'Mean CV Accuracy': [cv_results[model]['mean_accuracy'] for model in cv_results],
        'Std CV Accuracy': [cv_results[model]['std_accuracy'] for model in cv_results]
    })
    
    st.dataframe(cv_df)
    
    # Plot cross-validation results
    fig = me.plot_cv_results(cv_df)
    st.plotly_chart(fig)

# Prediction
elif page == "Prediction":
    st.header("Cardiovascular Disease Prediction")
    
    # Check if models have been trained
    if 'trained_models' not in st.session_state:
        st.warning("Please train models first in the Model Training section.")
        st.stop()
    
    # Create tabs for prediction
    pred_tabs = st.tabs(["Single Prediction", "Batch Prediction"])
    
    with pred_tabs[0]:
        st.subheader("Predict Cardiovascular Disease for a Single Patient")
        
        # Create form for user input
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
                gender = st.radio("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
                height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
                weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
                bmi = weight / ((height/100) ** 2)
                st.info(f"BMI: {bmi:.2f} ({dp.categorize_bmi(pd.Series([bmi]))[0]})")
            
            with col2:
                ap_hi = st.number_input("Systolic Blood Pressure", min_value=80, max_value=250, value=120)
                ap_lo = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=180, value=80)
                cholesterol = st.radio("Cholesterol Level", [1, 2, 3], 
                                     format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
                gluc = st.radio("Glucose Level", [1, 2, 3], 
                              format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
            
            col3, col4 = st.columns(2)
            
            with col3:
                smoke = st.checkbox("Smoker")
                alco = st.checkbox("Alcohol Consumption")
            
            with col4:
                active = st.checkbox("Physically Active", value=True)
                
            # Model selection
            if 'best_model' in st.session_state:
                default_model = st.session_state.best_model
            else:
                default_model = next(iter(st.session_state.trained_models.keys()))
                
            model_name = st.selectbox(
                "Select model for prediction",
                list(st.session_state.trained_models.keys()),
                index=list(st.session_state.trained_models.keys()).index(default_model) if 'best_model' in st.session_state else 0
            )
            
            # Submit button
            submit_button = st.form_submit_button("Predict")
        
        # Make prediction when form is submitted
        if submit_button:
            # Create feature vector
            age_days = age * 365.25
            features = {
                'id': f"manual_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'age': age_days,
                'age_years': age,
                'gender': gender,
                'height': height,
                'weight': weight,
                'bmi': bmi,
                'ap_hi': ap_hi,
                'ap_lo': ap_lo,
                'cholesterol': cholesterol,
                'gluc': gluc,
                'smoke': int(smoke),
                'alco': int(alco),
                'active': int(active)
            }
            
            # Convert to DataFrame
            X_pred = pd.DataFrame([features])
            
            # Get model features from cached function
            model_features = get_model_features()
            X_pred_model = X_pred[model_features]
            
            # Get selected model
            model = st.session_state.trained_models[model_name]['model']
            
            # Make prediction
            prediction = model.predict(X_pred_model)[0]
            
            # Get prediction probability if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_pred_model)[0]
                probability = probabilities[1]  # Probability of the positive class (1)
            else:
                probability = 0.5
            
            # Save prediction to database
            db_id = database.save_prediction(features, prediction, probability, model_name)
            
            # Display prediction
            st.subheader("Prediction Result")
            
            if prediction == 1:
                st.error(f"⚠️ High Risk of Cardiovascular Disease (Probability: {probability:.2f})")
            else:
                st.success(f"✅ Low Risk of Cardiovascular Disease (Probability: {probability:.2f})")
            
            # Display radar chart of risk factors
            st.subheader("Risk Factor Analysis")
            
            # Create risk factors values
            risk_factors = {
                'Age': (age - 18) / (100 - 18),  # Normalize between 18 and 100
                'BMI': (bmi - 18.5) / (40 - 18.5) if bmi >= 18.5 else 0,  # Normalize between 18.5 and 40
                'Blood Pressure': ((ap_hi - 90) / (180 - 90) + (ap_lo - 60) / (120 - 60)) / 2,  # Average of systolic and diastolic
                'Cholesterol': (cholesterol - 1) / 2,  # Normalize between 1 and 3
                'Glucose': (gluc - 1) / 2,  # Normalize between 1 and 3
                'Lifestyle': (int(smoke) + int(alco) + (1 - int(active))) / 3  # Average of risk factors
            }
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=list(risk_factors.values()),
                theta=list(risk_factors.keys()),
                fill='toself',
                name='Risk Factors'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=False
            )
            
            st.plotly_chart(fig)
            
            # Provide explanation
            st.subheader("Risk Explanation")
            
            high_risk_factors = []
            
            if age > 50:
                high_risk_factors.append(f"Age ({age} years) - Risk of cardiovascular disease increases with age")
            
            if bmi >= 25:
                high_risk_factors.append(f"BMI ({bmi:.1f}) - {'Overweight' if bmi < 30 else 'Obese'}, which increases cardiovascular risks")
            
            if ap_hi >= 140 or ap_lo >= 90:
                high_risk_factors.append(f"Blood Pressure ({ap_hi}/{ap_lo}) - Indicates hypertension, a major risk factor")
            
            if cholesterol > 1:
                high_risk_factors.append(f"Cholesterol (Level {cholesterol}) - {'Above normal' if cholesterol == 2 else 'Well above normal'}, increasing risk of heart disease")
            
            if gluc > 1:
                high_risk_factors.append(f"Glucose (Level {gluc}) - {'Above normal' if gluc == 2 else 'Well above normal'}, indicating possible diabetes or pre-diabetes")
            
            if smoke:
                high_risk_factors.append("Smoking - Significantly increases risk of heart disease")
            
            if alco:
                high_risk_factors.append("Alcohol consumption - Can contribute to heart problems when excessive")
            
            if not active:
                high_risk_factors.append("Physical inactivity - Lack of exercise increases cardiovascular risk")
            
            if high_risk_factors:
                st.write("Key risk factors identified:")
                for factor in high_risk_factors:
                    st.markdown(f"- {factor}")
            else:
                st.write("No major risk factors identified. Continue maintaining a healthy lifestyle.")
    
    with pred_tabs[1]:
        st.subheader("Batch Prediction")
        st.write("Upload a CSV file with patient data to make predictions for multiple patients.")
        
        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Load data
                batch_data = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(batch_data.head())
                
                # Check required columns
                required_columns = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                                    'cholesterol', 'gluc', 'smoke', 'alco', 'active']
                
                missing_columns = [col for col in required_columns if col not in batch_data.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    st.stop()
                
                # Select model for prediction
                batch_model_name = st.selectbox(
                    "Select model for batch prediction",
                    list(st.session_state.trained_models.keys()),
                    key="batch_model_selector"
                )
                
                # Process button
                if st.button("Process Batch"):
                    with st.spinner("Processing batch predictions..."):
                        # Preprocess data
                        batch_data_copy = batch_data.copy()
                        
                        # Create derived features
                        if 'age_years' not in batch_data_copy.columns:
                            batch_data_copy['age_years'] = dp.convert_age_to_years(batch_data_copy['age'])
                        
                        if 'bmi' not in batch_data_copy.columns:
                            batch_data_copy['bmi'] = dp.calculate_bmi(batch_data_copy['weight'], batch_data_copy['height'])
                        
                        # Prepare features for model
                        model_features = get_model_features()
                        X_batch = batch_data_copy[model_features]
                        
                        # Get model
                        model = st.session_state.trained_models[batch_model_name]['model']
                        
                        # Make predictions
                        predictions = model.predict(X_batch)
                        
                        # Get prediction probabilities if available
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(X_batch)[:, 1]  # Probability of the positive class (1)
                        else:
                            probabilities = [0.5] * len(predictions)
                        
                        # Add predictions to the dataframe
                        batch_data_copy['predicted_cardio'] = predictions
                        batch_data_copy['prediction_probability'] = probabilities
                        
                        # Save predictions to database if id column is present
                        if 'id' in batch_data.columns:
                            for i, row in batch_data_copy.iterrows():
                                database.save_prediction(row, row['predicted_cardio'], row['prediction_probability'], batch_model_name)
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(batch_data_copy)
                        
                        # Download link for results
                        csv = batch_data_copy.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Summary statistics
                        st.subheader("Summary")
                        total = len(predictions)
                        positive = sum(predictions)
                        negative = total - positive
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Patients", total)
                        col2.metric("Predicted Positive", positive)
                        col3.metric("Predicted Negative", negative)
                        
                        # Plot distribution
                        fig = px.pie(
                            values=[positive, negative],
                            names=['Disease Risk', 'No Disease Risk'],
                            title='Prediction Distribution'
                        )
                        st.plotly_chart(fig)
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.stop()
                
# Prediction History
elif page == "Prediction History":
    st.header("Prediction History and Statistics")
    
    # Create tabs for different views
    history_tabs = st.tabs(["Recent Predictions", "Statistics", "Database Status"])
    
    with history_tabs[0]:
        st.subheader("Recent Prediction Records")
        
        # Get prediction history
        limit = st.slider("Number of records to show", 10, 100, 20)
        history_df = database.get_prediction_history(limit=limit)
        
        if history_df.empty:
            st.info("No prediction records found. Make some predictions first!")
        else:
            # Format the dataframe for display
            display_df = history_df.copy()
            
            # Format timestamp
            if 'timestamp' in display_df.columns:
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Format gender
            if 'gender' in display_df.columns:
                display_df['gender'] = display_df['gender'].apply(lambda x: 'Female' if x == 1 else 'Male')
            
            # Format prediction
            if 'predicted_cardio' in display_df.columns:
                display_df['predicted_cardio'] = display_df['predicted_cardio'].apply(lambda x: 'Yes' if x else 'No')
            
            # Format probability
            if 'prediction_probability' in display_df.columns:
                display_df['prediction_probability'] = display_df['prediction_probability'].apply(lambda x: f"{x:.2f}")
            
            # Display the dataframe
            st.dataframe(display_df)
            
            # Download option
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download History",
                data=csv,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with history_tabs[1]:
        st.subheader("Prediction Statistics")
        
        # Get prediction statistics
        stats = database.get_prediction_stats()
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predictions", stats['total'])
        
        with col2:
            st.metric("Positive Predictions", stats['positive'])
        
        with col3:
            st.metric("Negative Predictions", stats['negative'])
        
        # Plot distribution
        if stats['total'] > 0:
            # Create distribution chart
            fig1 = px.pie(
                values=[stats['positive'], stats['negative']],
                names=['Disease Risk', 'No Disease Risk'],
                title='Overall Prediction Distribution'
            )
            st.plotly_chart(fig1)
            
            # Create model distribution chart if there are models
            if stats['models']:
                model_names = list(stats['models'].keys())
                model_counts = list(stats['models'].values())
                
                fig2 = px.bar(
                    x=model_names,
                    y=model_counts,
                    title='Predictions by Model',
                    labels={'x': 'Model', 'y': 'Number of Predictions'}
                )
                st.plotly_chart(fig2)
        else:
            st.info("No prediction data available for statistics.")
    
    with history_tabs[2]:
        st.subheader("Database Status")
        
        # Check if database is configured
        if database.Session is None:
            st.error("Database is not configured. Prediction storage is disabled.")
            
            # Show database configuration info
            st.write("To enable database functionality, make sure the following environment variables are set:")
            st.code("DATABASE_URL - PostgreSQL connection string")
            
            # Show sample configuration
            st.write("Example configuration:")
            st.code("DATABASE_URL=postgresql://username:password@localhost:5432/database_name")
        else:
            st.success("Database is configured and accessible.")
            
            # Show database info
            st.write("Database Settings:")
            st.json({
                "Tables": ["prediction_results"],
                "Engine": str(database.engine.url).replace(":*****@", ":xxxxx@")  # Hide password
            })
    
    # Check if a model has been trained
    if 'best_model' not in st.session_state:
        st.warning("Please train and compare models first to select the best model.")
        st.stop()
    
    # Get the best model from session state
    best_model = st.session_state.best_model
    best_model_name = st.session_state.best_model_name
    
    st.info(f"Using {best_model_name} for prediction.")
    
    # Create two columns for input methods
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Patient Information")
        
        # Age input (in years)
        age = st.number_input("Age (years)", min_value=10, max_value=100, value=40)
        
        # Gender selection
        gender = st.radio("Gender", ["Female", "Male"])
        gender_code = 1 if gender == "Female" else 2
        
        # Physical measurements
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=165)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        
        # Blood pressure
        ap_hi = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=240, value=120)
        ap_lo = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=160, value=80)
        
        # Other health indicators
        cholesterol = st.selectbox("Cholesterol Level", ["Normal", "Above Normal", "Well Above Normal"])
        cholesterol_code = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[cholesterol]
        
        gluc = st.selectbox("Glucose Level", ["Normal", "Above Normal", "Well Above Normal"])
        gluc_code = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[gluc]
        
        # Lifestyle factors
        smoke = st.checkbox("Smoker")
        alco = st.checkbox("Alcohol Consumption")
        active = st.checkbox("Physically Active")
        
        # Calculate BMI
        bmi = weight / ((height/100)**2)
        
        # Create feature vector
        features = {
            'age_years': age,
            'gender': gender_code,
            'height': height,
            'weight': weight,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': cholesterol_code,
            'gluc': gluc_code,
            'smoke': int(smoke),
            'alco': int(alco),
            'active': int(active),
            'bmi': bmi
        }
        
        input_df = pd.DataFrame([features])
        
        # Make prediction when button is clicked
        if st.button("Predict"):
            with st.spinner("Predicting..."):
                # Make prediction
                prediction = best_model.predict(input_df)[0]
                prediction_proba = best_model.predict_proba(input_df)[0]
                
                # Display result
                st.subheader("Prediction Result")
                
                if prediction == 1:
                    st.error(f"**Cardiovascular Disease Detected** (Confidence: {prediction_proba[1]:.2f})")
                else:
                    st.success(f"**No Cardiovascular Disease Detected** (Confidence: {prediction_proba[0]:.2f})")
                
                # Display risk factors
                st.subheader("Risk Factor Analysis")
                
                risk_factors = []
                
                # Check age
                if age > 50:
                    risk_factors.append("Age above 50")
                
                # Check BMI
                if bmi >= 30:
                    risk_factors.append("Obesity (BMI >= 30)")
                elif bmi >= 25:
                    risk_factors.append("Overweight (BMI >= 25)")
                
                # Check blood pressure
                if ap_hi >= 140 or ap_lo >= 90:
                    risk_factors.append("Hypertension")
                
                # Check cholesterol
                if cholesterol_code > 1:
                    risk_factors.append("Elevated Cholesterol")
                
                # Check glucose
                if gluc_code > 1:
                    risk_factors.append("Elevated Glucose")
                
                # Check lifestyle factors
                if smoke:
                    risk_factors.append("Smoking")
                
                if alco:
                    risk_factors.append("Alcohol Consumption")
                
                if not active:
                    risk_factors.append("Physical Inactivity")
                
                # Display risk factors
                if risk_factors:
                    st.write("Identified Risk Factors:")
                    for factor in risk_factors:
                        st.write(f"- {factor}")
                else:
                    st.write("No significant risk factors identified.")
    
    with col2:
        st.subheader("Prediction Explanation")
        
        # Display BMI information
        st.write(f"**BMI:** {bmi:.2f}")
        bmi_category = dp.categorize_bmi(pd.Series([bmi]))[0]
        st.write(f"**BMI Category:** {bmi_category}")
        
        # Display blood pressure category
        bp_category = dp.categorize_blood_pressure(pd.Series([ap_hi]), pd.Series([ap_lo]))[0]
        st.write(f"**Blood Pressure Category:** {bp_category}")
        
        # Model explanation (if SHAP available)
        st.write("**Individual Feature Contribution:**")
        st.info("This section would display the contribution of each feature to the prediction using SHAP values if available.")
        
        # General cardiovascular disease information
        with st.expander("Cardiovascular Disease Information"):
            st.markdown("""
            **Cardiovascular disease (CVD)** includes conditions affecting the heart and blood vessels. Common CVDs include:
            
            - Coronary heart disease
            - Cerebrovascular disease
            - Peripheral arterial disease
            - Rheumatic heart disease
            - Congenital heart disease
            - Deep vein thrombosis and pulmonary embolism
            
            **Common risk factors include:**
            - High blood pressure
            - High cholesterol
            - Smoking
            - Obesity
            - Physical inactivity
            - Diabetes
            - Family history
            - Age
            - Stress
            
            **Prevention strategies:**
            - Regular physical activity
            - Healthy diet
            - Avoiding tobacco use
            - Limiting alcohol consumption
            - Regular health check-ups
            - Medication as prescribed
            """)
