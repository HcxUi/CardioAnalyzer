import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def convert_age_to_years(age_days):
    """
    Convert age from days to years
    
    Args:
        age_days: Age in days
        
    Returns:
        Age in years
    """
    return age_days / 365.25

def calculate_bmi(weight, height):
    """
    Calculate Body Mass Index (BMI) from weight and height
    
    Args:
        weight: Weight in kg
        height: Height in cm
        
    Returns:
        BMI value
    """
    # Convert height from cm to m
    height_m = height / 100
    
    # Calculate BMI
    bmi = weight / (height_m ** 2)
    
    return bmi

def categorize_bmi(bmi):
    """
    Categorize BMI values according to WHO standards
    
    Args:
        bmi: BMI values
        
    Returns:
        BMI categories
    """
    conditions = [
        (bmi < 18.5),
        (bmi >= 18.5) & (bmi < 25),
        (bmi >= 25) & (bmi < 30),
        (bmi >= 30)
    ]
    choices = ['Underweight', 'Normal', 'Overweight', 'Obese']
    return pd.Series(np.select(conditions, choices, default='Unknown'))

def categorize_blood_pressure(systolic, diastolic):
    """
    Categorize blood pressure according to American Heart Association standards
    
    Args:
        systolic: Systolic blood pressure values
        diastolic: Diastolic blood pressure values
        
    Returns:
        Blood pressure categories
    """
    conditions = [
        (systolic < 120) & (diastolic < 80),
        (systolic >= 120) & (systolic < 140) & (diastolic < 90),
        (systolic >= 140) | (diastolic >= 90)
    ]
    choices = ['Normal', 'Elevated', 'High']
    return pd.Series(np.select(conditions, choices, default='Unknown'))

def handle_missing_values(data):
    """
    Handle missing values in the dataset
    
    Args:
        data: DataFrame containing the dataset
        
    Returns:
        DataFrame with handled missing values
    """
    # For numerical columns, fill with median
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        data[col] = data[col].fillna(data[col].median())
    
    # For categorical columns, fill with mode
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    return data

def handle_outliers(data, columns):
    """
    Detect and handle outliers using IQR method
    
    Args:
        data: DataFrame containing the dataset
        columns: List of columns to check for outliers
        
    Returns:
        DataFrame with handled outliers and count of outliers removed
    """
    data_cleaned = data.copy()
    outliers_removed = 0
    
    for col in columns:
        # Calculate Q1, Q3, and IQR
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = ((data[col] < lower_bound) | (data[col] > upper_bound))
        outlier_count = outliers.sum()
        outliers_removed += outlier_count
        
        # Remove outliers
        data_cleaned.loc[outliers, col] = data[col].median()
    
    return data_cleaned, outliers_removed

def scale_features(data):
    """
    Scale numerical features using StandardScaler
    
    Args:
        data: DataFrame containing the features to scale
        
    Returns:
        Scaled features and the scaler object
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler
