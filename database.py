import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

# Get database URL from environment variable
DATABASE_URL = os.environ.get('DATABASE_URL')

# Check if DATABASE_URL is available
if not DATABASE_URL:
    print("WARNING: DATABASE_URL is not set. Database functionality will be disabled.")
    engine = None
    Session = None
else:
    try:
        # Create SQLAlchemy engine with SSL settings to handle connection issues
        connect_args = {"sslmode": "require"}
        engine = create_engine(DATABASE_URL, connect_args=connect_args)
        
        # Create session factory
        Session = sessionmaker(bind=engine)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        engine = None
        Session = None

# Create declarative base
Base = declarative_base()

# Define Prediction table
class PredictionResult(Base):
    __tablename__ = 'prediction_results'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(String(50))
    timestamp = Column(DateTime, default=datetime.now)
    age = Column(Float)
    gender = Column(Integer)
    height = Column(Float)
    weight = Column(Float)
    bmi = Column(Float)
    ap_hi = Column(Integer)
    ap_lo = Column(Integer)
    cholesterol = Column(Integer)
    gluc = Column(Integer)
    smoke = Column(Boolean)
    alco = Column(Boolean)
    active = Column(Boolean)
    predicted_cardio = Column(Boolean)
    prediction_probability = Column(Float)
    model_used = Column(String(100))
    feature_values = Column(Text)  # JSON string of all features

    def __repr__(self):
        return f"<PredictionResult(id={self.id}, patient_id={self.patient_id}, predicted_cardio={self.predicted_cardio})>"

# Create tables in the database if engine is available
if engine:
    Base.metadata.create_all(engine)

def save_prediction(patient_data, prediction, probability, model_name):
    """
    Save a prediction result to the database
    
    Args:
        patient_data: Dictionary or DataFrame row with patient information
        prediction: Predicted value (0 or 1)
        probability: Prediction probability
        model_name: Name of the model used
    
    Returns:
        Saved prediction record ID or None if database is not available
    """
    # Check if database is configured
    if Session is None:
        print("Warning: Database not configured. Prediction not saved.")
        return None
    
    session = Session()
    
    try:
        # Convert to dictionary if it's a DataFrame row
        if isinstance(patient_data, pd.Series):
            patient_data = patient_data.to_dict()
        
        # Create feature values JSON
        feature_values = json.dumps(patient_data)
        
        # Create prediction result record
        prediction_result = PredictionResult(
            patient_id=str(patient_data.get('id', 'unknown')),
            age=patient_data.get('age_years', patient_data.get('age', 0) / 365.25),
            gender=patient_data.get('gender', 0),
            height=patient_data.get('height', 0),
            weight=patient_data.get('weight', 0),
            bmi=patient_data.get('bmi', 0),
            ap_hi=patient_data.get('ap_hi', 0),
            ap_lo=patient_data.get('ap_lo', 0),
            cholesterol=patient_data.get('cholesterol', 0),
            gluc=patient_data.get('gluc', 0),
            smoke=bool(patient_data.get('smoke', 0)),
            alco=bool(patient_data.get('alco', 0)),
            active=bool(patient_data.get('active', 0)),
            predicted_cardio=bool(prediction),
            prediction_probability=float(probability),
            model_used=model_name,
            feature_values=feature_values
        )
        
        # Add and commit
        session.add(prediction_result)
        session.commit()
        
        return prediction_result.id
    
    except Exception as e:
        if session:
            session.rollback()
        print(f"Database error: {e}")
        return None
    
    finally:
        if session:
            session.close()

@pd.api.extensions.register_dataframe_accessor("cache_key")
class CacheKeyAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        
    def __call__(self):
        return f"df_{hash(str(self._obj.head(5)) + str(self._obj.shape))}"

def get_prediction_history(limit=100):
    """
    Get prediction history from the database with caching
    
    Args:
        limit: Maximum number of records to return
        
    Returns:
        DataFrame with prediction history or empty DataFrame if database is not available
    """
    # Check if database is configured
    if Session is None:
        print("Warning: Database not configured. Cannot retrieve prediction history.")
        return pd.DataFrame()
    
    # Try to get from cache first
    cache_key = f"prediction_history_{limit}"
    if cache_key in st.session_state:
        # Check if cache is still valid (less than 30 seconds old)
        cache_time = st.session_state.get(f"{cache_key}_time", None)
        if cache_time and (datetime.now() - cache_time).total_seconds() < 30:
            return st.session_state[cache_key]
    
    session = Session()
    
    try:
        # Query prediction results with optimized SQL
        results = session.query(
            PredictionResult.id,
            PredictionResult.patient_id,
            PredictionResult.timestamp,
            PredictionResult.age,
            PredictionResult.gender,
            PredictionResult.bmi,
            PredictionResult.predicted_cardio,
            PredictionResult.prediction_probability,
            PredictionResult.model_used
        ).order_by(PredictionResult.timestamp.desc()).limit(limit).all()
        
        # Convert to DataFrame
        if not results:
            df = pd.DataFrame()
        else:
            # Use a more efficient method to create DataFrame
            df = pd.DataFrame(results, columns=[
                'id', 'patient_id', 'timestamp', 'age', 'gender', 'bmi',
                'predicted_cardio', 'prediction_probability', 'model_used'
            ])
        
        # Cache the result
        st.session_state[cache_key] = df
        st.session_state[f"{cache_key}_time"] = datetime.now()
        
        return df
    
    except Exception as e:
        print(f"Database error: {e}")
        return pd.DataFrame()
    
    finally:
        if session:
            session.close()

def get_prediction_stats():
    """
    Get prediction statistics from the database with caching
    
    Returns:
        Dictionary with prediction statistics or default values if database is not available
    """
    # Default stats in case of error
    default_stats = {
        'total': 0,
        'positive': 0,
        'negative': 0,
        'positive_percentage': 0,
        'negative_percentage': 0,
        'models': {}
    }
    
    # Check if database is configured
    if Session is None:
        print("Warning: Database not configured. Cannot retrieve prediction statistics.")
        return default_stats
    
    # Try to get from cache first
    cache_key = "prediction_stats"
    if cache_key in st.session_state:
        # Check if cache is still valid (less than 30 seconds old)
        cache_time = st.session_state.get(f"{cache_key}_time", None)
        if cache_time and (datetime.now() - cache_time).total_seconds() < 30:
            return st.session_state[cache_key]
    
    session = Session()
    
    try:
        # Use SQL aggregation for better performance
        from sqlalchemy import func
        
        # Get counts in a single query - using newer case syntax
        counts = session.query(
            func.count(PredictionResult.id).label('total'),
            func.count(PredictionResult.id).filter(PredictionResult.predicted_cardio == True).label('positive'),
            func.count(PredictionResult.id).filter(PredictionResult.predicted_cardio == False).label('negative')
        ).first()
        
        # Handle None values
        total_count = counts[0] if counts and counts[0] is not None else 0
        positive_count = counts[1] if counts and counts[1] is not None else 0
        negative_count = counts[2] if counts and counts[2] is not None else 0
        
        # Get counts by model in a single query
        model_counts = {}
        model_stats = session.query(
            PredictionResult.model_used,
            func.count(PredictionResult.id)
        ).group_by(PredictionResult.model_used).all()
        
        for model_name, count in model_stats:
            model_counts[model_name] = count
        
        # Create stats dictionary
        stats = {
            'total': total_count,
            'positive': positive_count,
            'negative': negative_count,
            'positive_percentage': (positive_count / total_count * 100) if total_count > 0 else 0,
            'negative_percentage': (negative_count / total_count * 100) if total_count > 0 else 0,
            'models': model_counts
        }
        
        # Cache the result
        st.session_state[cache_key] = stats
        st.session_state[f"{cache_key}_time"] = datetime.now()
        
        return stats
    
    except Exception as e:
        print(f"Database error: {e}")
        return default_stats
    
    finally:
        if session:
            session.close()