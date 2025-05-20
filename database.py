import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

# Initialize SQLAlchemy
Base = declarative_base()

# Define PredictionResult model
class PredictionResult(Base):
    __tablename__ = 'prediction_results'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    age = Column(Float)
    gender = Column(Integer)
    height = Column(Float)
    weight = Column(Float)
    ap_hi = Column(Integer)
    ap_lo = Column(Integer)
    cholesterol = Column(Integer)
    gluc = Column(Integer)
    smoke = Column(Boolean)
    alco = Column(Boolean)
    active = Column(Boolean)
    bmi = Column(Float)
    predicted_cardio = Column(Boolean)
    prediction_probability = Column(Float)
    model_name = Column(String)

# Initialize database connection
try:
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///predictions.db')
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
except Exception as e:
    print(f"Database initialization error: {e}")
    Session = None

def save_prediction(features, prediction, probability, model_name):
    """Save prediction results to database."""
    if Session is None:
        return None
    
    try:
        session = Session()
        result = PredictionResult(
            age=features['age_years'],
            gender=features['gender'],
            height=features['height'],
            weight=features['weight'],
            ap_hi=features['ap_hi'],
            ap_lo=features['ap_lo'],
            cholesterol=features['cholesterol'],
            gluc=features['gluc'],
            smoke=bool(features['smoke']),
            alco=bool(features['alco']),
            active=bool(features['active']),
            bmi=features['bmi'],
            predicted_cardio=bool(prediction),
            prediction_probability=float(probability),
            model_name=model_name
        )
        session.add(result)
        session.commit()
        db_id = result.id
        session.close()
        return db_id
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return None

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
            PredictionResult.model_name
        ).order_by(PredictionResult.timestamp.desc()).limit(limit).all()
        
        # Convert to DataFrame
        if not results:
            df = pd.DataFrame()
        else:
            # Use a more efficient method to create DataFrame
            df = pd.DataFrame(results, columns=[
                'id', 'patient_id', 'timestamp', 'age', 'gender', 'bmi',
                'predicted_cardio', 'prediction_probability', 'model_name'
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
            PredictionResult.model_name,
            func.count(PredictionResult.id)
        ).group_by(PredictionResult.model_name).all()
        
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