import numpy as np
from sklearn.ensemble import RandomForestClassifier

def get_feature_importance(X, y):
    """Get feature importance using Random Forest."""
    # Train a Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    importance = rf.feature_importances_
    
    return importance
