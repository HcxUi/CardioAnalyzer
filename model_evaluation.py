import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import cross_val_score

def plot_confusion_matrix(confusion_matrix, model_name):
    """Plot confusion matrix."""
    fig = px.imshow(confusion_matrix,
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['No Disease', 'Disease'],
                   y=['No Disease', 'Disease'],
                   title=f'Confusion Matrix - {model_name}',
                   color_continuous_scale='Blues')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=j, y=i,
                text=str(confusion_matrix[i, j]),
                showarrow=False,
                font=dict(color='white' if confusion_matrix[i, j] > confusion_matrix.max()/2 else 'black')
            )
    
    return fig

def plot_roc_curve(fpr, tpr, roc_auc, model_name):
    """Plot ROC curve."""
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f'ROC curve (AUC = {roc_auc:.2f})',
        line=dict(color='blue')
    ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    
    return fig

def plot_model_comparison(metrics_df):
    """Plot model comparison."""
    fig = px.bar(metrics_df,
                x='Model',
                y=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                title='Model Performance Comparison',
                barmode='group')
    
    fig.update_layout(
        xaxis_title='Model',
        yaxis_title='Score',
        yaxis_range=[0, 1]
    )
    
    return fig

def plot_cv_results(cv_df):
    """Plot cross-validation results."""
    fig = px.bar(cv_df,
                x='Model',
                y='Mean CV Accuracy',
                error_y='Std CV Accuracy',
                title='Cross-Validation Results',
                labels={'Mean CV Accuracy': 'Mean Accuracy', 'Std CV Accuracy': 'Standard Deviation'})
    
    fig.update_layout(
        xaxis_title='Model',
        yaxis_title='Accuracy',
        yaxis_range=[0, 1]
    )
    
    return fig

def perform_cross_validation(models, X, y, cv=5):
    """Perform cross-validation for all models."""
    cv_results = {}
    
    for model_name, model_data in models.items():
        model = model_data['model']
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv)
        
        cv_results[model_name] = {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std()
        }
    
    return cv_results
