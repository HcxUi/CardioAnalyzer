import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_age_distribution(data):
    """
    Plot age distribution with respect to cardiovascular disease
    
    Args:
        data: DataFrame containing the dataset
        
    Returns:
        Plotly figure object
    """
    # Convert age to years if not already
    if 'age_years' not in data.columns:
        data['age_years'] = data['age'] / 365.25
    
    # Create histogram for age distribution by disease status
    fig = px.histogram(
        data, 
        x='age_years',
        color='cardio',
        barmode='overlay',
        histnorm='percent',
        color_discrete_map={0: 'blue', 1: 'red'},
        labels={'age_years': 'Age (years)', 'cardio': 'Cardiovascular Disease'},
        title='Age Distribution by Disease Status'
    )
    
    # Update figure layout
    fig.update_layout(
        xaxis_title='Age (years)',
        yaxis_title='Percentage (%)',
        legend_title='Cardiovascular Disease',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update color of legend items
    fig.update_traces(
        selector=dict(name='0'),
        name='No Disease'
    )
    fig.update_traces(
        selector=dict(name='1'),
        name='Disease'
    )
    
    return fig

def plot_gender_distribution(data):
    """
    Plot gender distribution with respect to cardiovascular disease
    
    Args:
        data: DataFrame containing the dataset
        
    Returns:
        Plotly figure object
    """
    # Calculate counts by gender and disease status
    gender_counts = data.groupby(['gender', 'cardio']).size().reset_index(name='count')
    
    # Map numeric gender to labels
    gender_mapping = {1: 'Female', 2: 'Male'}
    gender_counts['gender_label'] = gender_counts['gender'].map(gender_mapping)
    
    # Map numeric disease status to labels
    cardio_mapping = {0: 'No Disease', 1: 'Disease'}
    gender_counts['cardio_label'] = gender_counts['cardio'].map(cardio_mapping)
    
    # Create grouped bar chart
    fig = px.bar(
        gender_counts, 
        x='gender_label', 
        y='count', 
        color='cardio_label',
        barmode='group',
        title='Gender Distribution by Disease Status',
        labels={'gender_label': 'Gender', 'count': 'Count', 'cardio_label': 'Cardiovascular Disease'}
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Gender',
        yaxis_title='Count',
        legend_title='Cardiovascular Disease'
    )
    
    # Update legend items
    fig.update_traces(
        selector=dict(name='0'),
        name='No Disease'
    )
    fig.update_traces(
        selector=dict(name='1'),
        name='Disease'
    )
    
    return fig

def plot_age_gender_disease(data):
    """
    Plot age vs gender with respect to cardiovascular disease
    
    Args:
        data: DataFrame containing the dataset
        
    Returns:
        Plotly figure object
    """
    # Create a copy of the data with age in years
    if 'age_years' not in data.columns:
        data['age_years'] = data['age'] / 365.25
    
    # Map numeric gender to labels
    gender_mapping = {1: 'Female', 2: 'Male'}
    data_copy = data.copy()
    data_copy['gender_label'] = data_copy['gender'].map(gender_mapping)
    
    # Create box plot
    fig = px.box(
        data_copy,
        x='gender_label',
        y='age_years',
        color='cardio',
        notched=True,
        points='outliers',
        title='Age Distribution by Gender and Disease Status',
        labels={'gender_label': 'Gender', 'age_years': 'Age (years)', 'cardio': 'Cardiovascular Disease'}
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Gender',
        yaxis_title='Age (years)',
        legend_title='Cardiovascular Disease'
    )
    
    # Update legend items
    fig.update_traces(
        selector=dict(name='0'),
        name='No Disease'
    )
    fig.update_traces(
        selector=dict(name='1'),
        name='Disease'
    )
    
    return fig

def plot_blood_pressure_distribution(data):
    """
    Plot blood pressure distribution with respect to cardiovascular disease
    
    Args:
        data: DataFrame containing the dataset
        
    Returns:
        Plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Systolic Blood Pressure', 'Diastolic Blood Pressure'),
        shared_yaxes=True
    )
    
    # Add histograms for systolic blood pressure
    for cardio in [0, 1]:
        fig.add_trace(
            go.Histogram(
                x=data[data['cardio'] == cardio]['ap_hi'],
                opacity=0.7,
                name=f'Cardio: {cardio}',
                marker_color='blue' if cardio == 0 else 'red'
            ),
            row=1, col=1
        )
    
    # Add histograms for diastolic blood pressure
    for cardio in [0, 1]:
        fig.add_trace(
            go.Histogram(
                x=data[data['cardio'] == cardio]['ap_lo'],
                opacity=0.7,
                name=f'Cardio: {cardio}',
                marker_color='blue' if cardio == 0 else 'red',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text='Blood Pressure Distribution by Disease Status',
        barmode='overlay',
        height=500,
        width=900,
        legend=dict(
            title='Cardiovascular Disease',
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x and y axis labels
    fig.update_xaxes(title_text='Systolic BP (mmHg)', row=1, col=1)
    fig.update_xaxes(title_text='Diastolic BP (mmHg)', row=1, col=2)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    
    # Update legend items
    fig.update_traces(
        selector=dict(name='0'),
        name='No Disease'
    )
    fig.update_traces(
        selector=dict(name='1'),
        name='Disease'
    )
    
    return fig

def plot_cholesterol_glucose(data):
    """
    Plot cholesterol and glucose levels with respect to cardiovascular disease
    
    Args:
        data: DataFrame containing the dataset
        
    Returns:
        Plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Cholesterol Levels', 'Glucose Levels'),
        shared_yaxes=True
    )
    
    # Map cholesterol levels to labels
    chol_mapping = {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'}
    
    # Calculate counts for cholesterol
    chol_counts = data.groupby(['cholesterol', 'cardio']).size().reset_index(name='count')
    chol_counts['chol_label'] = chol_counts['cholesterol'].map(chol_mapping)
    
    # Add bar charts for cholesterol
    for cardio in [0, 1]:
        temp_df = chol_counts[chol_counts['cardio'] == cardio]
        fig.add_trace(
            go.Bar(
                x=temp_df['chol_label'],
                y=temp_df['count'],
                name=f'Cardio: {cardio}',
                marker_color='blue' if cardio == 0 else 'red'
            ),
            row=1, col=1
        )
    
    # Map glucose levels to labels
    gluc_mapping = {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'}
    
    # Calculate counts for glucose
    gluc_counts = data.groupby(['gluc', 'cardio']).size().reset_index(name='count')
    gluc_counts['gluc_label'] = gluc_counts['gluc'].map(gluc_mapping)
    
    # Add bar charts for glucose
    for cardio in [0, 1]:
        temp_df = gluc_counts[gluc_counts['cardio'] == cardio]
        fig.add_trace(
            go.Bar(
                x=temp_df['gluc_label'],
                y=temp_df['count'],
                name=f'Cardio: {cardio}',
                marker_color='blue' if cardio == 0 else 'red',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text='Cholesterol and Glucose Levels by Disease Status',
        barmode='group',
        height=500,
        width=900,
        legend=dict(
            title='Cardiovascular Disease',
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x and y axis labels
    fig.update_xaxes(title_text='Cholesterol Level', row=1, col=1)
    fig.update_xaxes(title_text='Glucose Level', row=1, col=2)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    
    # Update legend items
    fig.update_traces(
        selector=dict(name='0'),
        name='No Disease'
    )
    fig.update_traces(
        selector=dict(name='1'),
        name='Disease'
    )
    
    return fig

def plot_bmi_distribution(data):
    """
    Plot BMI distribution with respect to cardiovascular disease
    
    Args:
        data: DataFrame containing the dataset
        
    Returns:
        Plotly figure object
    """
    # Ensure BMI is calculated
    if 'bmi' not in data.columns:
        data['bmi'] = data['weight'] / ((data['height']/100) ** 2)
    
    # Create histogram
    fig = px.histogram(
        data,
        x='bmi',
        color='cardio',
        barmode='overlay',
        nbins=50,
        opacity=0.7,
        marginal='box',
        title='BMI Distribution by Disease Status',
        labels={'bmi': 'BMI', 'cardio': 'Cardiovascular Disease'}
    )
    
    # Add vertical lines for BMI categories
    fig.add_vline(x=18.5, line_dash="dash", line_color="green", annotation_text="Underweight")
    fig.add_vline(x=25, line_dash="dash", line_color="orange", annotation_text="Overweight")
    fig.add_vline(x=30, line_dash="dash", line_color="red", annotation_text="Obese")
    
    # Update layout
    fig.update_layout(
        xaxis_title='BMI',
        yaxis_title='Count',
        legend_title='Cardiovascular Disease',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update color of legend items
    fig.update_traces(
        selector=dict(name='0'),
        name='No Disease'
    )
    fig.update_traces(
        selector=dict(name='1'),
        name='Disease'
    )
    
    return fig

def plot_lifestyle_factors(data):
    """
    Plot lifestyle factors (smoking, alcohol, activity) with respect to cardiovascular disease
    
    Args:
        data: DataFrame containing the dataset
        
    Returns:
        Plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Smoking Status', 'Alcohol Consumption', 'Physical Activity'),
        shared_yaxes=True
    )
    
    # Lifestyle factors
    lifestyle_factors = ['smoke', 'alco', 'active']
    factor_labels = ['Smoker', 'Alcohol Consumer', 'Physically Active']
    
    # Add bar charts for each lifestyle factor
    for i, (factor, label) in enumerate(zip(lifestyle_factors, factor_labels)):
        # Calculate counts
        counts = data.groupby([factor, 'cardio']).size().reset_index(name='count')
        
        # Map values to labels
        value_mapping = {0: 'No', 1: 'Yes'}
        counts['value_label'] = counts[factor].map(value_mapping)
        
        # Add bar charts
        for cardio in [0, 1]:
            temp_df = counts[counts['cardio'] == cardio]
            show_legend = (i == 0)  # Only show legend for the first factor
            
            fig.add_trace(
                go.Bar(
                    x=temp_df['value_label'],
                    y=temp_df['count'],
                    name=f'Cardio: {cardio}',
                    marker_color='blue' if cardio == 0 else 'red',
                    showlegend=show_legend
                ),
                row=1, col=i+1
            )
    
    # Update layout
    fig.update_layout(
        title_text='Lifestyle Factors by Disease Status',
        barmode='group',
        height=500,
        width=900,
        legend=dict(
            title='Cardiovascular Disease',
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x and y axis labels
    fig.update_xaxes(title_text='Smoking Status', row=1, col=1)
    fig.update_xaxes(title_text='Alcohol Consumption', row=1, col=2)
    fig.update_xaxes(title_text='Physical Activity', row=1, col=3)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    
    # Update legend items
    fig.update_traces(
        selector=dict(name='0'),
        name='No Disease'
    )
    fig.update_traces(
        selector=dict(name='1'),
        name='Disease'
    )
    
    return fig

def plot_lifestyle_combination(data):
    """
    Plot combination of lifestyle factors with respect to cardiovascular disease
    
    Args:
        data: DataFrame containing the dataset
        
    Returns:
        Plotly figure object
    """
    # Create a lifestyle score: +1 for each healthy choice
    data['lifestyle_score'] = (1 - data['smoke']) + (1 - data['alco']) + data['active']
    
    # Calculate percentages of disease by lifestyle score
    lifestyle_cardio = data.groupby('lifestyle_score')['cardio'].mean().reset_index()
    lifestyle_cardio['cardio_percentage'] = lifestyle_cardio['cardio'] * 100
    
    # Count number of people in each lifestyle score group
    lifestyle_counts = data.groupby('lifestyle_score').size().reset_index(name='count')
    
    # Merge the two dataframes
    lifestyle_data = pd.merge(lifestyle_cardio, lifestyle_counts, on='lifestyle_score')
    
    # Map scores to descriptive labels
    score_labels = {
        0: 'Unhealthy\n(Smoker, Drinks, Inactive)',
        1: '1 Healthy Habit',
        2: '2 Healthy Habits',
        3: 'Healthy\n(Non-smoker, No Drinks, Active)'
    }
    lifestyle_data['score_label'] = lifestyle_data['lifestyle_score'].map(score_labels)
    
    # Create a dual-axis figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart for counts
    fig.add_trace(
        go.Bar(
            x=lifestyle_data['score_label'],
            y=lifestyle_data['count'],
            name='Population Count',
            marker_color='lightblue',
            opacity=0.7
        ),
        secondary_y=False
    )
    
    # Add line chart for disease percentage
    fig.add_trace(
        go.Scatter(
            x=lifestyle_data['score_label'],
            y=lifestyle_data['cardio_percentage'],
            name='Disease Percentage',
            marker_color='red',
            mode='lines+markers'
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title_text='Cardiovascular Disease by Lifestyle Habits',
        xaxis_title='Lifestyle Habits',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text='Population Count', secondary_y=False)
    fig.update_yaxes(title_text='Disease Percentage (%)', secondary_y=True)
    
    # Update legend items
    fig.update_traces(
        selector=dict(name='0'),
        name='No Disease'
    )
    fig.update_traces(
        selector=dict(name='1'),
        name='Disease'
    )
    
    return fig

def plot_scatter_matrix(data):
    """
    Plot scatter matrix of key numerical variables
    
    Args:
        data: DataFrame containing the dataset
        
    Returns:
        Plotly figure object
    """
    # Select key numerical variables
    if 'age_years' not in data.columns:
        data['age_years'] = data['age'] / 365.25
    
    if 'bmi' not in data.columns:
        data['bmi'] = data['weight'] / ((data['height']/100) ** 2)
    
    key_vars = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
    
    # Sample data to keep the plot manageable
    sample_data = data.sample(min(5000, len(data)), random_state=42)
    
    # Create scatter matrix
    fig = px.scatter_matrix(
        sample_data, 
        dimensions=key_vars,
        color='cardio',
        symbol='cardio',
        title='Scatter Matrix of Key Variables',
        labels={
            'age_years': 'Age (years)',
            'height': 'Height (cm)',
            'weight': 'Weight (kg)',
            'ap_hi': 'Systolic BP',
            'ap_lo': 'Diastolic BP',
            'bmi': 'BMI',
            'cardio': 'Disease'
        }
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=900
    )
    
    # Update color of legend items
    fig.update_traces(
        selector=dict(name='0'),
        name='No Disease'
    )
    fig.update_traces(
        selector=dict(name='1'),
        name='Disease'
    )
    
    return fig

def plot_3d_visualization(data):
    """
    Create a 3D visualization of key variables
    
    Args:
        data: DataFrame containing the dataset
        
    Returns:
        Plotly figure object
    """
    # Ensure necessary features are available
    if 'age_years' not in data.columns:
        data['age_years'] = data['age'] / 365.25
    
    if 'bmi' not in data.columns:
        data['bmi'] = data['weight'] / ((data['height']/100) ** 2)
    
    # Sample data to keep the plot manageable
    sample_data = data.sample(min(5000, len(data)), random_state=42)
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        sample_data,
        x='bmi',
        y='age_years',
        z='ap_hi',
        color='cardio',
        size='ap_lo',
        size_max=10,
        opacity=0.7,
        title='3D Visualization of Key Risk Factors',
        labels={
            'bmi': 'BMI',
            'age_years': 'Age (years)',
            'ap_hi': 'Systolic BP',
            'ap_lo': 'Diastolic BP',
            'cardio': 'Disease'
        }
    )
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='BMI',
            yaxis_title='Age (years)',
            zaxis_title='Systolic BP'
        ),
        height=700,
        width=900
    )
    
    # Update color of legend items
    fig.update_traces(
        selector=dict(name='0'),
        name='No Disease'
    )
    fig.update_traces(
        selector=dict(name='1'),
        name='Disease'
    )
    
    return fig

def plot_top_features_pair(data, features):
    """
    Create pair plots for the top features
    
    Args:
        data: DataFrame containing the dataset
        features: List of top features
        
    Returns:
        Plotly figure object
    """
    # Sample data to keep the plot manageable
    sample_data = data.sample(min(5000, len(data)), random_state=42)
    
    # Create pair plot
    fig = px.scatter_matrix(
        sample_data,
        dimensions=features[:-1],  # Exclude the target variable
        color='cardio',
        title='Relationships Between Top Features',
        labels={
            'age_years': 'Age (years)',
            'height': 'Height (cm)',
            'weight': 'Weight (kg)',
            'ap_hi': 'Systolic BP',
            'ap_lo': 'Diastolic BP',
            'bmi': 'BMI',
            'cholesterol': 'Cholesterol',
            'gluc': 'Glucose',
            'smoke': 'Smoking',
            'alco': 'Alcohol',
            'active': 'Physical Activity',
            'cardio': 'Disease'
        }
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        width=900
    )
    
    # Update color of legend items
    fig.update_traces(
        selector=dict(name='0'),
        name='No Disease'
    )
    fig.update_traces(
        selector=dict(name='1'),
        name='Disease'
    )
    
    return fig
