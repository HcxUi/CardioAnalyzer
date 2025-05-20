# CardioAnalyzer

A machine learning-powered web application for cardiovascular disease prediction and analysis.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

## Features

- Cardiovascular disease risk prediction using machine learning
- Interactive data visualization and analysis
- Feature correlation analysis
- Model comparison and evaluation
- Real-time prediction with detailed risk factor analysis
- Historical prediction tracking

## Demo

Check out the live demo of the application [here](https://share.streamlit.io/).

## Tech Stack

- Python 3.9+
- Streamlit for web interface
- Pandas & NumPy for data processing
- Scikit-learn for machine learning
- Plotly & Seaborn for visualization
- SQLite for prediction history

## Installation

1. Clone the repository:
```bash
git clone https://github.com/HcxUi/CardioAnalyzer.git
cd CardioAnalyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Launch the application using `streamlit run app.py`
2. Navigate through different sections using the sidebar
3. Upload your dataset or use the sample dataset
4. Explore data visualizations and analysis
5. Train and compare different models
6. Make predictions and view detailed risk analysis

## Project Structure

- `app.py`: Main Streamlit application
- `data_preprocessing.py`: Data cleaning and preprocessing functions
- `data_visualization.py`: Visualization components
- `model_training.py`: ML model training and evaluation
- `model_evaluation.py`: Model performance metrics
- `database.py`: Database operations for prediction history
- `utils.py`: Utility functions
- `attached_assets/`: Contains sample dataset

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
