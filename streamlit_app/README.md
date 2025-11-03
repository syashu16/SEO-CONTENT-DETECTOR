# SEO Content Quality & Duplicate Detector - Streamlit App

This is the bonus Streamlit web application for the SEO Content Quality & Duplicate Detector assignment.

## Features

🎯 **URL Analysis**: Analyze any webpage for SEO content quality
📊 **Dataset Overview**: Interactive visualizations of the processed dataset
🔍 **Duplicate Detection**: View detected duplicate content pairs

## How to Run Locally

1. Install dependencies:

```bash
pip install streamlit plotly pandas scikit-learn beautifulsoup4 sentence-transformers textstat requests
```

2. Run the app:

```bash
streamlit run app.py
```

3. Open your browser to `http://localhost:8501`

## Usage

- **URL Analysis Tab**: Enter any URL to get real-time SEO quality analysis
- **Dataset Overview Tab**: Explore interactive charts of quality scores and features
- **Duplicate Detection Tab**: Browse detected duplicate content pairs with similarity scores

## Deployment

This app is designed to be deployed on Streamlit Cloud. The deployed URL will be added here once deployed.

**Deployed URL**: [To be added after deployment for +15 bonus points]

## Technical Details

- Built with Streamlit for the web interface
- Uses trained Random Forest model for quality prediction
- Incorporates BeautifulSoup for HTML parsing
- Features interactive Plotly visualizations
- Implements caching for optimal performance
