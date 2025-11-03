import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import requests
import re
import textstat
import time
import random
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="SEO Content Quality & Duplicate Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #3b82f6;
        margin: 1rem 0;
    }
    .quality-high {
        color: #059669;
        font-weight: bold;
    }
    .quality-medium {
        color: #d97706;
        font-weight: bold;
    }
    .quality-low {
        color: #dc2626;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    """Load processed data"""
    try:
        # Fixed file paths for Streamlit Cloud deployment
        features_df = pd.read_csv('data/features.csv')
        # Convert embedding strings back to lists for the first few rows only (to avoid memory issues)
        if 'embedding' in features_df.columns:
            features_df['embedding'] = features_df['embedding'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else []
            )
        duplicates_df = pd.read_csv('data/duplicates.csv')
        extracted_df = pd.read_csv('data/extracted_content.csv')
        return features_df, duplicates_df, extracted_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

@st.cache_resource
def load_model():
    """Load trained ML model"""
    try:
        # Fixed file path for Streamlit Cloud deployment
        with open('models/quality_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Model file not found. The app will run in demo mode without quality predictions.")
        return None

@st.cache_resource
def load_sentence_transformer():
    """Load sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_random_user_agent():
    """Get a random user agent string to avoid bot detection"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0'
    ]
    return random.choice(user_agents)

def fetch_url_with_retry(url, max_retries=3):
    """Fetch URL with multiple retry strategies to avoid 403 errors"""
    
    for attempt in range(max_retries):
        try:
            # Different header strategies for each attempt
            if attempt == 0:
                # Standard headers
                headers = {
                    'User-Agent': get_random_user_agent(),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
            elif attempt == 1:
                # More browser-like headers
                headers = {
                    'User-Agent': get_random_user_agent(),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-User': '?1',
                    'Cache-Control': 'max-age=0',
                    'Upgrade-Insecure-Requests': '1',
                }
            else:
                # Minimal headers as last resort
                headers = {
                    'User-Agent': get_random_user_agent(),
                    'Accept': '*/*',
                }
            
            # Add random delay between attempts
            if attempt > 0:
                time.sleep(random.uniform(1, 3))
            
            # Create session for better connection handling
            session = requests.Session()
            session.headers.update(headers)
            
            # Make request with timeout
            response = session.get(url, timeout=15, allow_redirects=True)
            
            # Check if request was successful
            if response.status_code == 200:
                return response
            elif response.status_code == 403:
                if attempt < max_retries - 1:
                    continue  # Try next strategy
                else:
                    raise requests.exceptions.RequestException(f"403 Forbidden: Website blocks automated access")
            else:
                response.raise_for_status()
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                continue
            else:
                raise requests.exceptions.RequestException(f"Request timeout after {max_retries} attempts")
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                continue
            else:
                raise e
    
    # If all attempts failed
    raise requests.exceptions.RequestException(f"Failed to fetch URL after {max_retries} attempts")

def extract_content_from_html(html_content):
    """Extract title, body text, and word count from HTML content."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else "No Title"
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        body_text = ""
        content_tags = soup.find_all(['p', 'article', 'main', 'div', 'section'])
        for tag in content_tags:
            text = tag.get_text()
            if len(text.strip()) > 20:
                body_text += text + " "
        
        if not body_text.strip():
            body = soup.find('body')
            body_text = body.get_text() if body else soup.get_text()
        
        body_text = re.sub(r'\s+', ' ', body_text).strip()
        word_count = len(body_text.split()) if body_text else 0
        
        return {
            'title': title,
            'body_text': body_text,
            'word_count': word_count
        }
    except Exception as e:
        return {
            'title': "Parse Error",
            'body_text': "",
            'word_count': 0
        }

def extract_features(text):
    """Extract comprehensive features from text content."""
    try:
        if not text or len(text.strip()) == 0:
            return {
                'sentence_count': 0,
                'flesch_reading_ease': 0,
                'avg_word_length': 0
            }
        
        clean_text = re.sub(r'\s+', ' ', text.lower().strip())
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        flesch_score = textstat.flesch_reading_ease(text)
        
        words = clean_text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        return {
            'sentence_count': sentence_count,
            'flesch_reading_ease': flesch_score,
            'avg_word_length': avg_word_length
        }
    except Exception as e:
        return {
            'sentence_count': 0,
            'flesch_reading_ease': 0,
            'avg_word_length': 0
        }

def predict_quality(model, content_info, features):
    """Predict content quality using the trained model."""
    try:
        if model is None:
            # Return basic quality assessment if no model is available
            word_count = content_info['word_count']
            readability = features['flesch_reading_ease']
            
            if word_count >= 1000 and readability >= 60:
                return {'label': 'High', 'confidence': {'High': 0.8, 'Medium': 0.15, 'Low': 0.05}}
            elif word_count >= 500 and readability >= 30:
                return {'label': 'Medium', 'confidence': {'High': 0.2, 'Medium': 0.7, 'Low': 0.1}}
            else:
                return {'label': 'Low', 'confidence': {'High': 0.1, 'Medium': 0.2, 'Low': 0.7}}
        
        feature_vector = np.array([[
            content_info['word_count'],
            features['sentence_count'],
            features['flesch_reading_ease'],
            features['avg_word_length']
        ]])
        
        quality_prediction = model.predict(feature_vector)[0]
        quality_proba = model.predict_proba(feature_vector)[0]
        
        class_names = model.classes_
        confidence_dict = {}
        for i, class_name in enumerate(class_names):
            confidence_dict[class_name] = quality_proba[i]
        
        return {
            'label': quality_prediction,
            'confidence': confidence_dict
        }
    except Exception as e:
        return {
            'label': 'Unknown',
            'confidence': {'High': 0, 'Medium': 0, 'Low': 0}
        }

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 SEO Content Quality & Duplicate Detector</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This application analyzes web content for SEO quality assessment and duplicate detection using 
    machine learning. Enter a URL to get instant analysis!
    """)
    
    # Load data and model
    features_df, duplicates_df, extracted_df = load_data()
    model = load_model()
    
    # Check if data files exist
    if features_df is None:
        st.warning("⚠️ Dataset files not found. Running in demo mode.")
        st.info("You can still analyze URLs, but historical data won't be available.")
        # Create dummy data for demo
        features_df = pd.DataFrame({
            'word_count': [100, 500, 1000],
            'quality_label': ['Low', 'Medium', 'High']
        })
        duplicates_df = pd.DataFrame()
        extracted_df = pd.DataFrame()
    
    # Sidebar with dataset statistics
    st.sidebar.markdown("## 📊 Dataset Statistics")
    
    # Show corrected statistics
    valid_pages = len(features_df[features_df['word_count'] > 0]) if len(features_df) > 0 else 0
    failed_pages = len(features_df[features_df['word_count'] == 0]) if len(features_df) > 0 else 0
    
    st.sidebar.metric("Total Pages Processed", len(features_df))
    st.sidebar.metric("Valid Pages Analyzed", valid_pages)
    st.sidebar.metric("Failed/Excluded Pages", failed_pages)
    st.sidebar.metric("Duplicate Pairs Found", len(duplicates_df) if duplicates_df is not None else 0)
    
    # Add information about the correction
    if duplicates_df is not None and len(duplicates_df) > 0 and valid_pages > 0:
        duplication_rate = len(duplicates_df) / valid_pages * 100
        st.sidebar.metric("Duplication Rate", f"{duplication_rate:.1f}% (among valid content)")
    
    if len(features_df) > 0:
        thin_content = (features_df['word_count'] < 500).sum()
        st.sidebar.metric("Thin Content Pages", f"{thin_content} ({thin_content/len(features_df)*100:.1f}%)")
    

    
    # Quality distribution
    if 'quality_label' in features_df.columns:
        quality_counts = features_df['quality_label'].value_counts()
        st.sidebar.markdown("### Quality Distribution")
        for label, count in quality_counts.items():
            percentage = count / len(features_df) * 100
            st.sidebar.write(f"**{label}**: {count} ({percentage:.1f}%)")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 URL Analysis", "📈 Dataset Overview", "🔄 Duplicates"])
    
    with tab1:
        url_analysis_tab(model, features_df)
    
    with tab2:
        dataset_overview_tab(features_df, extracted_df)
    
    with tab3:
        duplicates_tab(duplicates_df, features_df)

def url_analysis_tab(model, features_df):
    """URL analysis functionality"""
    st.markdown('<h2 class="section-header">Real-Time URL Analysis</h2>', unsafe_allow_html=True)
    
    # URL input
    url = st.text_input("Enter URL to analyze:", placeholder="https://example.com/article")
    
    # Add some example URLs that are more likely to work
    st.markdown("**Try these example URLs:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📰 BBC News", help="Analyze BBC article"):
            url = "https://www.bbc.com/news"
    with col2:
        if st.button("📖 Wikipedia", help="Analyze Wikipedia page"):
            url = "https://en.wikipedia.org/wiki/Machine_learning"
    with col3:
        if st.button("🌐 Example.com", help="Test with example site"):
            url = "https://example.com"
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("🔍 Analyze", type="primary")
    
    if analyze_button and url:
        with st.spinner("Analyzing URL..."):
            try:
                # Use improved fetch function
                response = fetch_url_with_retry(url)
                
                # Extract content
                content_info = extract_content_from_html(response.text)
                features = extract_features(content_info['body_text'])
                
                # Predict quality
                quality_result = predict_quality(model, content_info, features)
                
                # Display results
                display_analysis_results(url, content_info, features, quality_result)
                
            except requests.exceptions.RequestException as e:
                error_msg = str(e)
                if "403" in error_msg:
                    st.error("🚫 **Website Access Blocked**")
                    st.warning("""
                    This website blocks automated access (403 Forbidden). This is common for:
                    - Corporate websites (ConnectWise, etc.)
                    - Sites with strong anti-bot protection
                    - Cloudflare-protected sites
                    
                    **Try these alternatives:**
                    - News websites (BBC, Reuters, etc.)
                    - Wikipedia articles
                    - Blog posts or documentation sites
                    - Educational institution websites
                    """)
                elif "timeout" in error_msg.lower():
                    st.error("⏱️ **Request Timeout** - The website took too long to respond.")
                else:
                    st.error(f"❌ **Failed to fetch URL:** {error_msg}")
                    
                st.info("💡 **Tip:** Some websites work better than others. Try different URLs if you encounter issues.")
                
            except Exception as e:
                st.error(f"🔧 **Analysis failed:** {str(e)}")

def display_analysis_results(url, content_info, features, quality_result):
    """Display analysis results"""
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Word Count", content_info['word_count'])
    
    with col2:
        st.metric("Sentence Count", features['sentence_count'])
    
    with col3:
        st.metric("Readability Score", f"{features['flesch_reading_ease']:.1f}")
    
    with col4:
        avg_word_len = features.get('avg_word_length', 0)
        st.metric("Avg Word Length", f"{avg_word_len:.1f}")
    
    # Quality prediction
    st.markdown("### 🎯 Quality Assessment")
    quality_label = quality_result['label']
    confidence = quality_result['confidence'][quality_label] if quality_label in quality_result['confidence'] else 0
    
    quality_class = f"quality-{quality_label.lower()}"
    st.markdown(f'<p class="{quality_class}">Quality: {quality_label} (Confidence: {confidence:.2f})</p>', 
                unsafe_allow_html=True)
    
    # Confidence breakdown
    conf_data = quality_result['confidence']
    if conf_data:
        fig = px.bar(
            x=list(conf_data.keys()),
            y=list(conf_data.values()),
            title="Quality Confidence Scores",
            color=list(conf_data.values()),
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig, width="stretch")
    
    # Content analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 Content Info")
        st.write(f"**Title:** {content_info['title']}")
        st.write(f"**Word Count:** {content_info['word_count']}")
        is_thin = content_info['word_count'] < 500
        st.write(f"**Thin Content:** {'Yes' if is_thin else 'No'}")
    
    with col2:
        st.markdown("### 📊 Text Features")
        st.write(f"**Sentences:** {features['sentence_count']}")
        st.write(f"**Readability:** {features['flesch_reading_ease']:.1f}")
        st.write(f"**Avg Word Length:** {features.get('avg_word_length', 0):.1f}")

def dataset_overview_tab(features_df, extracted_df):
    """Dataset overview tab"""
    st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    if extracted_df is not None and len(extracted_df) > 0:
        # Sample content
        st.markdown("### 📄 Sample Content")
        sample_data = extracted_df[['url', 'title', 'word_count']].head(10)
        st.dataframe(sample_data, width="stretch")
    else:
        st.info("No extracted content data available.")
    
    if len(features_df) > 0:
        # Word count distribution
        st.markdown("### 📊 Word Count Distribution")
        fig = px.histogram(
            features_df, 
            x='word_count', 
            nbins=30,
            title="Distribution of Word Counts",
            labels={'word_count': 'Word Count', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, width="stretch")
        
        # Quality distribution
        if 'quality_label' in features_df.columns:
            quality_counts = features_df['quality_label'].value_counts()
            
            fig = px.pie(
                values=quality_counts.values,
                names=quality_counts.index,
                title="Quality Distribution"
            )
            st.plotly_chart(fig, width="stretch")

def duplicates_tab(duplicates_df, features_df):
    """Duplicates analysis tab"""
    st.markdown('<h2 class="section-header">Duplicate Content Analysis</h2>', unsafe_allow_html=True)
    
    if duplicates_df is not None and len(duplicates_df) > 0:
        # Calculate statistics
        valid_pages = len(features_df[features_df['word_count'] > 0])
        duplication_rate = len(duplicates_df) / valid_pages * 100 if valid_pages > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duplicate Pairs", len(duplicates_df))
        with col2:
            st.metric("Valid Pages Analyzed", valid_pages)
        with col3:
            st.metric("Duplication Rate", f"{duplication_rate:.1f}%")
        
        # Show similarity statistics
        st.markdown("### 📊 Similarity Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min Similarity", f"{duplicates_df['similarity'].min():.3f}")
        with col2:
            st.metric("Max Similarity", f"{duplicates_df['similarity'].max():.3f}")
        with col3:
            st.metric("Mean Similarity", f"{duplicates_df['similarity'].mean():.3f}")
        with col4:
            st.metric("Std Similarity", f"{duplicates_df['similarity'].std():.3f}")
        
        # Similarity distribution
        fig = px.histogram(
            duplicates_df,
            x='similarity',
            nbins=20,
            title="Distribution of Similarity Scores (Corrected)",
            labels={'similarity': 'Similarity Score', 'count': 'Frequency'}
        )
        fig.add_vline(x=0.8, line_dash="dash", line_color="red", 
                      annotation_text="Threshold: 0.8")
        st.plotly_chart(fig, width="stretch")
        
        # Top duplicates
        st.markdown("### 🔝 Top Duplicate Pairs")
        top_duplicates = duplicates_df.nlargest(10, 'similarity')
        
        for idx, row in top_duplicates.iterrows():
            with st.expander(f"Similarity: {row['similarity']:.3f}"):
                st.write(f"**URL 1:** {row['url1']}")
                st.write(f"**URL 2:** {row['url2']}")
    else:
        st.info("No duplicate content found above the similarity threshold.")
        st.markdown("### 🎯 This is actually a good result!")
        st.markdown("""
        Having few or no duplicates means:
        - Content is unique and diverse
        - No algorithmic errors producing false positives
        - Quality content analysis is working correctly
        """)

if __name__ == "__main__":
    main()
