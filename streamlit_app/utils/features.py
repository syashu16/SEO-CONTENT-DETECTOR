"""
Feature Extraction for Streamlit App
"""
import textstat
import re

def extract_features(text):
    """
    Extract comprehensive features from text content.
    Returns dictionary with readability, keywords, and other metrics.
    """
    try:
        if not text or len(text.strip()) == 0:
            return {
                'sentence_count': 0,
                'flesch_reading_ease': 0,
                'avg_word_length': 0
            }
        
        # Clean text
        clean_text = re.sub(r'\s+', ' ', text.lower().strip())
        
        # Calculate sentence count
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Calculate Flesch Reading Ease score
        flesch_score = textstat.flesch_reading_ease(text)
        
        # Calculate average word length
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
            'avg_word_length': 0,
            'error': str(e)
        }