"""
Quality Scoring and Similarity Detection for Streamlit App
"""
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Initialize sentence transformer model
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def predict_quality(model, content_info, features):
    """
    Predict content quality using the trained model.
    """
    try:
        # Prepare feature vector
        feature_vector = np.array([[
            content_info['word_count'],
            features['sentence_count'],
            features['flesch_reading_ease'],
            features['avg_word_length']
        ]])
        
        # Predict quality
        quality_prediction = model.predict(feature_vector)[0]
        quality_proba = model.predict_proba(feature_vector)[0]
        
        # Map probabilities to labels (assuming order: High, Low, Medium)
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
            'confidence': {'High': 0, 'Medium': 0, 'Low': 0},
            'error': str(e)
        }

def check_similarity(content_info, features_df, threshold=0.75):
    """
    Check similarity with existing content in the dataset.
    """
    try:
        # Load sentence transformer model
        model = load_sentence_model()
        
        # Generate embedding for new content
        combined_text = f"{content_info['title']} {content_info['body_text']}"[:512]
        new_embedding = model.encode([combined_text])[0]
        
        # Get existing embeddings
        existing_embeddings = np.array([np.array(emb) for emb in features_df['embedding']])
        
        # Compute similarities
        similarities = cosine_similarity([new_embedding], existing_embeddings)[0]
        
        # Find similar content above threshold
        similar_content = []
        for i, sim_score in enumerate(similarities):
            if sim_score > threshold:
                similar_content.append({
                    'url': features_df.iloc[i]['url'],
                    'similarity': float(sim_score)
                })
        
        # Sort by similarity
        similar_content = sorted(similar_content, key=lambda x: x['similarity'], reverse=True)
        
        return similar_content[:5]  # Return top 5
    
    except Exception as e:
        return []