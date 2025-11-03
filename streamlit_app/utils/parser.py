"""
HTML Content Parser for Streamlit App
"""
from bs4 import BeautifulSoup
import re

def extract_content_from_html(html_content):
    """
    Extract title, body text, and word count from HTML content.
    Returns a dictionary with extracted information.
    """
    try:
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else "No Title"
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text from main content areas
        body_text = ""
        
        # Try to find main content in common tags
        content_tags = soup.find_all(['p', 'article', 'main', 'div', 'section'])
        for tag in content_tags:
            text = tag.get_text()
            if len(text.strip()) > 20:  # Only add meaningful text blocks
                body_text += text + " "
        
        # If no content found, use body text
        if not body_text.strip():
            body = soup.find('body')
            body_text = body.get_text() if body else soup.get_text()
        
        # Clean the text
        body_text = re.sub(r'\s+', ' ', body_text).strip()
        
        # Calculate word count
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
            'word_count': 0,
            'error': str(e)
        }