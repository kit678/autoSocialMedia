import os
import json
import logging
import requests
from dotenv import load_dotenv
from agent.utils import http_retry_session

load_dotenv()

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
API_URL = "https://api.deepseek.com/v1/chat/completions"

KEYWORD_EXTRACTION_PROMPT = """
You are an expert at identifying visual concepts for video production. Given a video script, extract 8-12 specific visual keywords/phrases that would make excellent search terms for stock images or screenshots.

RULES:
- Focus on concrete, visual nouns and concepts
- Include company names, technology terms, and product names mentioned
- Prefer specific terms over generic ones (e.g., "artificial intelligence robot" over "technology")
- Include both technical terms and human/emotional concepts
- Each keyword should be 1-4 words maximum
- Return ONLY a valid JSON array of strings

EXAMPLES:
Script: "AI just revolutionized healthcare with new diagnostic tools..."
Response: ["artificial intelligence healthcare", "medical diagnosis", "AI doctor", "hospital technology", "healthcare innovation", "medical AI tools", "diagnostic equipment", "futuristic medicine"]

Script: "Tesla's new robot can now cook dinner..."
Response: ["Tesla robot", "cooking robot", "kitchen automation", "robotic chef", "Tesla humanoid", "AI cooking", "home robotics", "automated kitchen"]

Now extract keywords from this script:
"""

def run(script_text: str, audio_duration: float = 40.0):
    """
    Extracts visual keywords from script text using DeepSeek.
    Args:
        script_text (str): The video script text
        audio_duration (float): Duration of audio in seconds (for calculating number of images needed)
    Returns:
        dict: Contains 'keywords' list and 'num_images_needed' int, or None on failure
    """
    if not DEEPSEEK_API_KEY:
        logging.error("DEEPSEEK_API_KEY not found in .env file.")
        return None
    
    if not script_text:
        logging.error("No script text provided for keyword extraction.")
        return None
    
    # Calculate how many images we need based on audio duration
    # Assuming 4-6 seconds per image for good pacing
    seconds_per_image = 5.0
    num_images_needed = max(3, int(audio_duration / seconds_per_image))
    
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {DEEPSEEK_API_KEY}'}
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": KEYWORD_EXTRACTION_PROMPT
            },
            {
                "role": "user", 
                "content": script_text
            }
        ],
        "max_tokens": 500,
        "temperature": 0.3  # Lower temperature for more consistent JSON output
    }
    
    try:
        session = http_retry_session()
        response = session.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        content = data['choices'][0]['message']['content'].strip()
        
        # Parse the JSON response
        try:
            # Clean up response - remove markdown code blocks if present
            content_clean = content.strip()
            if content_clean.startswith('```json'):
                content_clean = content_clean[7:]  # Remove ```json
            if content_clean.startswith('```'):
                content_clean = content_clean[3:]  # Remove ```
            if content_clean.endswith('```'):
                content_clean = content_clean[:-3]  # Remove trailing ```
            content_clean = content_clean.strip()
            
            keywords = json.loads(content_clean)
            if not isinstance(keywords, list):
                raise ValueError("Response is not a list")
            
            # Ensure we have strings
            keywords = [str(kw).strip() for kw in keywords if str(kw).strip()]
            
            if not keywords:
                raise ValueError("No valid keywords extracted")
            
            logging.info(f"  > Extracted {len(keywords)} visual keywords")
            logging.info(f"  > Keywords: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
            
            return {
                'keywords': keywords,
                'num_images_needed': num_images_needed,
                'seconds_per_image': seconds_per_image
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to parse keywords JSON: {e}")
            logging.error(f"Raw response: {content}")
            
            raise Exception(f"Failed to parse keywords JSON: {e}")
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling DeepSeek API for keyword extraction: {e}")
        
        raise Exception(f"Error calling DeepSeek API: {e}") 