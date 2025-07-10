import os
import logging
import requests
import re
from dotenv import load_dotenv
from PIL import Image
import io
import base64
from typing import List, Optional

from ..decision_logger import log_decision

load_dotenv()

def sanitize_filename(text: str, max_length: int = 20) -> str:
    """
    Sanitize text for use as a Windows-compatible filename.
    
    Args:
        text: The text to sanitize
        max_length: Maximum length of the resulting filename part
    
    Returns:
        A sanitized string safe for use in filenames
    """
    # Remove invalid Windows filename characters: < > : " | ? * \ /
    safe_text = re.sub(r'[<>:"|?*\\/]', '', text[:max_length])
    # Replace spaces, dots, and commas with underscores
    safe_text = safe_text.replace(' ', '_').replace('.', '_').replace(',', '')
    # Remove leading/trailing underscores and return
    return safe_text.strip('_')

# Try multiple AI image generation APIs
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
STABILITY_API_KEY = os.getenv('STABILITY_API_KEY')

def generate_ai_images(prompts: list, output_dir: str):
    """
    Main entry point for AI image generation.
    """
    return run(prompts, output_dir)

def run(prompts: list, output_dir: str):
    """
    Generates images using AI image generation APIs.
    
    Priority order: OpenAI DALL-E > Stability AI > Gemini 2.0 Flash
    
    Args:
        prompts (list): List of text prompts for image generation
        output_dir (str): Directory to save generated images
    Returns:
        list: List of generated image paths
    """
    # Try OpenAI DALL-E first
    if OPENAI_API_KEY:
        results = generate_with_openai(prompts, output_dir)
        if results:
            return results
    
    # Try Stability AI as fallback
    if STABILITY_API_KEY:
        results = generate_with_stability(prompts, output_dir)
        if results:
            return results
    
    # Try Gemini 2.0 Flash as last resort (but now it works!)
    if GEMINI_API_KEY:
        results = generate_with_gemini(prompts, output_dir)
        if results:
            return results
    
    logging.error("No AI image generation API keys found. Set OPENAI_API_KEY, STABILITY_API_KEY, or GEMINI_API_KEY")
    return []

def generate_with_openai(prompts: list, output_dir: str):
    """
    Generate images using OpenAI DALL-E API.
    """
    if not OPENAI_API_KEY:
        return []
    
    generated_images = []
    
    try:
        for i, prompt in enumerate(prompts):
            logging.info(f"  > Generating DALL-E image {i+1}/{len(prompts)}: '{prompt[:50]}...'")
            
            try:
                # OpenAI DALL-E API endpoint
                url = "https://api.openai.com/v1/images/generations"
                headers = {
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                # Optimize prompt for DALL-E with portrait orientation
                optimized_prompt = f"{prompt}, high quality, detailed, vertical composition, portrait orientation, 9:16 aspect ratio"
                
                data = {
                    "model": "dall-e-3",
                    "prompt": optimized_prompt[:4000],  # DALL-E has a 4000 char limit
                    "n": 1,
                    "size": "1024x1792",  # Portrait aspect ratio (approximately 9:16)
                    "quality": "standard",
                    "response_format": "url"
                }
                
                response = requests.post(url, headers=headers, json=data, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                
                if result.get('data') and len(result['data']) > 0:
                    image_url = result['data'][0]['url']
                    
                    # Download the generated image
                    img_response = requests.get(image_url, timeout=30)
                    img_response.raise_for_status()
                    
                    # Save and standardize image to 1080x1920
                    image_filename = f"dalle_{i:02d}.jpg"  # Use .jpg for standardized output
                    temp_path = os.path.join(output_dir, f"dalle_{i:02d}_temp.png")
                    image_path = os.path.join(output_dir, image_filename)
                    
                    # First save the original image temporarily
                    with open(temp_path, 'wb') as f:
                        f.write(img_response.content)
                    
                    # Standardize the image to 1080x1920 pixels
                    from agent.utils import standardize_image_for_video
                    try:
                        standardize_image_for_video(temp_path, image_path)
                        
                        # Remove temporary file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                        if os.path.exists(image_path) and os.path.getsize(image_path) > 1000:
                            generated_images.append(image_path)
                            logging.info(f"    > Saved and standardized DALL-E image: {image_filename}")
                        else:
                            if os.path.exists(image_path):
                                os.remove(image_path)
                    except Exception as e:
                        logging.error(f"Failed to standardize DALL-E image: {e}")
                        # Clean up temporary files
                        for cleanup_path in [temp_path, image_path]:
                            if os.path.exists(cleanup_path):
                                os.remove(cleanup_path)
                else:
                    logging.warning(f"  > No image returned from DALL-E for prompt: '{prompt[:50]}...'")
                    
            except requests.exceptions.RequestException as e:
                logging.error(f"OpenAI API error for prompt '{prompt[:50]}...': {e}")
                continue
            except Exception as e:
                logging.error(f"Error generating DALL-E image for prompt '{prompt[:50]}...': {e}")
                continue
                    
        logging.info(f"  > Successfully generated {len(generated_images)} DALL-E images")
        return generated_images
        
    except Exception as e:
        logging.error(f"Critical error in DALL-E generation: {e}")
        return []

def generate_with_stability(prompts: list, output_dir: str):
    """
    Generate images using Stability AI API.
    """
    if not STABILITY_API_KEY:
        return []
    
    generated_images = []
    
    try:
        for i, prompt in enumerate(prompts):
            logging.info(f"  > Generating Stability AI image {i+1}/{len(prompts)}: '{prompt[:50]}...'")
            
            try:
                url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
                
                headers = {
                    "Authorization": f"Bearer {STABILITY_API_KEY}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                
                data = {
                    "text_prompts": [
                        {
                            "text": f"{prompt}, vertical composition, portrait orientation, 9:16 aspect ratio",
                            "weight": 1
                        }
                    ],
                    "cfg_scale": 7,
                    "height": 1792,  # Portrait height for 9:16 aspect ratio
                    "width": 1024,   # Portrait width for 9:16 aspect ratio
                    "samples": 1,
                    "steps": 30,
                }
                
                response = requests.post(url, headers=headers, json=data, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                
                if result.get('artifacts') and len(result['artifacts']) > 0:
                    # Stability AI returns base64 encoded images
                    image_data = base64.b64decode(result['artifacts'][0]['base64'])
                    
                    # Save and standardize image to 1080x1920
                    image_filename = f"stability_{i:02d}.jpg"  # Use .jpg for standardized output
                    temp_path = os.path.join(output_dir, f"stability_{i:02d}_temp.png")
                    image_path = os.path.join(output_dir, image_filename)
                    
                    # First save the original image temporarily
                    with open(temp_path, 'wb') as f:
                        f.write(image_data)
                    
                    # Standardize the image to 1080x1920 pixels
                    from agent.utils import standardize_image_for_video
                    try:
                        standardize_image_for_video(temp_path, image_path)
                        
                        # Remove temporary file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                        if os.path.exists(image_path) and os.path.getsize(image_path) > 1000:
                            generated_images.append(image_path)
                            logging.info(f"    > Saved and standardized Stability AI image: {image_filename}")
                        else:
                            if os.path.exists(image_path):
                                os.remove(image_path)
                    except Exception as e:
                        logging.error(f"Failed to standardize Stability AI image: {e}")
                        # Clean up temporary files
                        for cleanup_path in [temp_path, image_path]:
                            if os.path.exists(cleanup_path):
                                os.remove(cleanup_path)
                else:
                    logging.warning(f"  > No image returned from Stability AI for prompt: '{prompt[:50]}...'")
                    
            except Exception as e:
                logging.error(f"Error generating Stability AI image for prompt '{prompt[:50]}...': {e}")
                continue
        
        logging.info(f"  > Successfully generated {len(generated_images)} Stability AI images")
        return generated_images
        
    except Exception as e:
        logging.error(f"Critical error in Stability AI generation: {e}")
        return []

def generate_with_gemini(prompts: list, output_dir: str):
    """
    Generate images using the Gemini 2.0 Flash Image Generation model.
    This implementation matches the working test script.
    """
    if not GEMINI_API_KEY:
        logging.warning("GEMINI_API_KEY not found. Skipping Gemini image generation.")
        return []
    
    generated_images = []
    model_id = "gemini-2.0-flash-preview-image-generation"
    
    for i, prompt in enumerate(prompts):
        logging.info(f"  > Generating Gemini image {i+1}/{len(prompts)} with model '{model_id}': '{prompt[:50]}...'")
        
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GEMINI_API_KEY}"
            headers = {'Content-Type': 'application/json'}
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}
            }

            response = requests.post(url, headers=headers, json=data, timeout=90)
            response.raise_for_status()
            
            response_data = response.json()

            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                candidate = response_data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    for part in parts:
                        if 'inlineData' in part and 'data' in part['inlineData']:
                            b64_string = part['inlineData']['data']
                            image_bytes = base64.b64decode(b64_string)
                            
                            # Save the image with sanitized filename
                            safe_prompt = sanitize_filename(prompt)
                            file_name = f"ai_generated_{i}_{safe_prompt}.png"
                            img_path = os.path.join(output_dir, file_name)
                            with open(img_path, "wb") as f:
                                f.write(image_bytes)
                            
                            generated_images.append(img_path)
                            logging.info(f"    > AI Image saved successfully as '{img_path}'")
                            break # Assume one image per prompt
            else:
                logging.warning(f"No valid image content found in Gemini response for prompt: {prompt}")

        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling Gemini Image Generation API for prompt '{prompt[:50]}...': {e}")
            # Optionally, inspect the response body for more details
            if e.response is not None:
                logging.error(f"    > Response body: {e.response.text}")
            continue
        except Exception as e:
            logging.error(f"An unexpected error occurred during Gemini image generation for prompt '{prompt[:50]}...': {e}")
            continue
            
    return generated_images

def generate_with_style(prompts: list, style: str, output_dir: str):
    """
    Generates images with a specific style applied to all prompts.
    
    Args:
        prompts (list): List of base prompts
        style (str): Style to apply (e.g., "photorealistic", "digital art", "watercolor")
        output_dir (str): Directory to save generated images
    Returns:
        list: List of generated image paths
    """
    # Enhance prompts with style
    styled_prompts = []
    for prompt in prompts:
        if style == "photorealistic":
            styled_prompt = f"{prompt}, photorealistic, high quality, professional photography, 8k resolution"
        elif style == "digital art":
            styled_prompt = f"{prompt}, digital art style, vibrant colors, modern illustration"
        elif style == "watercolor":
            styled_prompt = f"{prompt}, watercolor painting style, soft colors, artistic"
        elif style == "tech":
            styled_prompt = f"{prompt}, futuristic tech style, neon colors, cyberpunk aesthetic"
        else:
            styled_prompt = f"{prompt}, {style}"
        
        styled_prompts.append(styled_prompt)
    
    return run(styled_prompts, output_dir)

def generate_for_video_concepts(concepts: list, script_context: str, output_dir: str):
    """
    Generates images specifically optimized for video content.
    
    Args:
        concepts (list): List of visual concepts needed
        script_context (str): Context from the script to ensure consistency
        output_dir (str): Directory to save generated images
    Returns:
        list: List of generated image paths
    """
    # Create prompts optimized for video
    video_prompts = []
    
    for concept in concepts:
        # Add video-friendly specifications with portrait orientation
        prompt = (
            f"{concept}, "
            f"9:16 aspect ratio, portrait orientation, "
            f"vertical composition, "
            f"clear focal point, "
            f"high contrast, "
            f"suitable for mobile video, "
            f"professional quality"
        )
        
        # Add context if it helps
        if "tech" in script_context.lower() or "ai" in script_context.lower():
            prompt += ", modern technology aesthetic"
        elif "game" in script_context.lower():
            prompt += ", gaming aesthetic, vibrant"
        
        video_prompts.append(prompt)
    
    return run(video_prompts, output_dir) 

def generate_ai_images(terms: List[str], max_images: int, output_dir: str, debug: bool = False) -> List[str]:
    """Generate AI images using Gemini 2.0 Flash Image Generation"""
    
    if debug:
        log_decision("ai_image_generation", 
                    f"Starting AI image generation for {len(terms)} terms: {terms}", 
                    f"Attempting to generate {max_images} images using Gemini 2.0 Flash",
                    0.8, 
                    ["Use OpenAI DALL-E", "Use Stability AI", "Skip AI generation"])
    
    # Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        error_msg = "GEMINI_API_KEY environment variable not set"
        if debug:
            log_decision("ai_image_generation", 
                        "API key missing", 
                        error_msg,
                        0.0, 
                        ["Set up API key", "Use different image source"])
        raise Exception(error_msg)
    
    generated_files = []
    
    for i, term in enumerate(terms[:max_images]):
        if debug:
            log_decision("ai_image_generation", 
                        f"Generating image {i+1}/{min(len(terms), max_images)}", 
                        f"Creating image for term: '{term}' using Gemini 2.0 Flash",
                        0.7, 
                        ["Skip this term", "Use different prompt"])
        
        try:
            # The 'term' is now a fully-formed prompt from the collector
            prompt = term
            
            # Use Gemini 2.0 Flash model
            model_id = "gemini-2.0-flash-preview-image-generation"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"]
                }
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        
                        for part in parts:
                            if 'inlineData' in part:
                                inline_data = part['inlineData']
                                if 'data' in inline_data:
                                    # Decode and save image
                                    image_data = base64.b64decode(inline_data['data'])
                                    
                                    # Sanitize filename for Windows compatibility - use .jpg for standardized output
                                    safe_term = sanitize_filename(term)
                                    filename = f"ai_generated_{i}_{safe_term}.jpg"
                                    temp_filename = f"ai_generated_{i}_{safe_term}_temp.png"
                                    filepath = os.path.join(output_dir, filename)
                                    temp_filepath = os.path.join(output_dir, temp_filename)
                                    
                                    # First save the original image temporarily
                                    with open(temp_filepath, 'wb') as f:
                                        f.write(image_data)
                                    
                                    # Standardize the image to 1080x1920 pixels
                                    from agent.utils import standardize_image_for_video
                                    try:
                                        standardize_image_for_video(temp_filepath, filepath)
                                        
                                        # Remove temporary file
                                        if os.path.exists(temp_filepath):
                                            os.remove(temp_filepath)
                                        
                                        generated_files.append(filepath)
                                    except Exception as e:
                                        logging.error(f"Failed to standardize Gemini image: {e}")
                                        # Clean up temporary files
                                        for cleanup_path in [temp_filepath, filepath]:
                                            if os.path.exists(cleanup_path):
                                                os.remove(cleanup_path)
                                        continue
                                    
                                    if debug:
                                        log_decision("ai_image_generation", 
                                                    f"Successfully generated image {i+1}", 
                                                    f"Image saved to {filepath} ({len(image_data)} bytes)",
                                                    0.9, 
                                                    ["Generate another variation", "Move to next term"])
                                    break  # Take first image found
                        
                        if not any('inlineData' in part for part in parts):
                            if debug:
                                log_decision("ai_image_generation", 
                                            f"Image {i+1} filtered by safety", 
                                            f"No image data in response for term: {term}",
                                            0.3, 
                                            ["Try different prompt", "Skip this term"])
                    else:
                        if debug:
                            log_decision("ai_image_generation", 
                                        f"No content in response for image {i+1}", 
                                        f"Invalid response structure for term: {term}",
                                        0.2, 
                                        ["Retry with different prompt", "Skip this term"])
                else:
                    if debug:
                        log_decision("ai_image_generation", 
                                    f"No candidates in response for image {i+1}", 
                                    f"Empty response for term: {term}",
                                    0.2, 
                                    ["Retry with different prompt", "Skip this term"])
            else:
                error_msg = f"Gemini API error: {response.status_code} - {response.text}"
                if debug:
                    log_decision("ai_image_generation", 
                                f"API error for image {i+1}", 
                                error_msg,
                                0.1, 
                                ["Retry request", "Use fallback image source"])
                
                # Pipeline should crash on API failure as requested
                raise Exception(error_msg)
                
        except Exception as e:
            error_msg = f"Failed to generate AI image for '{term}': {str(e)}"
            if debug:
                log_decision("ai_image_generation", 
                            f"Exception during image {i+1} generation", 
                            error_msg,
                            0.0, 
                            ["Retry", "Skip this term", "Use fallback source"])
            
            # Pipeline should crash on API failure as requested
            raise Exception(error_msg)
    
    if debug:
        log_decision("ai_image_generation", 
                    f"AI image generation completed", 
                    f"Generated {len(generated_files)} images: {[os.path.basename(f) for f in generated_files]}",
                    0.9 if generated_files else 0.1, 
                    ["Generate more images", "Use these images"])
    
    return generated_files

# Alternative implementation using OpenAI DALL-E as fallback
def generate_openai_images(terms: List[str], max_images: int, output_dir: str, debug: bool = False) -> List[str]:
    """Generate AI images using OpenAI DALL-E"""
    
    if debug:
        log_decision("ai_image_generation", 
                    "Using OpenAI DALL-E fallback", 
                    f"Attempting to generate {max_images} images using DALL-E",
                    0.7, 
                    ["Use different AI service", "Skip AI generation"])
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        error_msg = "OPENAI_API_KEY environment variable not set"
        if debug:
            log_decision("ai_image_generation", 
                        "OpenAI API key missing", 
                        error_msg,
                        0.0, 
                        ["Set up API key", "Use different image source"])
        raise Exception(error_msg)
    
    generated_files = []
    
    for i, term in enumerate(terms[:max_images]):
        try:
            prompt = f"A professional, high-quality image representing: {term}. Modern, clean style suitable for a technology presentation."
            
            url = "https://api.openai.com/v1/images/generations"
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024",
                "response_format": "b64_json"
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                if 'data' in result and len(result['data']) > 0:
                    image_data = base64.b64decode(result['data'][0]['b64_json'])
                    
                    # Sanitize filename for Windows compatibility
                    safe_term = sanitize_filename(term)
                    filename = f"ai_generated_{i}_{safe_term}.png"
                    filepath = os.path.join(output_dir, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(image_data)
                    
                    generated_files.append(filepath)
                    
                    if debug:
                        log_decision("ai_image_generation", 
                                    f"OpenAI image {i+1} generated", 
                                    f"Image saved to {filepath}",
                                    0.8, 
                                    ["Generate another", "Move to next term"])
            else:
                error_msg = f"OpenAI API error: {response.status_code} - {response.text}"
                if debug:
                    log_decision("ai_image_generation", 
                                f"OpenAI API error for image {i+1}", 
                                error_msg,
                                0.1, 
                                ["Retry", "Skip this term"])
                raise Exception(error_msg)
                
        except Exception as e:
            error_msg = f"Failed to generate OpenAI image for '{term}': {str(e)}"
            if debug:
                log_decision("ai_image_generation", 
                            f"OpenAI generation failed for image {i+1}", 
                            error_msg,
                            0.0, 
                            ["Retry", "Skip this term"])
            raise Exception(error_msg)
    
    return generated_files 