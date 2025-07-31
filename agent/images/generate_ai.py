import vertexai
import os
from vertexai.preview.vision_models import ImageGenerationModel
import logging
import requests
import re
import json
from dotenv import load_dotenv
from PIL import Image
import io
import base64
from typing import List, Optional, Dict, Any
import google.generativeai as genai

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

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'image_generation_config.json')
IMAGE_GEN_CONFIG = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        IMAGE_GEN_CONFIG = json.load(f).get('ai_image_generation', {})
else:
    logging.warning(f"Image generation config not found at {CONFIG_PATH}, using defaults")

def generate_ai_images(prompts: list, output_dir: str):
    """
    Main entry point for AI image generation.
    """
    return run(prompts, output_dir)

def run(prompts: list, output_dir: str):
    """
    Generates images using AI image generation APIs.
    
    Uses configuration to determine provider order.
    
    Args:
        prompts (list): List of text prompts for image generation
        output_dir (str): Directory to save generated images
    Returns:
        list: List of generated image paths
    """
    # Get provider order from config
    primary_provider = IMAGE_GEN_CONFIG.get('primary_provider', 'vertex_ai')
    fallback_providers = IMAGE_GEN_CONFIG.get('fallback_providers', ['openai', 'gemini', 'stability'])
    
    # Create ordered list of providers to try
    providers_to_try = [primary_provider] + [p for p in fallback_providers if p != primary_provider]
    
    for provider in providers_to_try:
        provider_config = IMAGE_GEN_CONFIG.get('providers', {}).get(provider, {})
        if not provider_config.get('enabled', True):
            continue
            
        if provider == 'vertex_ai':
            # Check if we have the required environment variables for Vertex AI
            if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') and os.getenv('GOOGLE_CLOUD_PROJECT'):
                results = generate_with_vertex_ai_imagen(prompts, output_dir)
                if results:
                    return results
        elif provider == 'openai' and OPENAI_API_KEY:
            results = generate_with_openai(prompts, output_dir)
            if results:
                return results
        elif provider == 'gemini':
            # Gemini image generation temporarily disabled due to model availability issues
            logging.warning("Gemini image generation is temporarily disabled - using other providers")
            continue
        elif provider == 'stability' and STABILITY_API_KEY:
            results = generate_with_stability(prompts, output_dir)
            if results:
                return results
    
    logging.error("No AI image generation API keys found or all providers failed. Set up Vertex AI credentials, OPENAI_API_KEY, STABILITY_API_KEY, or GEMINI_API_KEY")
    return []

def generate_with_openai(prompts: list, output_dir: str):
    """
    Generate images using OpenAI DALL-E API.
    """
    if not OPENAI_API_KEY:
        return []
    
    generated_images = []
    config = IMAGE_GEN_CONFIG.get('providers', {}).get('openai', {})
    prompt_config = IMAGE_GEN_CONFIG.get('common_settings', {}).get('prompt_enhancement', {})
    
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
                
                # Enhance prompt if configured
                optimized_prompt = prompt
                if prompt_config.get('add_style_suffix', True):
                    optimized_prompt = f"{prompt}, {prompt_config.get('style_suffix', 'high quality')}"
                
                data = {
                    "model": config.get('model', 'dall-e-3'),
                    "prompt": optimized_prompt[:4000],  # 4000 char limit
                    "n": 1,
                    "size": config.get('size', '1024x1792'),
                    "quality": config.get('quality', 'standard'),
                    "style": config.get('style', 'natural')
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

from google.cloud import aiplatform
from google.auth import default

def generate_with_gemini(prompts: list, output_dir: str):
    """
    Generate images using Vertex AI Gemini model.
    """
    # Set up project details
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'gen-lang-client-0681207774')
    location = "us-central1"
    vertexai.init(project=project_id, location=location)

    # Load the pre-trained Gemini model
    model = ImageGenerationModel.from_pretrained("geminigeneration@latest")

    generated_files = []

    for i, prompt in enumerate(prompts):
        try:
            logging.info(f"   Generating Vertex AI Gemini image {i+1}/{len(prompts)}: '{prompt[:50]}...'")

            # Use the model's generate_images method
            response = model.generate_images(prompt=prompt, number_of_images=1)

            # Save each generated image
            image_filename = f"gemini_{i:02d}.jpg"
            temp_path = os.path.join(output_dir, f"gemini_{i:02d}_temp.png")
            image_path = os.path.join(output_dir, f"gemini_{i:02d}.jpg")  # Standardized output

            # Save temporary image first
            response.images[0].save(location=temp_path)

            # Standardize the image to 1080x1920 pixels
            from agent.utils import standardize_image_for_video
            try:
                standardize_image_for_video(temp_path, image_path)

                # Remove temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                if os.path.exists(image_path) and os.path.getsize(image_path) > 1000:
                    generated_files.append(image_path)
                    logging.info(f"     Saved and standardized Vertex AI Gemini image: {os.path.basename(image_path)}")
                else:
                    if os.path.exists(image_path):
                        os.remove(image_path)
            except Exception as e:
                logging.error(f"Failed to standardize Vertex AI Gemini image: {e}")
                # Clean up temporary files
                for cleanup_path in [temp_path, image_path]:
                    if os.path.exists(cleanup_path):
                        os.remove(cleanup_path)
                continue

        except Exception as e:
            logging.error(f"Failed to generate Vertex AI Gemini image for '{prompt}': {e}")

    logging.info(f"   Successfully generated {len(generated_files)} Vertex AI Gemini images")
    return generated_files

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

def generate_with_vertex_ai_imagen(prompts: List[str], output_dir: str):
    """Generate AI images using Vertex AI Imagen with ImageGenerationModel"""

    # Set up project details
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'gen-lang-client-0681207774')
    location = "us-central1"
    vertexai.init(project=project_id, location=location)

    # Load the pre-trained Imagen model using its simple identifier
    model = ImageGenerationModel.from_pretrained("imagegeneration@007")

    generated_files = []

    for i, prompt in enumerate(prompts):
        try:
            logging.info(f"  > Generating Vertex AI Imagen image {i+1}/{len(prompts)}: '{prompt[:50]}...'")

            # Use ImageGenerationModel's generate_images method
            response = model.generate_images(prompt=prompt, number_of_images=1)

            # Save each generated image
            image_filename = f"imagen_{i:02d}.png"
            temp_path = os.path.join(output_dir, f"imagen_{i:02d}_temp.png")
            image_path = os.path.join(output_dir, f"imagen_{i:02d}.jpg")  # Standardized output

            # Save temporary image first
            response.images[0].save(location=temp_path)

            # Standardize the image to 1080x1920 pixels
            from agent.utils import standardize_image_for_video
            try:
                standardize_image_for_video(temp_path, image_path)
                
                # Remove temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                if os.path.exists(image_path) and os.path.getsize(image_path) > 1000:
                    generated_files.append(image_path)
                    logging.info(f"    > Saved and standardized Vertex AI Imagen image: {os.path.basename(image_path)}")
                else:
                    if os.path.exists(image_path):
                        os.remove(image_path)
            except Exception as e:
                logging.error(f"Failed to standardize Vertex AI Imagen image: {e}")
                # Clean up temporary files
                for cleanup_path in [temp_path, image_path]:
                    if os.path.exists(cleanup_path):
                        os.remove(cleanup_path)
                continue

        except Exception as e:
            logging.error(f"Failed to generate Vertex AI Imagen image for '{prompt}': {e}")

    logging.info(f"  > Successfully generated {len(generated_files)} Vertex AI Imagen images")
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