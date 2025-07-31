from .search_stock import search_stock_images
from .search_google import search_google_images
from .generate_ai import generate_ai_images, generate_openai_images

def run_stock_search(terms, max_images, output_dir):
    """Wrapper for stock image search"""
    return search_stock_images(terms, max_images, output_dir)

def run_google_search(terms, max_images, output_dir):
    """Wrapper for Google image search"""
    return search_google_images(terms, max_images, output_dir)

def run_ai_generation(terms, max_images, output_dir):
    """Wrapper for AI image generation using Imagen 3"""
    return generate_ai_images(terms, output_dir)

def run_openai_generation(terms, max_images, output_dir):
    """Wrapper for AI image generation using OpenAI DALL-E"""
    return generate_openai_images(terms, max_images, output_dir)
