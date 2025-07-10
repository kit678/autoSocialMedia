import os
import logging
import requests
from dotenv import load_dotenv
from agent.utils import http_retry_session
import urllib.parse
from typing import List, Optional

load_dotenv()

UNSPLASH_ACCESS_KEY = os.getenv('UNSPLASH_ACCESS_KEY')
UNSPLASH_API_URL = "https://api.unsplash.com/search/photos"

def search_stock_images(search_terms: List[str], num_images: int, download_dir: str, category: Optional[str] = None) -> List[str]:
    """
    Main entry point for stock image search.
    """
    return search_pexels_primary(search_terms, num_images, download_dir, category=category)

def search_pexels_primary(search_terms: List[str], num_images: int, download_dir: str, category: Optional[str] = None) -> List[str]:
    """
    Search Pexels for primary images with an optional category filter.
    """
    pexels_api_key = os.getenv('PEXELS_API_KEY')
    if not pexels_api_key:
        logging.warning("PEXELS_API_KEY not found. Skipping Pexels search.")
        return []
    
    os.makedirs(download_dir, exist_ok=True)
    downloaded_images = []
    
    try:
        session = http_retry_session()
        headers = {'Authorization': pexels_api_key}
        
        for term in search_terms:
            if len(downloaded_images) >= num_images:
                break
                
            pexels_url = "https://api.pexels.com/v1/search"
            params = {
                'query': term,
                'per_page': min(num_images, 10),
                'orientation': 'portrait'
            }
            
            # Add category if provided
            if category:
                params['category'] = category
            
            try:
                logging.info(f"Searching Pexels for: '{term}'" + (f" in category '{category}'" if category else ""))
                response = session.get(pexels_url, headers=headers, params=params, timeout=15)
                response.raise_for_status()
                
                data = response.json()
                photos = data.get('photos', [])
                
                for photo in photos:
                    if len(downloaded_images) >= num_images:
                        break
                    
                    image_url = photo['src']['large']
                    image_id = photo['id']
                    
                    filename = f"pexels_{term.replace(' ', '_')}_{image_id}.jpg"
                    path = os.path.join(download_dir, filename)
                    
                    if _download_image(session, image_url, path):
                        # Save URL metadata
                        _save_url_metadata(path, photo['url'])
                        downloaded_images.append(path)
                        logging.info(f"  > Downloaded Pexels photo: {filename}")
                        
            except requests.exceptions.RequestException as e:
                logging.error(f"Pexels API search failed for term '{term}': {e}")
                continue
                
    except Exception as e:
        logging.error(f"Error in Pexels primary search: {e}")
        
    return downloaded_images

def run(keywords: list, num_images: int, output_dir: str):
    """
    Legacy wrapper - now uses Pexels as primary source.
    """
    return search_pexels_primary(keywords, num_images, output_dir)

def _download_image(session, image_url: str, output_path: str, max_size_mb: int = 5):
    """
    Downloads an image from URL, standardizes it to 1080x1920 pixels, and saves to local path.
    Args:
        session: requests session
        image_url (str): URL of the image
        output_path (str): Local path to save the standardized image
        max_size_mb (int): Maximum file size in MB for download
    Returns:
        bool: True on success, False on failure
    """
    # Use the standardized download and resize function
    from agent.utils import download_and_standardize_image
    
    try:
        # Download and standardize the image to 1080x1920 pixels
        return download_and_standardize_image(image_url, output_path)
            
    except Exception as e:
        logging.error(f"Error downloading and standardizing image from {image_url}: {e}")
        # Clean up any partial files
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return False

def search_pexels_fallback(keywords: list, num_images: int, output_dir: str):
    """
    Fallback function to search Pexels if Unsplash fails.
    Note: Requires PEXELS_API_KEY in environment.
    """
    pexels_api_key = os.getenv('PEXELS_API_KEY')
    if not pexels_api_key:
        logging.info("PEXELS_API_KEY not found, skipping Pexels fallback")
        return []
    
    logging.info("Attempting Pexels fallback search...")
    
    downloaded_images = []
    
    try:
        session = http_retry_session()
        headers = {'Authorization': pexels_api_key}
        
        for i, keyword in enumerate(keywords):
            if len(downloaded_images) >= num_images:
                break
                
            pexels_url = "https://api.pexels.com/v1/search"
            params = {
                'query': keyword,
                'per_page': min(5, num_images - len(downloaded_images)),
                'orientation': 'portrait'
            }
            
            try:
                response = session.get(pexels_url, headers=headers, params=params, timeout=15)
                response.raise_for_status()
                
                data = response.json()
                photos = data.get('photos', [])
                
                for j, photo in enumerate(photos):
                    if len(downloaded_images) >= num_images:
                        break
                    
                    image_url = photo['src']['large']
                    image_id = photo['id']
                    
                    image_filename = f"pexels_{i:02d}_{j:02d}_{image_id}.jpg"
                    image_path = os.path.join(output_dir, image_filename)
                    
                    if _download_image(session, image_url, image_path):
                        downloaded_images.append(image_path)
                        logging.info(f"    > Downloaded from Pexels: {image_filename}")
                        
            except requests.exceptions.RequestException as e:
                logging.error(f"Error searching Pexels for '{keyword}': {e}")
                continue
        
        return downloaded_images
        
    except Exception as e:
        logging.error(f"Error in Pexels fallback search: {e}")
        return [] 

def _save_url_metadata(image_path: str, source_url: str):
    """
    Save source URL metadata alongside the image.
    """
    try:
        metadata_path = image_path + '.url'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(source_url)
    except Exception as e:
        logging.warning(f"Could not save URL metadata for {image_path}: {e}") 