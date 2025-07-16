import os
import logging
import requests
from urllib.parse import quote
from bs4 import BeautifulSoup
from agent.utils import http_retry_session
from typing import List, Optional
import json
from agent.media_utils import standardize_image

def _download_searxng_image(session: requests.Session, image_url: str, filepath: str) -> bool:
    """
    Download an image from SearXNG search results.
    
    Args:
        session: HTTP session for downloading
        image_url: URL of the image to download
        filepath: Local path to save the image
        
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.google.com/'
        }
        
        response = session.get(image_url, headers=headers, timeout=10, stream=True)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            logging.warning(f"Invalid content type for image: {content_type}")
            return False
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            # Standardize the image to portrait
            if standardize_image(filepath):
                return True
            else:
                os.remove(filepath)
                return False
        else:
            if os.path.exists(filepath):
                os.remove(filepath)
            return False
            
    except Exception as e:
        logging.debug(f"Failed to download image from {image_url}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def _save_url_metadata(filepath: str, image_url: str) -> None:
    """
    Save metadata about the downloaded image for debugging purposes.
    
    Args:
        filepath: Path to the downloaded image
        image_url: Original URL of the image
    """
    try:
        metadata_path = filepath + '.meta.json'
        import time
        metadata = {
            'source_url': image_url,
            'source': 'searxng',
            'download_timestamp': time.time()
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
            
    except Exception as e:
        # Don't fail the whole download if metadata saving fails
        logging.debug(f"Failed to save metadata for {filepath}: {e}")

def search_google_images(
    search_terms: List[str], 
    num_images: int, 
    download_dir: str, 
    context_keywords: Optional[List[str]] = None
) -> List[str]:
    """
    Main entry point for Google image search. 
    First tries SearXNG if available, falls back to web scraping.
    """
    # Try SearXNG first (no rate limits, free)
    searxng_results = search_with_searxng(search_terms, num_images, download_dir, context_keywords=context_keywords)
    if searxng_results:
        return searxng_results
    
    # CRASH if SearXNG fails - no fallback to web scraping
    error_msg = "SearXNG image search failed. Pipeline configured to crash instead of using unreliable web scraping fallback."
    logging.error(error_msg)
    raise Exception(error_msg)

def run(keywords: list, num_images: int, output_dir: str):
    """
    Searches for images on Google Images and downloads them.
    Note: This uses web scraping which may be against Google's ToS.
    Consider using Google Custom Search API for production use.
    
    Args:
        keywords (list): List of search keywords
        num_images (int): Number of images needed
        output_dir (str): Directory to save downloaded images
    Returns:
        list: List of downloaded image paths
    """
    downloaded_images = []
    
    # User agent to avoid blocking
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        session = http_retry_session()
        images_per_keyword = max(1, num_images // len(keywords))
        
        for i, keyword in enumerate(keywords):
            if len(downloaded_images) >= num_images:
                break
                
            logging.info(f"  > Searching Google Images for: '{keyword}'")
            
            # Construct Google Images search URL with portrait bias
            search_url = f"https://www.google.com/search?q={quote(keyword + ' portrait vertical')}&tbm=isch&hl=en&gl=us&safe=active&tbs=iar:t"
            
            try:
                response = session.get(search_url, headers=headers, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find image URLs in the page
                img_urls = []
                
                # Look for image elements
                for img in soup.find_all('img'):
                    src = img.get('src') or img.get('data-src')
                    if src and src.startswith('http') and 'gstatic.com' not in src:
                        img_urls.append(src)
                
                # Also look for data URLs in script tags (Google often loads images dynamically)
                import re
                for script in soup.find_all('script'):
                    if script.string:
                        # Look for base64 images or direct URLs in JavaScript
                        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+\.(?:jpg|jpeg|png|gif|webp)', script.string)
                        img_urls.extend(urls[:5])  # Limit to avoid too many irrelevant URLs
                
                # Remove duplicates
                img_urls = list(set(img_urls))[:images_per_keyword + 2]
                
                if not img_urls:
                    logging.warning(f"  > No images found for keyword: '{keyword}'")
                    continue
                
                # Download images
                keyword_downloads = 0
                for j, img_url in enumerate(img_urls):
                    if len(downloaded_images) >= num_images or keyword_downloads >= images_per_keyword:
                        break
                    
                    try:
                        # Download the image
                        img_response = session.get(img_url, headers=headers, timeout=10, stream=True)
                        img_response.raise_for_status()
                        
                        # Determine file extension
                        content_type = img_response.headers.get('content-type', 'image/jpeg')
                        ext = 'jpg'
                        if 'png' in content_type:
                            ext = 'png'
                        elif 'gif' in content_type:
                            ext = 'gif'
                        elif 'webp' in content_type:
                            ext = 'webp'
                        
                        # Save image
                        image_filename = f"google_{i:02d}_{keyword_downloads:02d}_{keyword[:20].replace(' ', '_')}.{ext}"
                        image_path = os.path.join(output_dir, image_filename)
                        
                        with open(image_path, 'wb') as f:
                            for chunk in img_response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        # Verify file was created
                        if os.path.exists(image_path) and os.path.getsize(image_path) > 1000:  # At least 1KB
                            downloaded_images.append(image_path)
                            keyword_downloads += 1
                            logging.info(f"    > Downloaded: {image_filename}")
                        else:
                            if os.path.exists(image_path):
                                os.remove(image_path)
                                
                    except Exception as e:
                        logging.debug(f"Failed to download image: {e}")
                        continue
                        
            except requests.exceptions.RequestException as e:
                logging.error(f"Error searching Google Images for '{keyword}': {e}")
                continue
        
        logging.info(f"  > Downloaded {len(downloaded_images)} images from Google Images")
        return downloaded_images
        
    except Exception as e:
        logging.error(f"Error in Google Images search: {e}")
        return []

def search_with_searxng(
    search_terms: List[str], 
    num_images: int, 
    download_dir: str, 
    context_keywords: Optional[List[str]] = None
) -> List[str]:
    """
    Search for images using a public SearXNG instance with enriched queries.
    
    Args:
        search_terms: List of primary search terms.
        num_images: Max number of images to download.
        download_dir: Directory to save images.
        context_keywords: Optional list of keywords to enrich the query.
        
    Returns:
        List of downloaded image file paths.
    """
    searxng_url = os.getenv('SEARXNG_URL')
    if not searxng_url:
        logging.warning("SEARXNG_URL not found. Skipping SearXNG search.")
        return []
    
    session = http_retry_session()
    downloaded_images = []
    
    for term in search_terms:
        if len(downloaded_images) >= num_images:
            break
            
        # Enrich the query with context keywords
        enriched_query = f'"{term}"' # Prioritize exact term
        if context_keywords:
            enriched_query += " " + " ".join(context_keywords)
            
        params = {
            'q': enriched_query,
            'categories': 'images',
            'format': 'json',
            'safesearch': 1, # Enable safe search
            'engines': 'google images', # Exclusively search Google Images
            'tbs': 'iar:t' # Filter for tall/portrait images
        }
        
        try:
            logging.info(f"Searching SearXNG for: '{enriched_query}'")
            response = session.get(searxng_url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json().get('results', [])
            
            if not results:
                logging.warning(f"No SearXNG results for: '{term}'")
                continue
            
            # Download the top N images
            for i, item in enumerate(results[:num_images]):
                if len(downloaded_images) >= num_images:
                    break
                
                image_url = item.get('img_src')
                if not image_url:
                    continue
                
                # Create a safe filename
                safe_term = "".join(c for c in term if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
                filename = f"searxng_{i:02d}_{safe_term}.jpg"
                filepath = os.path.join(download_dir, filename)
                
                if _download_searxng_image(session, image_url, filepath):
                    _save_url_metadata(filepath, image_url)
                    downloaded_images.append(filepath)
                    logging.info(f"  > Downloaded SearXNG image: {filename}")

        except Exception as e:
            logging.error(f"SearXNG search failed for '{term}': {e}")
            
    return downloaded_images

def search_with_custom_api(keywords: list, num_images: int, output_dir: str):
    """
    DEPRECATED: Implementation using Google Custom Search API.
    This function is kept for backward compatibility but will return empty results.
    Use SearXNG instead for unlimited searches.
    """
    logging.warning("Google Custom Search API is deprecated due to quota limits. Using SearXNG instead.")
    return []