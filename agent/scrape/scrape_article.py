import requests
import logging
from readability import Document
from agent.utils import http_retry_session
from bs4 import BeautifulSoup

def run(article_url: str):
    """
    Scrapes the main text and all images from the article URL.
    Args:
        article_url (str): The URL of the article to scrape.
    Returns:
        dict: A dictionary with 'text', 'image_urls', and 'html_content', or None on failure.
    """
    if not article_url:
        logging.error("Scrape function called with no URL.")
        return None
        
    # Define a comprehensive set of browser headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'DNT': '1',
    }
        
    try:
        session = http_retry_session()
        response = session.get(article_url, timeout=20, headers=headers)
        response.raise_for_status()
        # Ensure we pass text (str) to Document to avoid bytes decoding issues
        doc = Document(response.text)
        html_summary = doc.summary(html_partial=False)
        # Strip HTML tags for clean text using BeautifulSoup
        soup = BeautifulSoup(html_summary, "html.parser")
        plain_text = soup.get_text(separator="\n")
        
        # Extract ALL images from the article content
        image_urls = []
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                # Handle relative URLs
                if src.startswith('//'):
                    src = 'https:' + src
                elif src.startswith('/'):
                    from urllib.parse import urljoin
                    src = urljoin(article_url, src)
                
                # Skip data URLs and very small tracking pixels
                if not src.startswith('data:') and 'pixel' not in src.lower():
                    image_urls.append(src)
                    logging.info(f"    > Found article image: {src[:80]}...")
        
        # Also check for og:image in the original page
        full_soup = BeautifulSoup(response.text, "html.parser")
        og_image = full_soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            og_url = og_image['content']
            if og_url not in image_urls:
                image_urls.insert(0, og_url)  # Put og:image first
                logging.info(f"    > Found og:image: {og_url[:80]}...")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_image_urls = []
        for url in image_urls:
            if url not in seen:
                seen.add(url)
                unique_image_urls.append(url)
        
        logging.info(f"    > Total unique images found: {len(unique_image_urls)}")
        
        return {
            'text': plain_text, 
            'image_urls': unique_image_urls,
            'html_content': html_summary  # Include the cleaned HTML for analysis
        }
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching article: {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing document from {article_url}: {e}")
        return None 