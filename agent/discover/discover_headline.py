import requests
import logging
from agent.utils import http_retry_session

def detect_paywall(url: str) -> bool:
    """
    Detect if a URL is likely behind a paywall or has aggressive bot detection.
    
    Args:
        url: URL to check
        
    Returns:
        True if paywall detected or bot blocking detected, False otherwise
    """
    try:
        # Known paywall domains
        paywall_domains = [
            'nytimes.com', 'wsj.com', 'ft.com', 'washingtonpost.com',
            'economist.com', 'bloomberg.com', 'reuters.com',
            'forbes.com', 'businessinsider.com', 'theatlantic.com',
            'newyorker.com', 'wired.com', 'medium.com'
        ]

        # Known problematic Cloudflare domains
        cloudflare_domains = [
            'freethink.com',
            # Add more domains here if needed
        ]

        # Quick check for Cloudflare problem domains
        for domain in cloudflare_domains:
            if domain in url.lower():
                logging.info(f"Cloudflare challenge detected and skipped: {url}")
                return True
        
        # Quick domain check first
        for domain in paywall_domains:
            if domain in url.lower():
                logging.info(f"Paywall detected (domain): {url}")
                return True
        
        # Enhanced headers to mimic real browser for bot detection test
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'DNT': '1',
        }
        
        session = http_retry_session()
        
        # Use HEAD request first for performance
        try:
            head_response = session.head(url, timeout=5, allow_redirects=True, headers=headers)
            # Some paywalls redirect to subscription pages
            if 'subscribe' in head_response.url.lower() or 'paywall' in head_response.url.lower():
                logging.info(f"Paywall detected (redirect): {url}")
                return True
        except:
            pass  # HEAD failed, try GET
        
        # Test for bot detection with comprehensive headers
        try:
            response = session.get(url, timeout=15, headers=headers)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            if 'RemoteDisconnected' in str(e) or 'Connection aborted' in str(e):
                logging.info(f"Bot detection detected (connection dropped): {url}")
                return True
            else:
                raise  # Re-raise if it's a different connection error
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logging.info(f"Bot detection detected (403 Forbidden): {url}")
                return True
            else:
                raise  # Re-raise if it's a different HTTP error
        
        content = response.text.lower()
        
        # Check for paywall indicators in content
        paywall_keywords = [
            'subscribe to continue', 'subscription required', 'become a subscriber',
            'paywall', 'premium content', 'members only', 'sign up to read',
            'free trial', 'unlimited access', 'subscriber exclusive',
            'this article is reserved', 'please subscribe', 'create free account'
        ]
        
        for keyword in paywall_keywords:
            if keyword in content:
                logging.info(f"Paywall detected (content): {url} - '{keyword}'")
                return True
                
        # Check for shortened content (potential soft paywall)
        if len(content) < 1000:  # Very short content might be truncated
            if any(word in content for word in ['continue reading', 'read more', 'full article']):
                logging.info(f"Paywall detected (truncated): {url}")
                return True
        
        logging.info(f"No paywall or bot detection found: {url}")
        return False
        
    except requests.exceptions.RequestException as e:
        logging.warning(f"URL validation failed for {url}: {e}")
        # If we can't access it during discovery, we definitely can't scrape it
        return True
    except Exception as e:
        logging.warning(f"Unexpected error in URL validation for {url}: {e}")
        return True

def _try_broader_search():
    """
    Try a broader tech/programming search as fallback when AI-specific search fails.
    """
    queries = [
        "technology",
        "programming",
        "software",
        "computer science"
    ]
    
    for query in queries:
        logging.info(f"Trying broader search with query: {query}")
        url = f"https://hn.algolia.com/api/v1/search_by_date?query={query}&tags=story&hitsPerPage=20"
        
        try:
            session = http_retry_session()
            response = session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['hits']:
                valid_stories = []
                
                for i, hit in enumerate(data['hits']):
                    title = hit.get('title')
                    story_url = hit.get('url')
                    
                    if not (title and story_url):
                        continue
                    
                    # Quick check without full validation to save time
                    if not detect_paywall(story_url):
                        valid_stories.append({'title': title, 'url': story_url})
                        
                        if len(valid_stories) >= 3:
                            break
                
                if valid_stories:
                    primary = valid_stories[0]
                    fallbacks = valid_stories[1:] if len(valid_stories) > 1 else []
                    
                    logging.info(f"Found valid story from broader search: {primary['title']}")
                    return {
                        'primary': primary,
                        'fallbacks': fallbacks
                    }
        except Exception as e:
            logging.warning(f"Broader search failed for query '{query}': {e}")
            continue
    
    # Last resort: return a hardcoded tech story that we know works
    logging.warning("All searches failed, using hardcoded fallback story")
    return {
        'primary': {
            'title': 'The Future of Technology: Open Source AI and Innovation',
            'url': 'https://github.com/trending'  # GitHub trending is usually accessible
        },
        'fallbacks': [
            {
                'title': 'Programming Languages and Tools',
                'url': 'https://stackoverflow.com/questions/tagged/python'
            }
        ]
    }

def run():
    """
    Fetches multiple AI-related stories from HN with valid URLs that are not paywalled.
    Returns the best candidate plus fallback options.
    
    Returns:
        dict: A dictionary with 'primary' (best candidate) and 'fallbacks' (list of alternatives)
    """
    # Try to fetch more stories to increase chances of finding valid ones
    url = "https://hn.algolia.com/api/v1/search_by_date?query=artificial%20intelligence&tags=story&hitsPerPage=30"
    try:
        session = http_retry_session()
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['hits']:
            logging.info(f"Found {len(data['hits'])} stories, checking for paywalls...")
            
            valid_stories = []
            
            # Check all stories and collect valid ones
            for i, hit in enumerate(data['hits']):
                title = hit.get('title')
                story_url = hit.get('url')
                
                if not (title and story_url):  # Skip if missing title or URL
                    continue
                
                logging.info(f"Checking story {i+1}/{len(data['hits'])}: {title}")
                
                # Check for paywall
                if detect_paywall(story_url):
                    logging.info(f"Skipping paywalled story: {title}")
                    continue
                
                # Found valid story
                logging.info(f"Found valid story: {title}")
                valid_stories.append({'title': title, 'url': story_url})
                
                # Stop once we have enough candidates (primary + 2 fallbacks)
                if len(valid_stories) >= 3:
                    break
            
            if valid_stories:
                primary = valid_stories[0]
                fallbacks = valid_stories[1:] if len(valid_stories) > 1 else []
                
                logging.info(f"Selected primary story: {primary['title']}")
                if fallbacks:
                    logging.info(f"Available fallbacks: {len(fallbacks)} stories")
                
                return {
                    'primary': primary,
                    'fallbacks': fallbacks
                }
            else:
                logging.warning("No valid stories found in AI query, trying broader tech search...")
                # Try a broader search as fallback
                return _try_broader_search()
        else:
            logging.warning("No hits found in Algolia HN search, trying broader search...")
            return _try_broader_search()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching headline from Algolia: {e}, trying broader search...")
        return _try_broader_search()
