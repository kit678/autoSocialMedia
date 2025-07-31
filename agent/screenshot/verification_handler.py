"""
Verification handler for bypassing Cloudflare and CAPTCHA challenges.
Uses cloudscraper for Cloudflare IUAM and python-anticaptcha for CAPTCHA solving.
"""
import os
import time
import logging
import cloudscraper
from typing import Optional, Dict, Any
from python_anticaptcha import AnticaptchaClient, NoCaptchaTaskProxylessTask
import re
try:
    import undetected_chromedriver as uc
    UC_AVAILABLE = True
except ImportError:
    UC_AVAILABLE = False
    logging.warning("undetected-chromedriver not available - some bypass methods disabled")

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Initialize cloudscraper instance
scraper = cloudscraper.create_scraper()

class VerificationHandler:
    """Handles various verification challenges including Cloudflare and CAPTCHAs."""
    
    def __init__(self, captcha_api_key: Optional[str] = None):
        """
        Initialize the verification handler.
        
        Args:
            captcha_api_key: Ignored now, as we're using Buster
        """
        self.captcha_api_key = None  # No longer needed
        logging.info("VerificationHandler initialized with Buster support")
    
    def get_cloudflare_tokens_with_uc(self, url: str) -> Optional[Dict[str, str]]:
        """
        Use undetected-chromedriver as a fallback to get Cloudflare tokens.
        
        Args:
            url: The URL to get tokens for
            
        Returns:
            Dict containing cookies if successful, None otherwise
        """
        if not UC_AVAILABLE:
            return None
            
        try:
            logging.info(f"  > Attempting undetected-chromedriver for: {url}")
            
            # Configure undetected Chrome
            options = uc.ChromeOptions()
            # DO NOT use headless mode - it's easily detected by Cloudflare
            # options.add_argument('--headless')  # DISABLED - detectable
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            driver = uc.Chrome(options=options, version_main=None)
            try:
                # Set window size to avoid detection
                driver.set_window_size(1920, 1080)
                
                # Navigate to URL
                driver.get(url)
                
                # Wait longer for Cloudflare challenge to resolve
                time.sleep(10)
                
                # Check if still on Cloudflare page
                if "checking your browser" in driver.page_source.lower():
                    # Wait more if still verifying
                    time.sleep(5)
                
                # Extract cookies
                cookies = {}
                for cookie in driver.get_cookies():
                    if cookie['name'] in ['cf_clearance', '__cf_bm', 'cf_chl_prog']:
                        cookies[cookie['name']] = cookie['value']
                
                if cookies:
                    logging.info(f"  > Successfully obtained {len(cookies)} tokens via undetected-chromedriver")
                    return cookies
                else:
                    return None
            finally:
                # Properly close the driver to avoid handle errors
                try:
                    driver.quit()
                except Exception:
                    # Silently ignore any errors during cleanup
                    pass
                
        except Exception as e:
            logging.error(f"  > Failed with undetected-chromedriver: {e}")
            return None
    
    def get_cloudflare_tokens(self, url: str) -> Optional[Dict[str, str]]:
        """
        Use cloudscraper to get Cloudflare clearance tokens.
        
        Args:
            url: The URL to get tokens for
            
        Returns:
            Dict containing cookies if successful, None otherwise
        """
        try:
            logging.info(f"  > Attempting to get Cloudflare tokens for: {url}")
            
            # Make request with cloudscraper - reduced timeout for faster failure
            response = scraper.get(url, timeout=10)
            
            # Extract relevant cookies
            cookies = {}
            for cookie in response.cookies:
                if cookie.name in ['cf_clearance', '__cf_bm', 'cf_chl_prog']:
                    cookies[cookie.name] = cookie.value
                    
            if cookies:
                logging.info(f"  > Successfully obtained {len(cookies)} Cloudflare tokens")
                return cookies
            else:
                logging.debug("  > No Cloudflare tokens found in response")
                return None
                
        except Exception as e:
            logging.debug(f"  > Cloudscraper failed (expected for most sites): {e}")
            return None
    
    def solve_recaptcha_v2(self, page, site_key: str, page_url: str) -> Optional[str]:
        """
        Solve reCAPTCHA v2 using undetected-chromedriver with Buster extension.
        
        Args:
            page: Playwright page object (unused in this implementation)
            site_key: The reCAPTCHA site key
            page_url: The URL of the page with CAPTCHA
            
        Returns:
            Solution token if successful, None otherwise
        """
        if not UC_AVAILABLE:
            logging.error("  > Cannot solve CAPTCHA - undetected-chromedriver not available")
            return None
            
        try:
            logging.info("  > Solving reCAPTCHA v2 with Buster...")
            
            # Configure undetected Chrome with Buster
            options = uc.ChromeOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            # Load Buster extension
            buster_path = os.path.join(os.getcwd(), 'buster.crx')
            if os.path.exists(buster_path):
                options.add_extension(buster_path)
            else:
                logging.error("  > Buster CRX file not found")
                return None
            
            driver = uc.Chrome(options=options)
            try:
                driver.get(page_url)
                time.sleep(5)
                
                # Switch to reCAPTCHA iframe
                WebDriverWait(driver, 10).until(EC.frame_to_be_available_and_switch_to_it((By.CSS_SELECTOR, "iframe[src^='https://www.google.com/recaptcha/api2/anchor']")))
                
                # Click checkbox to activate challenge
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//span[@id='recaptcha-anchor']"))).click()
                driver.switch_to.default_content()
                
                # Switch to audio challenge iframe
                WebDriverWait(driver, 10).until(EC.frame_to_be_available_and_switch_to_it((By.CSS_SELECTOR, "iframe[title='recaptcha challenge expires in two minutes']")))
                
                # Click Buster button
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "solver-button"))).click()
                time.sleep(10)  # Wait for solving
                
                # Extract token
                token = driver.execute_script("return document.getElementById('g-recaptcha-response').value;")
                
                if token:
                    logging.info("  > reCAPTCHA solved successfully with Buster")
                    return token
                else:
                    logging.error("  > Failed to get CAPTCHA solution")
                    return None
            finally:
                # Properly close the driver to avoid handle errors
                try:
                    driver.quit()
                except Exception:
                    # Silently ignore any errors during cleanup
                    pass
                
        except Exception as e:
            logging.error(f"  > Error solving reCAPTCHA with Buster: {e}")
            return None
    
    def detect_verification_type(self, page) -> str:
        """
        Detect the type of verification challenge present on the page.
        
        Args:
            page: Playwright page object
            
        Returns:
            Type of verification: 'cloudflare', 'recaptcha', 'hcaptcha', 'none'
        """
        try:
            page_content = page.content().lower()
            
            # Check for Cloudflare
            if any(indicator in page_content for indicator in [
                'checking your browser', 'cloudflare', 'cf-browser-verification'
            ]):
                return 'cloudflare'
            
            # Check for reCAPTCHA
            if page.locator('iframe[src*="recaptcha"]').count() > 0:
                return 'recaptcha'
                
            # Check for hCaptcha
            if page.locator('iframe[src*="hcaptcha"]').count() > 0:
                return 'hcaptcha'
                
            return 'none'
            
        except Exception as e:
            logging.error(f"  > Error detecting verification type: {e}")
            return 'none'
    
    def extract_recaptcha_sitekey(self, page) -> Optional[str]:
        """Extract reCAPTCHA site key from the page."""
        try:
            # Method 1: From iframe src
            iframe = page.locator('iframe[src*="recaptcha"]').first
            if iframe:
                src = iframe.get_attribute('src')
                match = re.search(r'[?&]k=([^&]+)', src)
                if match:
                    return match.group(1)
            
            # Method 2: From div attribute
            div = page.locator('div[data-sitekey]').first
            if div:
                return div.get_attribute('data-sitekey')
                
            # Method 3: From g-recaptcha class
            grecaptcha = page.locator('.g-recaptcha[data-sitekey]').first
            if grecaptcha:
                return grecaptcha.get_attribute('data-sitekey')
                
        except Exception as e:
            logging.error(f"  > Error extracting site key: {e}")
            
        return None
    
    def inject_recaptcha_solution(self, page, solution: str) -> bool:
        """Inject reCAPTCHA solution and trigger callback."""
        try:
            # Inject the solution
            page.evaluate(f'''
                document.getElementById('g-recaptcha-response').value = '{solution}';
                if (window.___grecaptcha_cfg && window.___grecaptcha_cfg.clients[0]) {{
                    window.___grecaptcha_cfg.clients[0].L.L.callback('{solution}');
                }}
            ''')
            
            # Also try alternative callback methods
            page.evaluate(f'''
                if (typeof grecaptcha !== 'undefined' && grecaptcha.execute) {{
                    grecaptcha.execute();
                }}
                if (typeof captchaCallback !== 'undefined') {{
                    captchaCallback('{solution}');
                }}
            ''')
            
            logging.info("  > Injected reCAPTCHA solution")
            return True
            
        except Exception as e:
            logging.error(f"  > Error injecting solution: {e}")
            return False
    
    def handle_verification(self, page, url: str, max_attempts: int = 3) -> bool:
        """
        Main method to handle any verification challenge.
        
        Args:
            page: Playwright page object
            url: The URL being accessed
            max_attempts: Maximum number of attempts
            
        Returns:
            True if verification passed, False otherwise
        """
        for attempt in range(max_attempts):
            verification_type = self.detect_verification_type(page)
            
            if verification_type == 'none':
                logging.info("  > No verification detected")
                return True
                
            logging.info(f"  > Detected {verification_type} verification (Attempt {attempt + 1}/{max_attempts})")
            
            if verification_type == 'cloudflare':
                # Try to get tokens from cloudscraper
                tokens = self.get_cloudflare_tokens(url)
                if not tokens:
                    # Fallback to undetected-chromedriver
                    tokens = self.get_cloudflare_tokens_with_uc(url)
                
                if tokens:
                    # Inject cookies into Playwright
                    for name, value in tokens.items():
                        page.context.add_cookies([{
                            'name': name,
                            'value': value,
                            'domain': page.url.split('/')[2],
                            'path': '/'
                        }])
                    
                    # Reload page with tokens
                    page.reload()
                    time.sleep(3)
                    
                    # Check if cleared
                    if self.detect_verification_type(page) == 'none':
                        logging.info("  > Cloudflare verification bypassed with tokens")
                        return True
                        
            elif verification_type == 'recaptcha':
                site_key = self.extract_recaptcha_sitekey(page)
                if site_key:
                    solution = self.solve_recaptcha_v2(page, site_key, url)
                    if solution:
                        if self.inject_recaptcha_solution(page, solution):
                            time.sleep(3)
                            
                            # Check if cleared
                            if self.detect_verification_type(page) == 'none':
                                logging.info("  > reCAPTCHA verification bypassed")
                                return True
                else:
                    logging.error("  > Could not extract reCAPTCHA site key")
                    
            # Wait before retry
            time.sleep(5)
            
        logging.error(f"  > Failed to bypass {verification_type} after {max_attempts} attempts")
        return False


# Global instance for easy access
_verification_handler = None

def get_verification_handler() -> VerificationHandler:
    """Get or create the global verification handler instance."""
    global _verification_handler
    if _verification_handler is None:
        _verification_handler = VerificationHandler()
    return _verification_handler
