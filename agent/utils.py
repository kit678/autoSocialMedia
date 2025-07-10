import requests
import logging
import subprocess
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import os
import time
from functools import wraps
from typing import Dict, Any

# Import for Gemini client
try:
    import google.genai as genai
except ImportError:
    genai = None

# Import for image processing
try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

# Enhanced rate limiting for multiple APIs
_api_call_history = {}
_rate_limits = {
    'gemini': {
        'calls_per_minute': 13, 
        'current_calls': 0, 
        'reset_time': 0,
        'min_interval': 4.5,
        'backoff_multiplier': 1.0
    },
    'google_cse': {
        'calls_per_day': 100, 
        'current_calls': 0, 
        'reset_time': 0,
        'min_interval': 1.0,
        'backoff_multiplier': 1.0
    },
    'pexels': {
        'calls_per_hour': 200, 
        'current_calls': 0, 
        'reset_time': 0,
        'min_interval': 0.5,
        'backoff_multiplier': 1.0
    }
}

# Function calling schema for visual validation
VALIDATION_FUNCTION_SCHEMA = {
    "name": "validate_visual_relevance",
    "description": "Validate if a visual matches the expected concept for video content",
    "parameters": {
        "type": "object",
        "properties": {
            "is_relevant": {
                "type": "boolean", 
                "description": "Whether the visual is relevant to the expected concept"
            },
            "confidence": {
                "type": "number", 
                "minimum": 0, 
                "maximum": 1, 
                "description": "Confidence score from 0.0 to 1.0"
            },
            "description": {
                "type": "string", 
                "description": "Brief description of what you see in the visual"
            },
            "issues": {
                "type": "array", 
                "items": {"type": "string"}, 
                "description": "List of any problems or issues with the visual"
            },
            "suggestions": {
                "type": "array", 
                "items": {"type": "string"}, 
                "description": "Suggestions for better search terms if needed"
            }
        },
        "required": ["is_relevant", "confidence", "description"]
    }
}

def smart_rate_limit(api_name: str):
    """Enhanced rate limiting with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check and enforce rate limit
            wait_time = calculate_smart_wait_time(api_name)
            if wait_time > 0:
                logging.info(f"Rate limiting: waiting {wait_time:.1f}s for {api_name}")
                time.sleep(wait_time)
            
            # Record call attempt
            record_api_call(api_name)
            
            try:
                result = func(*args, **kwargs)
                
                # Reset backoff on success
                if api_name in _rate_limits:
                    _rate_limits[api_name]['backoff_multiplier'] = 1.0
                
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Handle rate limit errors with exponential backoff
                if any(indicator in error_str for indicator in ['429', 'rate limit', 'quota exceeded', 'too many requests']):
                    backoff_time = get_exponential_backoff(api_name)
                    logging.warning(f"Rate limit hit for {api_name}, backing off {backoff_time:.1f}s")
                    time.sleep(backoff_time)
                    
                    # Retry once after backoff
                    try:
                        record_api_call(api_name)  # Record retry attempt
                        return func(*args, **kwargs)
                    except Exception as retry_e:
                        logging.error(f"Retry after backoff failed for {api_name}: {retry_e}")
                        raise retry_e
                else:
                    # Non-rate-limit error - re-raise immediately
                    raise
                    
        return wrapper
    return decorator

def calculate_smart_wait_time(api_name: str) -> float:
    """Calculate intelligent wait time based on API usage patterns"""
    if api_name not in _rate_limits:
        return 0.0
    
    limits = _rate_limits[api_name]
    now = time.time()
    
    # Reset counters if time window has passed
    if 'calls_per_minute' in limits and now - limits['reset_time'] > 60:
        limits['current_calls'] = 0
        limits['reset_time'] = now
    elif 'calls_per_hour' in limits and now - limits['reset_time'] > 3600:
        limits['current_calls'] = 0
        limits['reset_time'] = now
    elif 'calls_per_day' in limits and now - limits['reset_time'] > 86400:
        limits['current_calls'] = 0
        limits['reset_time'] = now
    
    # Check if we're approaching limits
    if 'calls_per_minute' in limits:
        max_calls = limits['calls_per_minute']
        if limits['current_calls'] >= max_calls:
            # Need to wait until next minute
            return max(0, 60 - (now - limits['reset_time']))
    
    # Apply minimum interval with backoff
    base_interval = limits['min_interval']
    backoff_multiplier = limits.get('backoff_multiplier', 1.0)
    
    return base_interval * backoff_multiplier

def get_exponential_backoff(api_name: str) -> float:
    """Calculate exponential backoff time"""
    if api_name not in _rate_limits:
        return 5.0  # Default backoff
    
    limits = _rate_limits[api_name]
    current_multiplier = limits.get('backoff_multiplier', 1.0)
    
    # Exponential backoff: 2x each time, max 60 seconds
    new_multiplier = min(current_multiplier * 2, 12.0)  # Max 12x base interval
    limits['backoff_multiplier'] = new_multiplier
    
    base_interval = limits['min_interval']
    backoff_time = base_interval * new_multiplier
    
    return min(backoff_time, 60.0)  # Cap at 60 seconds

def record_api_call(api_name: str):
    """Record an API call for rate limiting"""
    if api_name not in _rate_limits:
        return
    
    now = time.time()
    limits = _rate_limits[api_name]
    
    # Initialize reset time if needed
    if limits['reset_time'] == 0:
        limits['reset_time'] = now
    
    # Increment call counter
    limits['current_calls'] += 1
    
    # Store in history for debugging
    if api_name not in _api_call_history:
        _api_call_history[api_name] = []
    
    _api_call_history[api_name].append(now)
    
    # Keep only recent history (last hour)
    _api_call_history[api_name] = [
        call_time for call_time in _api_call_history[api_name] 
        if now - call_time < 3600
    ]

def get_rate_limit_status() -> Dict[str, Any]:
    """Get current rate limit status for all APIs"""
    status = {}
    now = time.time()
    
    for api_name, limits in _rate_limits.items():
        recent_calls = len(_api_call_history.get(api_name, []))
        
        status[api_name] = {
            'current_calls': limits['current_calls'],
            'recent_calls_hour': recent_calls,
            'backoff_multiplier': limits.get('backoff_multiplier', 1.0),
            'next_reset': limits['reset_time'] + (60 if 'calls_per_minute' in limits else 3600),
            'time_to_reset': max(0, limits['reset_time'] + (60 if 'calls_per_minute' in limits else 3600) - now)
        }
    
    return status

# Keep the original rate_limit_gemini for backward compatibility
def rate_limit_gemini(func):
    """Decorator to add rate limiting to Gemini API calls"""
    return smart_rate_limit('gemini')(func)

def http_retry_session(
    retries=3,
    backoff_factor=5,
    status_forcelist=(500, 502, 503, 504),
    session=None,
):
    """Creates a requests session with retry logic."""
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def run_command(command: list, timeout: int = None, **kwargs):
    """
    Executes a shell command and returns its output, with improved error handling.
    Args:
        command (list): The command and its arguments as a list of strings.
        timeout (int): Optional timeout in seconds for the command execution.
        **kwargs: Additional arguments for subprocess.Popen (e.g., cwd, env)
    Returns:
        tuple: (success (bool), stdout (str), stderr (str))
    """
    try:
        logging.info(f"Running command: {' '.join(command)}")
        
        # Ensure text=True for string output, allow overriding
        if 'text' not in kwargs:
            kwargs['text'] = True
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs
        )
        
        # Use communicate to get output and wait for completion
        stdout, stderr = process.communicate(timeout=timeout)
        
        # Sanitize stdout and stderr
        stdout = stdout.strip() if stdout else ""
        stderr = stderr.strip() if stderr else ""
        
        if process.returncode != 0:
            logging.error(f"Command failed with return code {process.returncode}")
            if stdout:
                logging.error(f"Stdout: {stdout}")
            if stderr:
                logging.error(f"Stderr: {stderr}")
            return False, stdout, stderr
            
        return True, stdout, stderr
        
    except subprocess.TimeoutExpired:
        logging.error(f"Command timed out after {timeout} seconds: {command[0]}")
        # Terminate the process
        process.terminate()
        try:
            # Wait a bit for graceful termination
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't terminate
            process.kill()
            process.wait()
        return False, "", f"Command timed out after {timeout} seconds"
    except FileNotFoundError:
        logging.error(f"Command not found: {command[0]}. Is it installed and in your PATH?")
        return False, "", f"Command not found: {command[0]}"
    except Exception as e:
        logging.error(f"An exception occurred while running command: {e}")
        # Capture traceback for better debugging
        import traceback
        return False, "", traceback.format_exc()

def get_audio_duration(audio_path: str):
    """
    Gets the duration of an audio file in seconds using FFprobe.
    Args:
        audio_path (str): Path to the audio file
    Returns:
        float: Duration in seconds, or 0.0 on failure
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-show_entries', 'format=duration',
            '-of', 'csv=p=0',
            audio_path
        ]
        
        success, stdout, stderr = run_command(cmd)
        
        if success and stdout.strip():
            return float(stdout.strip())
        else:
            logging.error(f"Failed to get audio duration: {stderr}")
            return 0.0
            
    except Exception as e:
        logging.error(f"Error getting audio duration: {e}")
        return 0.0

def download_file(url: str, output_path: str):
    """
    Downloads a file from URL to local path.
    Args:
        url (str): URL to download from
        output_path (str): Local path to save the file
    Returns:
        bool: True on success, False on failure
    """
    try:
        session = http_retry_session()
        response = session.get(url, timeout=60, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Verify file was created
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            return False
            
    except Exception as e:
        # Log the specific exception to diagnose download issues
        logging.error(f"Error downloading file from {url}: {e}", exc_info=True)
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return False 

def get_gemini_client():
    """
    Creates and returns a Gemini client instance.
    Returns:
        genai.Client: Configured Gemini client
    Raises:
        RuntimeError: If Gemini is not available or API key is missing
    """
    if genai is None:
        raise RuntimeError("google.genai not available. Install with: pip install google-genai")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment variables")
    
    return genai.Client(api_key=api_key)

def standardize_image_for_video(image_path: str, output_path: str = None) -> str:
    """
    Standardize image to 1080x1920 pixels (9:16 aspect ratio) for portrait videos.
    This ensures all images work with FFmpeg H.264 encoding and consistent video format.
    
    Args:
        image_path (str): Path to the input image
        output_path (str): Path for the standardized image (defaults to overwriting input)
    
    Returns:
        str: Path to the standardized image
    
    Raises:
        RuntimeError: If PIL is not available or image processing fails
    """
    if Image is None:
        raise RuntimeError("PIL (Pillow) not available. Install with: pip install Pillow")
    
    if output_path is None:
        output_path = image_path
    
    try:
        # Open and process the image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Target dimensions (9:16 aspect ratio for portrait videos)
            target_width = 1080
            target_height = 1920
            
            # Calculate current aspect ratio
            current_width, current_height = img.size
            current_aspect = current_width / current_height
            target_aspect = target_width / target_height
            
            # Resize to fit within target dimensions while maintaining aspect ratio
            if current_aspect > target_aspect:
                # Image is wider than target - fit to width
                new_width = target_width
                new_height = int(target_width / current_aspect)
            else:
                # Image is taller than target - fit to height
                new_height = target_height
                new_width = int(target_height * current_aspect)
            
            # Resize the image
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create a new image with exact target dimensions
            final_img = Image.new('RGB', (target_width, target_height), (0, 0, 0))  # Black background
            
            # Calculate position to center the resized image
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            
            # Paste the resized image onto the final canvas
            final_img.paste(img_resized, (x_offset, y_offset))
            
            # Save the standardized image
            # Use high quality JPEG to balance file size and quality
            final_img.save(output_path, 'JPEG', quality=85, optimize=True)
            
            logging.debug(f"Standardized image: {current_width}x{current_height} -> {target_width}x{target_height}")
            
        return output_path
        
    except Exception as e:
        logging.error(f"Failed to standardize image {image_path}: {e}")
        # If standardization fails, copy original file
        if output_path != image_path:
            try:
                import shutil
                shutil.copy2(image_path, output_path)
            except Exception as copy_e:
                logging.error(f"Failed to copy original image: {copy_e}")
                raise RuntimeError(f"Image standardization failed and could not copy original: {copy_e}")
        raise RuntimeError(f"Image standardization failed: {e}")

def download_and_standardize_image(url: str, output_path: str) -> bool:
    """
    Download an image from URL and standardize it to 1080x1920 pixels.
    Combines download_file and standardize_image_for_video for convenience.
    
    Args:
        url (str): URL to download from
        output_path (str): Local path to save the standardized image
    
    Returns:
        bool: True on success, False on failure
    """
    # Create temporary file for download
    temp_path = output_path + ".tmp"
    
    try:
        # Download the original image
        if not download_file(url, temp_path):
            return False
        
        # Standardize the image
        standardize_image_for_video(temp_path, output_path)
        
        # Remove temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Verify final file exists and has reasonable size
        if os.path.exists(output_path) and os.path.getsize(output_path) > 5000:  # At least 5KB
            return True
        else:
            return False
            
    except Exception as e:
        logging.error(f"Failed to download and standardize image from {url}: {e}")
        
        # Clean up any temporary files
        for path in [temp_path, output_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        
        return False