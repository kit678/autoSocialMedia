import os
import logging
import json
from playwright.sync_api import sync_playwright
import time
from typing import Dict, Any, List, Tuple, Optional
import math
from dotenv import load_dotenv
from agent.utils import run_command
import re
import google.generativeai as genai
import PIL.Image
import random

# Load environment variables from .env file
load_dotenv()

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

def run(url: str, output_path: str, viewport_width: int = 375, viewport_height: int = 812):
    """
    Captures a full-page screenshot of the given URL optimized for mobile portrait.
    
    Args:
        url (str): The URL to capture
        output_path (str): Path to save the screenshot
        viewport_width (int): Browser viewport width (mobile portrait optimized)
        viewport_height (int): Browser viewport height (mobile portrait optimized)
        
    Returns:
        bool: True on success, False on failure
    """
    if not url:
        logging.error("No URL provided for screenshot capture.")
        return False
    
    try:
        with sync_playwright() as p:
            # Launch browser with enhanced anti-detection settings
            browser = _launch_stealth_browser(p)
            page = _setup_stealth_page(browser, viewport_width, viewport_height)
            
            logging.info(f"  > Navigating to: {url}")
            
            # Navigate to the URL
            page.goto(url, wait_until='domcontentloaded')
            
            # Wait for initial load
            time.sleep(3)
            
            # Enhanced verification handling
            verification_cleared = _handle_verification_page(page, max_wait_time=45)
            if not verification_cleared:
                # Still on verification page - this URL is problematic
                logging.error(f"  ✗ Verification page could not be cleared for: {url}")
                raise Exception("VERIFICATION_FAILED")
            
            # Additional wait after verification
            time.sleep(2)
            
            # Try to dismiss common cookie banners/pop-ups
            try:
                # Common cookie banner selectors
                cookie_selectors = [
                    'button[id*="accept"]',
                    'button[class*="accept"]',
                    'button[id*="cookie"]',
                    'button[class*="cookie"]',
                    '[class*="cookie"] button',
                    '[id*="cookie"] button',
                    'button:has-text("Accept")',
                    'button:has-text("OK")',
                    'button:has-text("Got it")',
                    'button:has-text("Agree")',
                ]
                
                for selector in cookie_selectors:
                    try:
                        page.click(selector, timeout=1000)
                        logging.info("  > Dismissed cookie banner")
                        time.sleep(1)
                        break
                    except:
                        continue
            except Exception:
                pass  # No cookie banner found, continue
            
            # Try to wait for network idle, but don't fail if it times out
            try:
                page.wait_for_load_state('networkidle', timeout=15000)  # 15 seconds max
            except Exception:
                logging.info("  > Network idle timeout, proceeding with screenshot")
                pass
            
            # Take full-page screenshot
            logging.info(f"  > Capturing screenshot...")
            page.screenshot(path=output_path, full_page=True, type='png')
            
            browser.close()
            
            # Verify screenshot was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logging.info(f"  > Screenshot saved: {output_path}")
                return True
            else:
                logging.error("Screenshot file was not created or is empty.")
                return False
                
    except Exception as e:
        logging.error(f"Error capturing screenshot: {e}")
        return False

def capture_with_scroll_animation(url: str, output_dir: str, num_frames: int = 5):
    """Capture frames with scroll animation using mobile viewport."""
    logging.info(f"  > Capturing {num_frames} scroll frames with mobile viewport...")
    
    try:
        with sync_playwright() as p:
            browser = _launch_stealth_browser(p)
            page = _setup_stealth_page(browser, 375, 812)
            
            page.goto(url, wait_until='domcontentloaded')
            time.sleep(2)
            
            # Get page height for scroll calculation
            page_height = page.evaluate("document.body.scrollHeight")
            viewport_height = 812
            
            screenshot_paths = []
            
            # Capture frames while scrolling
            for i in range(num_frames):
                # Calculate scroll position
                scroll_position = (page_height - viewport_height) * (i / (num_frames - 1)) if num_frames > 1 else 0
                
                # Scroll to position
                page.evaluate(f"window.scrollTo(0, {scroll_position})")
                time.sleep(0.5)  # Wait for scroll to complete
                
                # Take screenshot
                frame_path = os.path.join(output_dir, f'scroll_frame_{i:02d}.png')
                page.screenshot(path=frame_path, type='png')
                screenshot_paths.append(frame_path)
                
                logging.info(f"  > Captured scroll frame {i+1}/{num_frames}")
            
            browser.close()
            return screenshot_paths
            
    except Exception as e:
        logging.error(f"Error capturing scroll animation: {e}")
        return [] 

def analyze_webpage_layout(url: str) -> Optional[Dict[str, Any]]:
    """Analyzes webpage layout using Gemini Vision for intelligent scroll and zoom."""
    try:
        logging.info("  > Analyzing mobile layout with Gemini Vision...")
        elements = find_main_elements(url)
        
        # Gracefully handle failed layout analysis
        if not elements or not elements.get('main_headline'):
            logging.warning("  > Main headline not found in layout analysis. Cannot perform intelligent scroll/zoom.")
            return None

        headline = elements.get('main_headline')
        image = elements.get('main_image')
        
        # Handle null text values properly
        headline_text = headline.get('text')
        if headline_text is None:
            text_preview = 'N/A'
        else:
            text_preview = headline_text[:100]
        logging.info(f"  > Main headline detected: {text_preview}...")
        
        headline_coords = headline.get('coordinates', {})
        img_coords = image.get('coordinates', {}) if image else {}
        
        # Check for valid coordinates before logging and processing
        if headline_coords and headline_coords.get('y') is not None:
            logging.info(f"    - Headline coordinates: x={headline_coords.get('x')}, y={headline_coords.get('y')}, w={headline_coords.get('width')}, h={headline_coords.get('height')}")
        else:
            logging.warning("  > Headline coordinates are null or invalid")
            return None
            
        if img_coords:
            logging.info(f"    - Image coordinates: x={img_coords.get('x')}, y={img_coords.get('y')}, w={img_coords.get('width')}, h={img_coords.get('height')}")

        # Calculate scroll range with null checks
        scroll_start_y = headline_coords.get('y', 300) if headline_coords.get('y') is not None else 300
        content_end_y = find_end_of_content(url)
        scroll_end_y = max(scroll_start_y + 200, content_end_y) # Ensure at least some scroll room
        
        scroll_range = {
            'start_y': scroll_start_y,
            'end_y': scroll_end_y,
            'total_scroll': scroll_end_y - scroll_start_y
        }
        logging.info(f"    - Scroll range: {scroll_range.get('start_y')} -> {scroll_range.get('end_y')} ({scroll_range.get('total_scroll')}px)")

        return {
            'main_headline': headline,
            'main_image': image,
            'scroll_range': scroll_range
        }
        
    except Exception as e:
        logging.error(f"Error analyzing webpage layout: {e}")
        return None

def create_intelligent_scroll_video(url: str, output_path: str, layout_data: Dict[str, Any], 
                                 duration: float = 10.0, fps: int = 15) -> bool:
    """
    Creates a video with intelligent two-phase scroll sequence using efficient frame capture.
    
    Args:
        url: The URL to capture
        output_path: Path to save the video
        layout_data: Layout analysis from Gemini
        duration: Total video duration
        fps: Frames per second (15fps for efficiency)
        
    Returns:
        bool: True on success, False on failure
    """
    if not layout_data:
        raise Exception("No layout data provided for intelligent scroll - aborting")
    
    browser = None
    temp_dir = None
    
    # Mobile viewport dimensions (iPhone standard)
    viewport_width = 375
    viewport_height = 812
    
    try:
        with sync_playwright() as p:
            browser = _launch_stealth_browser(p)
            page = _setup_stealth_page(browser, viewport_width, viewport_height)
            
            logging.info(f"  > Navigating to URL with mobile viewport ({viewport_width}x{viewport_height})...")
            page.goto(url, wait_until='domcontentloaded')
            time.sleep(3)
            
            # Enhanced verification handling for intelligent capture
            verification_cleared = _handle_verification_page(page, max_wait_time=45)
            if not verification_cleared:
                # Still on verification page - this URL is problematic
                logging.error(f"  ✗ Verification page could not be cleared for intelligent video capture: {url}")
                raise Exception("VERIFICATION_FAILED")
            
            # Additional wait after verification
            time.sleep(2)
            
            # Try to dismiss cookie banners
            _dismiss_cookie_banners(page)
            
            # Plan scroll sequence based on mobile layout
            scroll_sequence = _plan_scroll_sequence(layout_data, duration, viewport_width, viewport_height)
            
            # Create temporary directory for frames
            temp_dir = output_path.replace('.mp4', '_frames')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Additional popup dismissal before recording
            _dismiss_cookie_banners(page)
            
            # Generate frames
            total_frames = int(duration * fps)
            logging.info(f"  > Capturing {total_frames} frames for mobile-optimized intelligent scroll video...")
            
            frames_captured = 0
            failed_frames = 0
            max_failed_frames = 50
            
            for i in range(total_frames):
                if failed_frames >= max_failed_frames:
                    raise Exception(f"Too many failed frames ({failed_frames}), aborting")
                
                # Safety check for maximum frames
                if frames_captured >= 600:  # Max 20 seconds at 30fps
                    logging.warning("  > Reached maximum frame limit, stopping capture")
                    break
                
                # Dismiss popups periodically during recording (every 3 seconds)
                if i % 45 == 0:  # Every 3 seconds at 15fps
                    try:
                        _dismiss_cookie_banners(page)
                    except:
                        pass
                
                try:
                    # Calculate current time in video
                    current_time = (i / (total_frames - 1)) * duration if total_frames > 1 else 0
                    
                    # Time bounds checking
                    if current_time > duration:
                        break
                    
                    # Get scroll state for this frame
                    scroll_state = _get_scroll_state_at_time(scroll_sequence, current_time)
                    
                    # Apply scroll and positioning
                    _apply_scroll_state_and_wait(page, scroll_state, viewport_width, viewport_height)
                    
                    # Capture frame
                    frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
                    page.screenshot(path=frame_path, type='png', timeout=10000)
                    frames_captured += 1
                    
                    # Progress logging
                    if i % 30 == 0:
                        segment_info = next((seg for seg in scroll_sequence if seg['start_time'] <= current_time <= seg['end_time']), {})
                        description = segment_info.get('description', 'Unknown')
                        logging.info(f"    > Frame {i+1}/{total_frames} @ {current_time:.1f}s: {description}")
                        
                except Exception as frame_error:
                    failed_frames += 1
                    logging.warning(f"    > Failed to capture frame {i}: {frame_error}")
                    
                    # Try to recover by resetting page state
                    try:
                        page.evaluate("window.scrollTo(0, 0)", timeout=3000)
                        time.sleep(0.1)
                    except:
                        pass
                    continue
            
            if frames_captured == 0:
                raise Exception("No frames were captured successfully")
            
            logging.info(f"  > Successfully captured {frames_captured} frames")
            
            # Convert frames to final 1080x1920 video
            success = _convert_frames_to_mobile_video(temp_dir, output_path, fps)
            
            if success:
                logging.info(f"  > Intelligent scroll video completed: {output_path}")
                return True
            else:
                raise Exception("Video conversion failed")
            
    except Exception as e:
        logging.error(f"Error creating intelligent scroll video: {e}")
        raise
        
    finally:
        if browser:
            try:
                browser.close()
            except:
                pass
        
        # Clean up frames directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logging.info("  > Cleaned up temporary frame files")
            except Exception as cleanup_error:
                logging.warning(f"  > Failed to cleanup temp directory: {cleanup_error}")

def _dismiss_cookie_banners(page):
    """Dismiss common cookie banners and popup overlays."""
    popup_selectors = [
        # Cookie acceptance buttons
        'button[id*="accept"]',
        'button[class*="accept"]',
        'button[id*="cookie"]',
        'button[class*="cookie"]',
        '[class*="cookie"] button',
        '[id*="cookie"] button',
        'button:has-text("Accept")',
        'button:has-text("OK")',
        'button:has-text("Got it")',
        'button:has-text("Agree")',
        
        # Close buttons and dismiss options
        'button:has-text("Close")',
        'button:has-text("Dismiss")',
        'button:has-text("×")',
        'button[aria-label*="close"]',
        'button[aria-label*="dismiss"]',
        '[class*="close"] button',
        '[class*="dismiss"] button',
        
        # Membership and subscription popups (specific to Vox-style banners)
        'button:has-text("No thanks")',
        'button:has-text("Maybe later")',
        'button:has-text("Skip")',
        'button:has-text("Not now")',
        'button[class*="decline"]',
        'button[class*="reject"]',
        
        # Bottom banner specific - enhanced for membership prompts
        '[class*="bottom"] button',
        '[class*="footer"] button',
        '[class*="banner"] button',
        '[class*="popup"] button',
        '[class*="modal"] button',
        '[class*="membership"] button',
        '[class*="subscribe"] button',
        '[role="banner"] button',
        
        # X close buttons in corners
        'button[class*="close"]',
        'span[class*="close"]',
        'div[class*="close"]',
        '[aria-label="Close"]',
        '[aria-label="close"]',
        
        # Try any visible button in bottom area that might be close/dismiss
        '[style*="position: fixed"] button',
        '[style*="bottom:"] button'
    ]
    
    dismissed_count = 0
    
    # First pass: try specific close/dismiss buttons
    for selector in popup_selectors:
        try:
            # Check if element exists and is visible
            elements = page.locator(selector)
            if elements.count() > 0:
                for i in range(min(3, elements.count())):  # Try up to 3 matching elements
                    element = elements.nth(i)
                    if element.is_visible(timeout=500):
                        element.click(timeout=2000)
                        logging.info(f"  > Dismissed popup: {selector}")
                        dismissed_count += 1
                        time.sleep(1)  # Wait for popup to disappear
                        break
        except:
            continue
    
    # Second pass: try to click any X or close icon in fixed/bottom positioned elements
    try:
        # Look for X symbols or close icons specifically
        close_symbols = ['×', '✕', '⨯', 'X']
        for symbol in close_symbols:
            try:
                close_button = page.get_by_text(symbol, exact=True).first
                if close_button.is_visible(timeout=500):
                    close_button.click(timeout=2000)
                    logging.info(f"  > Dismissed popup by clicking: {symbol}")
                    dismissed_count += 1
                    time.sleep(1)
                    break
            except:
                continue
    except:
        pass
    
    if dismissed_count > 0:
        # Wait longer for DOM to settle after dismissing popups
        time.sleep(3)
        logging.info(f"  > Dismissed {dismissed_count} popup(s), waiting for page to settle")
    
    # Try to dismiss any remaining overlays by pressing Escape multiple times
    try:
        for _ in range(3):
            page.keyboard.press('Escape')
            time.sleep(0.5)
    except:
        pass

def _plan_scroll_sequence(layout_data: Dict[str, Any], duration: float, 
                       viewport_width: int, viewport_height: int) -> List[Dict[str, Any]]:
    """Plans a two-phase scroll sequence based on mobile layout."""
    
    # Check if we have valid scroll range data
    if 'scroll_range' not in layout_data:
        # Fallback: no scroll data, show static viewport
        return [{
            'start_time': 0.0,
            'end_time': duration,
            'scroll_y': 0,
            'description': 'Static viewport - no scroll range detected'
        }]
    
    scroll_range = layout_data['scroll_range']
    start_y = scroll_range.get('start_y', 0)
    end_y = scroll_range.get('end_y', 400)
    total_scroll = scroll_range.get('total_scroll', end_y - start_y)
    
    # Two-phase scroll sequence:
    # Phase 1: Original webpage view (2 seconds)
    # Phase 2: Scroll to headline with proper padding and STOP there
    
    phase1_duration = 2.0  # Show original page for 2 seconds
    phase2_duration = duration - phase1_duration  # Scrolling phase
    
    # Scroll positions
    scroll_start = 0  # Start at original page top
    
    # Calculate proper scroll target to show headline in focus
    # We want headline to appear in upper third of viewport for better readability
    headline_viewport_target = viewport_height // 4  # Position headline at 25% from top
    headline_target = max(0, start_y - headline_viewport_target)
    
    # Ensure minimum scroll for visible motion (user wants to see scrolling action)
    min_scroll = 200  # Guaranteed minimum scroll distance
    if headline_target < min_scroll:
        headline_target = min_scroll
    
    # Don't over-scroll past the detected end position
    if end_y > start_y:  # Valid scroll range detected
        max_scroll = end_y - viewport_height + 100  # Small buffer
        headline_target = min(headline_target, max_scroll)
    
    # Debug logging for scroll calculation
    logging.info(f"  > SCROLL CALCULATION:")
    logging.info(f"    - Headline Y position: {start_y}px")
    logging.info(f"    - End Y position: {end_y}px") 
    logging.info(f"    - Viewport height: {viewport_height}px")
    logging.info(f"    - Target headline position: {headline_target}px")
    logging.info(f"    - Calculated scroll target: {headline_target}px")
    logging.info(f"    - Minimum scroll enforced: {min_scroll}px")
    
    # Create sequence
    sequence = []
    total_frames = int(duration * 15)  # 15fps
    
    for i in range(total_frames):
        # Calculate time and phase - ensure final frame gets full duration
        if total_frames > 1:
            current_time = (i / (total_frames - 1)) * duration
        else:
            current_time = 0
        
        if current_time <= phase1_duration:
            # Phase 1: Show original webpage (no scroll)
            scroll_position = scroll_start
            description = "Original webpage view"
        else:
            # Phase 2: Scroll FROM start TO headline position (and stop there)
            phase2_time_elapsed = current_time - phase1_duration
            phase2_progress = min(1.0, phase2_time_elapsed / phase2_duration)  # Ensure max 1.0
            
            # Apply easing curve for smooth motion
            eased_progress = 0.5 - 0.5 * math.cos(phase2_progress * math.pi)
            
            # Calculate scroll position FROM scroll_start TO headline_target
            scroll_distance = headline_target - scroll_start
            scroll_position = scroll_start + (scroll_distance * eased_progress)
            description = f"Scroll to headline {phase2_progress:.1%} - y={scroll_position:.0f}px"
        
        sequence.append({
            'start_time': current_time,
            'end_time': current_time + (1/15),  # Each frame duration
            'scroll_y': scroll_position,
            'description': description
        })
    
    # Ensure final frame reaches exact target
    if sequence:
        sequence[-1]['scroll_y'] = headline_target
        sequence[-1]['description'] = f"Final position - y={headline_target:.0f}px"
    
    logging.info(f"  > Two-phase scroll sequence planned:")
    logging.info(f"    - Phase 1: Original view for {phase1_duration:.1f}s (scroll=0)")
    logging.info(f"    - Phase 2: Scroll to headline for {phase2_duration:.1f}s")
    logging.info(f"    - Detected headline position: y={start_y}px")
    logging.info(f"    - Final scroll target: {headline_target:.0f}px (headline at 25% viewport)")
    logging.info(f"    - Total scroll distance: {headline_target:.0f}px")
    logging.info(f"    - Total frames: {total_frames} @ 15fps")
    logging.info(f"    - Final frame will be at: y={headline_target:.0f}px")
    
    return sequence

def _get_important_elements(layout_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the content area element."""
    elements = []
    
    # Add content area
    if 'main_headline' in layout_data and 'main_image' in layout_data and 'scroll_range' in layout_data:
        headline = layout_data['main_headline']
        image = layout_data['main_image']
        scroll_range = layout_data['scroll_range']
        if isinstance(headline, dict) and 'coordinates' in headline:
            elements.append({**headline, 'type': 'main_headline'})
            
            # Log the element that will be processed
            text_preview = headline.get('text', '')[:100]
            priority = headline.get('priority', 'unset')
            coords = headline.get('coordinates', {})
            img_coords = image.get('coordinates', {})
            logging.info(f"  > Processing main headline (priority {priority}): {text_preview}...")
            logging.info(f"    - Headline coordinates: x={coords.get('x')}, y={coords.get('y')}, w={coords.get('width')}, h={coords.get('height')}")
            logging.info(f"    - Image coordinates: x={img_coords.get('x')}, y={img_coords.get('y')}, w={img_coords.get('width')}, h={img_coords.get('height')}")
            logging.info(f"    - Scroll range: {scroll_range.get('start_y')} -> {scroll_range.get('end_y')} ({scroll_range.get('total_scroll')}px)")
    
    return elements

def _get_scroll_state_at_time(scroll_sequence: List[Dict[str, Any]], current_time: float) -> Dict[str, Any]:
    """Get the scroll state at a specific time, with smooth transitions."""
    
    # Find the current sequence segment
    current_segment = None
    for segment in scroll_sequence:
        if segment['start_time'] <= current_time <= segment['end_time']:
            current_segment = segment
            break
    
    if not current_segment:
        # Default to mobile viewport center
        return {
            'scroll_y': 0
        }
    
    # Calculate transition progress within segment
    segment_duration = current_segment['end_time'] - current_segment['start_time']
    if segment_duration > 0:
        progress = (current_time - current_segment['start_time']) / segment_duration
        progress = max(0, min(1, progress))
        
        # Apply easing for smooth transitions
        eased_progress = 0.5 - 0.5 * math.cos(progress * math.pi)
    else:
        eased_progress = 1.0
    
    # If this is a transition segment, interpolate with previous segment
    prev_segment = None
    for i, segment in enumerate(scroll_sequence):
        if segment is current_segment and i > 0:
            prev_segment = scroll_sequence[i-1]
            break
    
    if prev_segment:
        # Interpolate between previous and current segment
        scroll_y = prev_segment['scroll_y'] + (current_segment['scroll_y'] - prev_segment['scroll_y']) * eased_progress
    else:
        # Use current segment values
        scroll_y = current_segment['scroll_y']
    
    return {
        'scroll_y': scroll_y
    }

def _apply_scroll_state_and_wait(page, scroll_state: Dict[str, Any], viewport_width: int, viewport_height: int):
    """Apply scroll state and wait for completion - optimized for mobile frame capture."""
    
    try:
        # Get scroll parameters
        scroll_y = max(0, int(scroll_state['scroll_y']))
        
        # Apply scroll position
        page.evaluate(f"window.scrollTo(0, {scroll_y})")
        
        # Wait for animations and rendering to complete
        time.sleep(0.1)  # Reduced wait time for mobile
        
        # Additional wait for any CSS transitions
        page.wait_for_timeout(50)  # 50ms timeout for mobile optimization
        
    except Exception as e:
        logging.warning(f"Failed to apply scroll state: {e}")
        # Reset to safe state on error
        try:
            page.evaluate("window.scrollTo(0, 0)", timeout=3000)
            time.sleep(0.1)
        except:
            pass

def _convert_frames_to_mobile_video(temp_dir: str, output_path: str, fps: int) -> bool:
    """Convert mobile frame images to 1080x1920 portrait MP4 video."""
    logging.info("  > Converting mobile frames to 1080x1920 portrait video...")
    
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'frame_%04d.png'),
        '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    success, stdout, stderr = run_command(cmd)
    
    if success:
        logging.info(f"  > Portrait video created successfully (1080x1920): {output_path}")
        return True
    else:
        logging.error(f"  > Failed to create portrait video: {stderr}")
        return False

def capture_webpage_video(url: str, output_path: str, duration: float = 10.0, 
                         layout_data: Optional[Dict[str, Any]] = None) -> bool:
    """
    Captures a video of webpage with intelligent two-phase scroll sequence.
    Falls back to basic scroll if layout analysis fails.
    
    Args:
        url: The URL to capture
        output_path: Path to save the video (MP4)
        duration: Total video duration in seconds
        layout_data: Optional pre-computed layout data from Gemini analysis
        
    Returns:
        bool: True on success, False on failure
    """
    logging.info(f"  > Starting intelligent scroll video capture for: {url}")
    
    # Analyze layout with Gemini if not provided
    if not layout_data:
        logging.info("  > No layout data provided, analyzing with Gemini...")
        layout_data = analyze_webpage_layout(url)
        
    if not layout_data:
        logging.warning("  > Gemini analysis failed or returned empty data - falling back to basic scroll")
        # Fall back to basic scroll video without layout intelligence
        return _capture_scroll_video_original(url, output_path, duration)
    
    # Create intelligent scroll video
    try:
        success = create_intelligent_scroll_video(url, output_path, layout_data, duration)
        
        if not success:
            logging.warning("  > Intelligent scroll video creation failed - falling back to basic scroll")
            return _capture_scroll_video_original(url, output_path, duration)
            
        logging.info(f"  > Intelligent scroll video completed: {output_path}")
        return True
    except Exception as e:
        logging.error(f"  > Error in intelligent scroll: {e}")
        logging.info("  > Falling back to basic scroll video")
        return _capture_scroll_video_original(url, output_path, duration)

def _capture_scroll_video_original(url: str, output_path: str, duration: float = 10.0, fps: int = 30):
    """
    Captures a scrolling video of the webpage.
    Args:
        url (str): The URL to capture
        output_path (str): Path to save the video
        duration (float): Duration of scroll in seconds
        fps (int): Frames per second for the video
    Returns:
        bool: True on success, False on failure
    """
    if not url:
        logging.error("No URL provided for scroll video capture.")
        return False
    
    temp_dir = None
    browser = None
    
    try:
        with sync_playwright() as p:
            browser = _launch_stealth_browser(p)
            page = _setup_stealth_page(browser, 375, 812)
            
            logging.info(f"  > Navigating to URL for mobile video capture...")
            page.goto(url, wait_until='networkidle', timeout=60000)
            time.sleep(3)
            
            # Enhanced verification handling for basic video capture
            verification_cleared = _handle_verification_page(page, max_wait_time=45)
            if not verification_cleared:
                # Still on verification page - this URL is problematic
                logging.error(f"  ✗ Verification page could not be cleared for basic video capture: {url}")
                raise Exception("VERIFICATION_FAILED")
            
            # Additional wait after verification
            time.sleep(2)
            
            # Try to dismiss cookie banners
            _dismiss_cookie_banners(page)
            
            # Get page dimensions
            page_height = page.evaluate("document.body.scrollHeight")
            viewport_height = 812
            
            # Calculate scroll parameters
            total_frames = int(duration * fps)
            scroll_distance = page_height - viewport_height
            
            logging.info(f"  > Page height: {page_height}px, scroll distance: {scroll_distance}px")
            
            if scroll_distance <= 0:
                logging.info("  > Page is short, creating static video from screenshot")
                # Just capture a single frame and repeat it
                screenshot_path = output_path.replace('.mp4', '_static.png')
                page.screenshot(path=screenshot_path, type='png')
                
                # Convert to video using FFmpeg
                cmd = [
                    'ffmpeg', '-y',
                    '-loop', '1',
                    '-i', screenshot_path,
                    '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920',
                    '-c:v', 'libx264',
                    '-t', str(duration),
                    '-pix_fmt', 'yuv420p',
                    '-r', str(fps),
                    output_path
                ]
                success, _, stderr = run_command(cmd)
                
                # Clean up temp file
                if os.path.exists(screenshot_path):
                    os.remove(screenshot_path)
                
                browser.close()
                
                if success:
                    logging.info(f"  > Static video created successfully")
                    return True
                else:
                    logging.error(f"  > Failed to create static video: {stderr}")
                    return False
            
            # Capture frames while scrolling
            temp_dir = output_path.replace('.mp4', '_frames')
            os.makedirs(temp_dir, exist_ok=True)
            
            logging.info(f"  > Capturing {total_frames} frames while scrolling...")
            
            frames_captured = 0
            for i in range(total_frames):
                try:
                    # Calculate scroll position (ease-in-out)
                    progress = i / (total_frames - 1) if total_frames > 1 else 0
                    # Apply ease-in-out curve
                    eased_progress = 0.5 - 0.5 * math.cos(progress * math.pi)
                    scroll_position = int(scroll_distance * eased_progress)
                    
                    # Scroll to position
                    page.evaluate(f"window.scrollTo(0, {scroll_position})")
                    time.sleep(0.05)  # Small delay for smooth rendering
                    
                    # Capture frame
                    frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
                    page.screenshot(path=frame_path, type='png')
                    frames_captured += 1
                    
                    if i % 30 == 0:  # Log every 30 frames (roughly every second at 30fps)
                        logging.info(f"    > Captured frame {i+1}/{total_frames}")
                        
                except Exception as frame_error:
                    logging.warning(f"    > Failed to capture frame {i}: {frame_error}")
                    continue
            
            browser.close()
            
            if frames_captured == 0:
                logging.error("  > No frames were captured")
                return False
            
            logging.info(f"  > Successfully captured {frames_captured}/{total_frames} frames")
            
            # Convert frames to video using FFmpeg
            logging.info("  > Converting frames to video...")
            
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%04d.png'),
                '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920',
                '-c:v', 'libx264',
                '-preset', 'fast',  # Changed to fast for better compatibility
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            
            success, stdout, stderr = run_command(cmd)
            
            if success:
                logging.info(f"  > Scroll video created successfully: {output_path}")
                return True
            else:
                logging.error(f"  > Failed to create scroll video: {stderr}")
                return False
                
    except Exception as e:
        logging.error(f"Error capturing scroll video: {e}")
        return False
        
    finally:
        # Clean up frames directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logging.info("  > Cleaned up temporary frame files")
            except Exception as cleanup_error:
                logging.warning(f"  > Failed to cleanup temp directory: {cleanup_error}")

def capture_scroll_video(url: str, output_path: str, duration: float = 10.0) -> bool:
    """
    Captures an intelligent two-phase scroll video of a webpage with mobile-first approach.
    Falls back to basic scroll if intelligent analysis fails.
    
    Args:
        url (str): The URL to capture
        output_path (str): Path to save the video file
        duration (float): Video duration in seconds
        
    Returns:
        bool: True on success, False on failure
    """
    try:
        # Mobile viewport dimensions
        viewport_width = 375
        viewport_height = 812
        
        logging.info(f"Starting intelligent mobile scroll video capture:")
        logging.info(f"  > URL: {url}")
        logging.info(f"  > Output: {output_path}")
        logging.info(f"  > Mobile viewport: {viewport_width}x{viewport_height}")
        logging.info(f"  > Final output: 1080x1920 portrait")
        logging.info(f"  > Duration: {duration:.1f}s")
        
        # Analyze layout with Gemini
        logging.info("  > Analyzing mobile webpage layout with Gemini...")
        layout_data = analyze_webpage_layout(url)
        
        if not layout_data:
            logging.warning("  > No layout data obtained from Gemini analysis - falling back to basic scroll")
            # Fall back to basic scroll video
            return _capture_scroll_video_original(url, output_path, duration)
        
        logging.info(f"  > Layout analysis completed: {len(layout_data)} regions found")
        
        # Create intelligent scroll video using mobile viewport
        success = create_intelligent_scroll_video(url, output_path, layout_data, duration)
        
        if success:
            logging.info(f"✅ Mobile-optimized intelligent scroll video captured successfully: {output_path}")
            return True
        else:
            logging.warning("❌ Failed to create intelligent scroll video - falling back to basic scroll")
            return _capture_scroll_video_original(url, output_path, duration)
            
    except Exception as e:
        logging.error(f"❌ Error in mobile scroll video capture: {e}")
        logging.info("  > Falling back to basic scroll video")
        return _capture_scroll_video_original(url, output_path, duration)

def capture_with_smart_focus(url: str, output_path: str, focus_areas: list = None):
    """
    Captures a screenshot with smart cropping based on focus areas.
    Args:
        url (str): The URL to capture
        output_path (str): Path to save the screenshot
        focus_areas (list): List of CSS selectors or coordinates to focus on
    Returns:
        bool: True on success, False on failure
    """
    if not url:
        logging.error("No URL provided for smart focus capture.")
        return False
    
    try:
        with sync_playwright() as p:
            browser = _launch_stealth_browser(p)
            page = _setup_stealth_page(browser, 375, 812)
            
            page.goto(url, wait_until='domcontentloaded')
            time.sleep(3)
            
            # Take full page screenshot first
            full_screenshot = page.screenshot(full_page=True, type='png')
            
            if focus_areas and len(focus_areas) > 0:
                # Get bounding boxes of focus areas
                for selector in focus_areas:
                    try:
                        element = page.locator(selector).first
                        if element:
                            # Get element position and take focused screenshot
                            box = element.bounding_box()
                            if box:
                                # Add some padding
                                padding = 50
                                clip = {
                                    'x': max(0, box['x'] - padding),
                                    'y': max(0, box['y'] - padding),
                                    'width': box['width'] + 2 * padding,
                                    'height': box['height'] + 2 * padding
                                }
                                
                                # Save focused screenshot
                                focused_path = output_path.replace('.png', f'_focus_{selector.replace(".", "_")}.png')
                                page.screenshot(path=focused_path, clip=clip, type='png')
                                logging.info(f"  > Captured focused area: {selector}")
                    except:
                        continue
            
            # Save full screenshot
            with open(output_path, 'wb') as f:
                f.write(full_screenshot)
            
            browser.close()
            
            logging.info(f"  > Smart focus screenshot saved: {output_path}")
            return True
            
    except Exception as e:
        logging.error(f"Error in smart focus capture: {e}")
        return False 

def _convert_frames_to_video(temp_dir: str, output_path: str, fps: int) -> bool:
    """Convert frame images to MP4 video (original function for scroll video)."""
    logging.info("  > Converting frames to video...")
    from agent.utils import run_command
    
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    success, stdout, stderr = run_command(cmd)
    
    if success:
        logging.info(f"  > Video created successfully: {output_path}")
        return True
    else:
        logging.error(f"  > Failed to create video: {stderr}")
        return False

def find_main_elements(url: str) -> dict:
    """Uses Gemini Vision to find the main headline and image of an article."""
    screenshot_path = "temp_mobile_analysis.png"
    try:
        logging.info("  > Taking screenshot for Gemini analysis...")
        with sync_playwright() as p:
            browser = _launch_stealth_browser(p)
            page = _setup_stealth_page(browser, 375, 812)
            page.goto(url, wait_until='domcontentloaded')
            time.sleep(3)
            
            # Enhanced verification handling for Gemini analysis
            verification_cleared = _handle_verification_page(page, max_wait_time=45)
            if not verification_cleared:
                # Still on verification page - this URL is problematic
                logging.error(f"  ✗ Verification page could not be cleared for Gemini analysis: {url}")
                raise Exception("VERIFICATION_FAILED")
            
            # Additional wait after verification
            time.sleep(2)
            
            _dismiss_cookie_banners(page)
            page.screenshot(path=screenshot_path, type='png', full_page=False)
            browser.close()

        if not GEMINI_API_KEY:
            logging.error("GEMINI_API_KEY environment variable not set")
            raise Exception("GEMINI_API_KEY environment variable not set")
        
        logging.info("  > Calling Gemini Vision API...")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if os.path.exists(screenshot_path):
            # --- FIX: Add retry logic for file opening on Windows ---
            image = None
            for i in range(3): # Retry up to 3 times
                try:
                    image = PIL.Image.open(screenshot_path)
                    break 
                except IOError:
                    logging.warning(f"Could not open screenshot file, retrying in 0.5s... (Attempt {i+1})")
                    time.sleep(0.5)
            
            if not image:
                logging.error(f"Failed to open screenshot file after multiple retries: {screenshot_path}")
                raise Exception(f"Failed to open screenshot file after multiple retries: {screenshot_path}")

            prompt = """Analyze this mobile webpage screenshot. Find the MAIN ARTICLE HEADLINE and the MAIN ARTICLE IMAGE.
Return a JSON object with `main_headline` and `main_image` keys, each containing the `text` and `coordinates` (x, y, width, height).
Focus on the primary editorial content, not ads or site branding. Respond with ONLY the JSON object."""
            
            try:
                logging.info("  > Sending request to Gemini API...")
                response = model.generate_content([prompt, image])
                
                # Log the raw response for debugging
                logging.info(f"  > Gemini API response received. Length: {len(response.text) if response.text else 0} chars")
                logging.info(f"  > Raw Gemini response: {response.text}")
                
                if not response.text:
                    logging.error("  > Gemini API returned empty response")
                    return None
                
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    logging.info(f"  > Found JSON in response: {json_str}")
                    try:
                        parsed_json = json.loads(json_str)
                        logging.info(f"  > Successfully parsed JSON: {parsed_json}")
                        return parsed_json
                    except json.JSONDecodeError as json_error:
                        logging.error(f"  > Failed to parse JSON: {json_error}")
                        logging.error(f"  > Attempted to parse: {json_str}")
                        return None
                else:
                    logging.error("  > No JSON object found in Gemini response")
                    logging.error(f"  > Full response was: {response.text}")
                    return None
                    
            except Exception as gemini_error:
                logging.error(f"  > Gemini API call failed: {gemini_error}")
                logging.error(f"  > Exception type: {type(gemini_error)}")
                raise
            finally:
                # --- FIX: Close image file and delete AFTER processing ---
                if image:
                    image.close()
                if os.path.exists(screenshot_path):
                    os.remove(screenshot_path)
        else:
            logging.error(f"  > Screenshot file not found: {screenshot_path}")
            return None
            
    except Exception as e:
        logging.error(f"Error in find_main_elements: {e}")
        logging.error(f"Exception type: {type(e)}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
    
    # Ensure cleanup happens even on error
    if os.path.exists(screenshot_path):
        try:
            os.remove(screenshot_path)
            logging.info("  > Cleaned up screenshot file")
        except Exception as cleanup_error:
            logging.warning(f"  > Failed to cleanup screenshot file: {cleanup_error}")

    return None

def find_end_of_content(url: str) -> int:
    """Estimates the Y-coordinate for the end of the main article content."""
    # This is a placeholder; a more sophisticated implementation could use Gemini
    # to find a specific element like "comments" or "related articles".
    return 3000 # Default fallback

def _handle_verification_page(page, max_wait_time: int = 60) -> bool:
    """
    Actively handle verification pages with multiple strategies.
    
    Args:
        page: Playwright page object
        max_wait_time: Maximum time to wait for verification to clear
        
    Returns:
        bool: True if verification cleared, False if still stuck
    """
    verification_start = time.time()
    
    # Try multiple strategies in a loop
    for attempt in range(max_wait_time // 10): # Try every 10s
        if time.time() - verification_start > max_wait_time:
            break

        try:
            # Check for iframe-based challenges (e.g., hCaptcha, reCAPTCHA)
            iframe = page.frame_locator('iframe[src*="challenge"]')
            if iframe.locator('input[type="checkbox"]').is_visible(timeout=5000):
                logging.info("  > Found iframe-based challenge, attempting to click checkbox...")
                iframe.locator('input[type="checkbox"]').click()
                time.sleep(5) # Wait for challenge to process

            # Check for common text indicators
            page_content = page.content().lower()
            verification_indicators = [
                'verifying you are human', 'cloudflare', 'checking your browser',
                'security check', 'captcha', 'challenge'
            ]
            if not any(indicator in page_content for indicator in verification_indicators):
                logging.info("  > Verification cleared.")
                return True

            logging.info(f"  > Verification page detected (Attempt {attempt+1})...")

            # Simulate human-like interaction
            page.mouse.move(random.randint(100, 500), random.randint(100, 500))
            time.sleep(0.5)
            page.keyboard.press('PageDown')
            time.sleep(1)

            # Wait before next check
            time.sleep(10)

        except Exception as e:
            logging.debug(f"  > Error during verification handling: {e}")
            time.sleep(5)
            
    # Final check
    final_content = page.content().lower()
    if any(indicator in final_content for indicator in [
        'verifying you are human', 'cloudflare', 'checking your browser'
    ]):
        logging.error("  > Failed to bypass verification page.")
        return False
        
    return True

def _launch_stealth_browser(p):
    """
    Launch browser with enhanced anti-detection settings to bypass Cloudflare.
    
    Args:
        p: Playwright instance
        
    Returns:
        browser: Configured browser instance
    """
    # Enhanced browser arguments for better anti-detection
    browser_args = [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-accelerated-2d-canvas',
        '--no-first-run',
        '--no-zygote',
        '--disable-gpu',
        '--disable-background-timer-throttling',
        '--disable-backgrounding-occluded-windows',
        '--disable-renderer-backgrounding',
        '--disable-features=TranslateUI',
        '--disable-ipc-flooding-protection',
        '--disable-hang-monitor',
        '--disable-prompt-on-repost',
        '--disable-background-downloads',
        '--disable-component-update',
        '--disable-domain-reliability',
        '--disable-features=AutofillServerCommunication',
        '--disable-features=VizDisplayCompositor'
    ]
    
    # Launch with anti-detection configuration
    browser = p.chromium.launch(
        headless=True,
        args=browser_args
    )
    
    return browser

def _setup_stealth_page(browser, viewport_width: int = 375, viewport_height: int = 812):
    """
    Create a page with enhanced anti-detection properties.
    
    Args:
        browser: Browser instance
        viewport_width: Browser viewport width  
        viewport_height: Browser viewport height
        
    Returns:
        page: Configured page instance
    """
    # Create context with enhanced user agent and settings
    context = browser.new_context(
        viewport={'width': viewport_width, 'height': viewport_height},
        user_agent='Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
        extra_http_headers={
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    )
    
    page = context.new_page()
    page.set_default_timeout(60000)
    
    # Override webdriver detection
    page.add_init_script("""
        // Override the navigator.webdriver property
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined,
        });
        
        // Mock languages and plugins to appear more human
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en'],
        });
        
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5],
        });
        
        // Override permissions query
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
        );
        
        // Add some randomness to make it look more human
        Object.defineProperty(navigator, 'hardwareConcurrency', {
            get: () => 4,
        });
        
        // Override chrome property  
        window.chrome = {
            runtime: {},
        };
    """)
    
    return page
