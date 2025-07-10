"""
Pexels Video API Integration - Search and download stock video footage
"""

import os
import logging
import requests
import time
from typing import Optional, Dict, Any
from ..decision_logger import log_decision

def search_and_download_pexels_video(search_term: str, download_dir: str, cue_id: str, category: Optional[str] = None) -> Optional[str]:
    """
    Search for a video on Pexels and download the first result.
    Note: Pexels Video API does not support category filtering.
    
    Args:
        search_term: The search term for the video.
        download_dir: The directory to save the downloaded video.
        cue_id: The cue ID to use in the filename.
        category: Optional category (ignored for videos - videos API doesn't support categories).
        
    Returns:
        The path to the downloaded video file, or None if failed.
    """
    
    pexels_api_key = os.getenv('PEXELS_API_KEY')
    if not pexels_api_key:
        logging.warning("PEXELS_API_KEY not found. Skipping Pexels video search.")
        return None
    
    try:
        # Pexels Video API endpoint (NOTE: different from photos endpoint)
        url = "https://api.pexels.com/videos/search"
        headers = {'Authorization': pexels_api_key}
        
        # Video API parameters (no category support)
        params = {
            'query': search_term,
            'per_page': 5,  # Get a few options to choose from
            'orientation': 'portrait'
        }
        
        logging.info(f"Searching Pexels Videos for: '{search_term}' (portrait)")
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'videos' not in data or not data['videos']:
            logging.warning(f"No videos found for search term: {search_term}")
            return None
            
        # Get the first video
        video = data['videos'][0]
        video_id = video['id']
        
        # Find the best quality video file
        video_files = video.get('video_files', [])
        if not video_files:
            logging.warning(f"No video files found for video ID: {video_id}")
            return None
            
        # Prefer HD quality, portrait video
        selected_file = None
        
        # Sort files to find the best portrait-oriented one
        video_files.sort(key=lambda x: x.get('width', 0)) # Sort by width to find narrowest (portrait)

        for file_info in video_files:
            width = file_info.get('width', 0)
            height = file_info.get('height', 0)
            
            # Ensure it's portrait
            if height > width:
                selected_file = file_info
                # Prefer HD quality if available
                if file_info.get('quality') == 'hd':
                    break
        
        # If no portrait video found after filtering
        if not selected_file:
            logging.warning(f"No portrait video files found for video ID: {video_id}")
            return None
            
        video_url = selected_file['link']
        
        # Download the video
        os.makedirs(download_dir, exist_ok=True)
        filename = f"pexels_video_{cue_id}_{video_id}.mp4"
        file_path = os.path.join(download_dir, filename)
        
        logging.info(f"Downloading video: {video_url}")
        video_response = requests.get(video_url, timeout=30, stream=True)
        video_response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in video_response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Log the decision
        log_decision("pexels_video", 
                     "Successfully downloaded video", 
                     f"Query: {search_term}, URL: {video_url}",
                     0.9)
        
        logging.info(f"Successfully downloaded Pexels video: {filename}")
        return file_path
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error searching/downloading Pexels video for '{search_term}': {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error in Pexels video search for '{search_term}': {e}")
        return None

def search_pexels_videos(query: str, api_key: str, duration_preference: str = "short") -> list:
    """
    Search for videos using Pexels API
    """
    try:
        # Pexels Video Search API endpoint
        url = "https://api.pexels.com/videos/search"
        
        # Set parameters based on duration preference
        if duration_preference == "short":
            min_duration = 5
            max_duration = 30
        elif duration_preference == "medium":
            min_duration = 20
            max_duration = 60
        else:  # long
            min_duration = 30
            max_duration = 120
        
        params = {
            'query': query,
            'per_page': 10,  # Get top 10 results
            'orientation': 'portrait',  # Prefer vertical videos for social media
            'size': 'medium',  # Good balance of quality and file size
            'min_duration': min_duration,
            'max_duration': max_duration
        }
        
        headers = {
            'Authorization': api_key,
            'User-Agent': 'AutoSocialMedia/1.0'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        videos = data.get('videos', [])
        
        if not videos:
            # Fallback search without duration constraints
            logging.info(f"No videos found with duration preference, trying broader search")
            params.pop('min_duration', None)
            params.pop('max_duration', None)
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            videos = data.get('videos', [])
        
        # Process and rank videos
        processed_videos = []
        for video in videos:
            try:
                # Get best quality for portrait orientation
                video_files = video.get('video_files', [])
                best_file = select_best_video_quality(video_files)
                
                if best_file:
                    processed_videos.append({
                        'id': video.get('id'),
                        'duration': video.get('duration'),
                        'width': video.get('width'),
                        'height': video.get('height'),
                        'download_url': best_file['link'],
                        'selected_quality': best_file['quality'],
                        'file_type': best_file['file_type']
                    })
                    
            except Exception as e:
                logging.warning(f"Error processing video {video.get('id')}: {e}")
                continue
        
        # Sort by duration preference and portrait aspect ratio
        processed_videos.sort(key=lambda x: (
            abs(x['duration'] - (min_duration + max_duration) / 2),  # Prefer middle of duration range
            abs(x['height'] / x['width'] - 16/9) if x['width'] > 0 else 999  # Prefer 9:16 portrait aspect ratio (inverted for height/width)
        ))
        
        logging.info(f"Found {len(processed_videos)} suitable videos for '{query}'")
        return processed_videos
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Pexels API request failed: {e}")
        return []
    except Exception as e:
        # Log the decision to fallback with a clear reason
        log_decision(
            "pexels_video_search",
            "Pexels video search failed",
            f"An unexpected error occurred during Pexels video search for '{query}': {e}",
            0.1
        )
        logging.error(f"Error in Pexels video search: {e}")
        return []

def select_best_video_quality(video_files: list) -> Optional[Dict[str, Any]]:
    """
    Select the best video quality for our use case
    """
    if not video_files:
        return None
    
    # Prefer HD quality videos that are not too large
    quality_preference = ['hd', 'sd', 'mobile']
    
    for quality in quality_preference:
        for video_file in video_files:
            if video_file.get('quality') == quality:
                # Prefer MP4 format
                if video_file.get('file_type', '').lower() == 'mp4':
                    return video_file
    
    # Fallback to first available
    return video_files[0] if video_files else None

def download_pexels_video(video_data: Dict[str, Any], output_dir: str, filename: str) -> Optional[str]:
    """
    Download a video from Pexels
    """
    try:
        download_url = video_data['download_url']
        file_type = video_data.get('file_type', 'mp4').lower()
        
        # Ensure file_type is just the extension without dots or MIME type
        if file_type.startswith('.'):
            file_type = file_type[1:]
        elif '/' in file_type:  # Handle MIME types like 'video/mp4'
            file_type = file_type.split('/')[-1]
        
        # Prepare output path
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{filename}.{file_type}")
        
        # Download with progress tracking
        logging.info(f"Downloading Pexels video: {video_data['id']} to {output_path}")
        logging.debug(f"Download URL: {download_url}")
        
        headers = {
            'User-Agent': 'AutoSocialMedia/1.0'
        }
        
        response = requests.get(download_url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Log progress for large files
                    if total_size > 0 and downloaded_size % (1024 * 1024) == 0:  # Every MB
                        progress = (downloaded_size / total_size) * 100
                        logging.debug(f"Download progress: {progress:.1f}%")
        
        # Verify file was downloaded
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logging.info(f"Successfully downloaded video: {output_path}")
            return output_path
        else:
            logging.error("Downloaded file is empty or missing")
            return None
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download video: {e}")
        return None
    except Exception as e:
        logging.error(f"Error downloading Pexels video: {e}")
        return None

def get_video_info(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Get basic information about a video file using moviepy
    """
    try:
        from moviepy.editor import VideoFileClip
        
        with VideoFileClip(video_path) as clip:
            return {
                'duration': clip.duration,
                'width': clip.w,
                'height': clip.h,
                'fps': clip.fps,
                'aspect_ratio': clip.w / clip.h
            }
            
    except Exception as e:
        logging.warning(f"Could not get video info for {video_path}: {e}")
        return None 