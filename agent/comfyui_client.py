"""
ComfyUI API Client

This module provides a client interface for ComfyUI workflows including
image generation, image diffusion, and video generation endpoints.
"""

import requests
import logging
import os
import time
from typing import Dict, Any, Optional, Union
import json


class ComfyUIClient:
    """Client for interacting with ComfyUI API endpoints."""
    
    def __init__(self, config: Dict[str, str]):
        """
        Initialize ComfyUI client with API endpoints.
        
        Args:
            config: Dictionary containing API endpoint URLs
                - IMAGE_API_URL: Image generation endpoint
                - DIFFUSION_API_URL: Image diffusion/animation endpoint  
                - VIDEO_API_URL: Video generation endpoint
        """
        self.config = config
        self.timeout = 300  # 5 minute timeout for generation requests
        
    def generate_image(self, prompt: str, negative_prompt: str = "", 
                      width: int = 1080, height: int = 1920) -> Dict[str, Any]:
        """
        Generate an image using the ComfyUI image generation workflow.
        
        Args:
            prompt: Positive prompt for image generation
            negative_prompt: Negative prompt for image generation
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Dictionary containing the API response with image data
        """
        try:
            payload = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'width': width,
                'height': height,
                'steps': 30,
                'guidance_scale': 7.5,
                'sampler': 'DPM++ 2M Karras'
            }
            
            logging.info(f"🖼️ Generating image: {prompt[:100]}...")
            
            response = requests.post(
                self.config['IMAGE_API_URL'],
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'image_url' in result or 'image_path' in result:
                logging.info("✅ Image generation successful")
                return {
                    'success': True,
                    'image_path': result.get('image_path'),
                    'image_url': result.get('image_url'),
                    'metadata': {
                        'prompt': prompt,
                        'negative_prompt': negative_prompt,
                        'dimensions': f"{width}x{height}"
                    }
                }
            else:
                logging.error(f"Image generation failed: {result}")
                return {'success': False, 'error': 'No image data in response'}
                
        except requests.exceptions.Timeout:
            logging.error("Image generation timed out")
            return {'success': False, 'error': 'Request timed out'}
        except requests.exceptions.RequestException as e:
            logging.error(f"Image generation request failed: {e}")
            return {'success': False, 'error': str(e)}
        except Exception as e:
            logging.error(f"Unexpected error in image generation: {e}")
            return {'success': False, 'error': str(e)}
    
    def animate_image(self, image_path: str, motion_strength: float = 0.7,
                     duration: float = 4.0) -> Dict[str, Any]:
        """
        Animate an image using the ComfyUI diffusion workflow.
        
        Args:
            image_path: Path to the input image
            motion_strength: Strength of motion (0.0 to 1.0)
            duration: Duration of animation in seconds
            
        Returns:
            Dictionary containing the API response with video data
        """
        try:
            if not os.path.exists(image_path):
                return {'success': False, 'error': f'Image file not found: {image_path}'}
            
            logging.info(f"🎬 Animating image: {os.path.basename(image_path)}")
            
            with open(image_path, 'rb') as img_file:
                files = {'image': img_file}
                data = {
                    'motion_strength': motion_strength,
                    'duration': duration,
                    'fps': 24
                }
                
                response = requests.post(
                    self.config['DIFFUSION_API_URL'],
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
            
            result = response.json()
            
            if 'video_url' in result or 'video_path' in result:
                logging.info("✅ Image animation successful")
                return {
                    'success': True,
                    'video_path': result.get('video_path'),
                    'video_url': result.get('video_url'),
                    'metadata': {
                        'source_image': image_path,
                        'motion_strength': motion_strength,
                        'duration': duration
                    }
                }
            else:
                logging.error(f"Image animation failed: {result}")
                return {'success': False, 'error': 'No video data in response'}
                
        except requests.exceptions.Timeout:
            logging.error("Image animation timed out")
            return {'success': False, 'error': 'Request timed out'}
        except requests.exceptions.RequestException as e:
            logging.error(f"Image animation request failed: {e}")
            return {'success': False, 'error': str(e)}
        except Exception as e:
            logging.error(f"Unexpected error in image animation: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_video(self, prompt: str, negative_prompt: str = "",
                      duration: float = 4.0, fps: int = 24) -> Dict[str, Any]:
        """
        Generate a video directly using the ComfyUI video generation workflow.
        
        Args:
            prompt: Positive prompt for video generation
            negative_prompt: Negative prompt for video generation
            duration: Duration of video in seconds
            fps: Frames per second
            
        Returns:
            Dictionary containing the API response with video data
        """
        try:
            payload = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'duration': duration,
                'fps': fps,
                'width': 1080,
                'height': 1920,
                'motion_strength': 0.8
            }
            
            logging.info(f"🎥 Generating video: {prompt[:100]}...")
            
            response = requests.post(
                self.config['VIDEO_API_URL'],
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'video_url' in result or 'video_path' in result:
                logging.info("✅ Video generation successful")
                return {
                    'success': True,
                    'video_path': result.get('video_path'),
                    'video_url': result.get('video_url'),
                    'metadata': {
                        'prompt': prompt,
                        'negative_prompt': negative_prompt,
                        'duration': duration,
                        'fps': fps
                    }
                }
            else:
                logging.error(f"Video generation failed: {result}")
                return {'success': False, 'error': 'No video data in response'}
                
        except requests.exceptions.Timeout:
            logging.error("Video generation timed out")
            return {'success': False, 'error': 'Request timed out'}
        except requests.exceptions.RequestException as e:
            logging.error(f"Video generation request failed: {e}")
            return {'success': False, 'error': str(e)}
        except Exception as e:
            logging.error(f"Unexpected error in video generation: {e}")
            return {'success': False, 'error': str(e)}
    
    def download_asset(self, url: str, local_path: str) -> bool:
        """
        Download a generated asset from a URL to local storage.
        
        Args:
            url: URL of the asset to download
            local_path: Local path where to save the asset
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            logging.info(f"📥 Downloaded asset: {os.path.basename(local_path)}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to download asset from {url}: {e}")
            return False
