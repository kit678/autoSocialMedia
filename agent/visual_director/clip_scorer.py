"""
CLIP-based Visual Semantic Scorer

This module implements CLIP-based visual similarity scoring for the enhanced
visual director system, providing semantic matching between text queries and images.
"""

import os
import logging
import hashlib
from typing import Optional, List, Tuple, Dict, Any
from urllib.parse import urlparse
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    import torch
    import clip
    from PIL import Image
    import requests
    from io import BytesIO
    import cv2
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("CLIP not available. Install with: pip install torch torchvision clip-by-openai pillow opencv-python")

from .asset_types import Asset


class CLIPScorer:
    """
    Provides CLIP-based visual semantic scoring for asset selection.
    
    Uses OpenAI's CLIP model to compute similarity between text queries
    and visual content (images/video thumbnails).
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """
        Initialize CLIP scorer.
        
        Args:
            model_name: CLIP model to use (ViT-B/32, ViT-L/14, etc.)
            device: Device to run on (cuda/cpu). Auto-detected if None.
        """
        self.logger = logging.getLogger(__name__)
        
        if not CLIP_AVAILABLE:
            self.enabled = False
            self.logger.warning("CLIP scorer disabled - required packages not available")
            return
            
        try:
            # Auto-detect device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.device = device
            self.model, self.preprocess = clip.load(model_name, device=device)
            self.model.eval()
            self.enabled = True
            
            # Cache for embeddings to avoid recomputation
            self._image_cache = {}  # url -> embedding
            self._text_cache = {}   # text -> embedding
            
            # Thread pool for async operations
            self._executor = ThreadPoolExecutor(max_workers=4)
            
            self.logger.info(f"CLIP scorer initialized with {model_name} on {device}")
            
        except Exception as e:
            self.enabled = False
            self.logger.error(f"Failed to initialize CLIP: {e}")
    
    async def score_assets(self, assets: List[Asset], text_query: str) -> List[Asset]:
        """
        Score assets using CLIP similarity with text query.
        
        Args:
            assets: List of assets to score
            text_query: Text query to match against
            
        Returns:
            Assets with semantic_score attribute added
        """
        if not self.enabled:
            # Set neutral scores when CLIP unavailable
            for asset in assets:
                asset.semantic_score = 0.5
            return assets
        
        try:
            # Get text embedding
            text_features = await self._get_text_features_async(text_query)
            
            # Process assets in parallel
            tasks = []
            for asset in assets:
                task = self._score_single_asset(asset, text_features)
                tasks.append(task)
            
            # Wait for all scoring to complete
            await asyncio.gather(*tasks)
            
            return assets
            
        except Exception as e:
            self.logger.error(f"CLIP scoring failed: {e}")
            # Fallback to neutral scores
            for asset in assets:
                asset.semantic_score = 0.5
            return assets
    
    async def _score_single_asset(self, asset: Asset, text_features: np.ndarray) -> None:
        """
        Score a single asset against text features.
        
        Args:
            asset: Asset to score
            text_features: Pre-computed text features
        """
        try:
            # Get image features
            image_features = await self._get_image_features_async(asset)
            
            if image_features is not None:
                # Calculate cosine similarity
                similarity = self._cosine_similarity(image_features, text_features)
                # Convert to 0-1 range (CLIP similarities are typically -1 to 1)
                asset.semantic_score = (similarity + 1.0) / 2.0
            else:
                asset.semantic_score = 0.0
                
        except Exception as e:
            self.logger.warning(f"Failed to score asset {asset.id}: {e}")
            asset.semantic_score = 0.5
    
    async def _get_image_features_async(self, asset: Asset) -> Optional[np.ndarray]:
        """
        Get image features asynchronously.
        
        Args:
            asset: Asset to process
            
        Returns:
            Image features or None if failed
        """
        # Check cache first
        image_url = self._get_image_url(asset)
        cache_key = self._url_to_cache_key(image_url)
        
        if cache_key in self._image_cache:
            return self._image_cache[cache_key]
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            features = await loop.run_in_executor(
                self._executor, 
                self._extract_image_features, 
                asset
            )
            
            if features is not None:
                self._image_cache[cache_key] = features
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Failed to get image features for {asset.id}: {e}")
            return None
    
    async def _get_text_features_async(self, text: str) -> np.ndarray:
        """
        Get text features asynchronously.
        
        Args:
            text: Text to process
            
        Returns:
            Text features
        """
        # Check cache
        if text in self._text_cache:
            return self._text_cache[text]
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(
            self._executor,
            self._extract_text_features,
            text
        )
        
        # Cache result
        self._text_cache[text] = features
        return features
    
    def _get_image_url(self, asset: Asset) -> str:
        """
        Get the best image URL for the asset.
        
        Args:
            asset: Asset to get URL for
            
        Returns:
            Image URL to use
        """
        # Prefer preview URL for faster loading
        if 'preview_url' in asset.metadata:
            return asset.metadata['preview_url']
        
        # Use local path if available
        if asset.local_path and os.path.exists(asset.local_path):
            return asset.local_path
        
        # Fall back to main URL
        return asset.url
    
    def _extract_image_features(self, asset: Asset) -> Optional[np.ndarray]:
        """
        Extract CLIP features from an asset's image.
        
        Args:
            asset: Asset to process
            
        Returns:
            Image features or None if failed
        """
        try:
            image_url = self._get_image_url(asset)
            
            # Load image
            if image_url.startswith(('http://', 'https://')):
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                # Local file
                if asset.type == 'video':
                    # Extract frame from video
                    image = self._extract_video_frame(image_url)
                    if image is None:
                        return None
                else:
                    image = Image.open(image_url)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess and encode
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features.cpu().numpy().flatten()
                image_features = image_features / np.linalg.norm(image_features)
            
            return image_features
            
        except Exception as e:
            self.logger.warning(f"Failed to extract image features: {e}")
            return None
    
    def _extract_text_features(self, text: str) -> np.ndarray:
        """
        Extract CLIP features from text.
        
        Args:
            text: Text to process
            
        Returns:
            Text features
        """
        # Tokenize and encode
        text_input = clip.tokenize([text], truncate=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features = text_features.cpu().numpy().flatten()
            text_features = text_features / np.linalg.norm(text_features)
        
        return text_features
    
    def _extract_video_frame(self, video_path: str, timestamp: float = 1.0) -> Optional['Image.Image']:
        """
        Extract a frame from a video file.
        
        Args:
            video_path: Path to video file
            timestamp: Time in seconds to extract frame
            
        Returns:
            PIL Image or None if failed
        """
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(fps * timestamp)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame_rgb)
            
        except Exception as e:
            self.logger.warning(f"Failed to extract video frame: {e}")
        
        return None
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity
        """
        return float(np.dot(a, b))
    
    def _url_to_cache_key(self, url: str) -> str:
        """
        Convert URL to cache key.
        
        Args:
            url: URL to convert
            
        Returns:
            Cache key
        """
        return hashlib.md5(url.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        self._image_cache.clear()
        self._text_cache.clear()
        self.logger.info("CLIP scorer cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scorer statistics."""
        return {
            'enabled': self.enabled,
            'device': getattr(self, 'device', 'unknown'),
            'cache_size': {
                'images': len(self._image_cache) if self.enabled else 0,
                'text': len(self._text_cache) if self.enabled else 0
            }
        }


# Global instance for reuse
_clip_scorer = None


def get_clip_scorer(model_name: str = "ViT-B/32") -> CLIPScorer:
    """
    Get or create a global CLIPScorer instance.
    
    Args:
        model_name: CLIP model to use
        
    Returns:
        CLIPScorer instance
    """
    global _clip_scorer
    if _clip_scorer is None:
        _clip_scorer = CLIPScorer(model_name)
    return _clip_scorer


async def score_asset_semantic_relevance(
    asset: Asset,
    text_query: str,
    model_name: str = "ViT-B/32"
) -> float:
    """
    Score an asset's semantic relevance to a text query.
    
    Args:
        asset: Asset to score
        text_query: Text query to match against
        model_name: CLIP model to use
        
    Returns:
        Semantic relevance score (0-1)
    """
    scorer = get_clip_scorer(model_name)
    
    if not scorer.enabled:
        return 0.5  # Neutral score when CLIP unavailable
    
    # Score single asset
    assets = await scorer.score_assets([asset], text_query)
    return getattr(assets[0], 'semantic_score', 0.5)
