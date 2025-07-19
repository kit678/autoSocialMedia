"""
Visual Semantic Scoring using CLIP

This module implements CLIP-based visual similarity scoring to match
images/videos with text queries for better visual selection.
"""

import os
import logging
import hashlib
from typing import Optional, List, Tuple, Dict, Any
from urllib.parse import urlparse
import numpy as np

try:
    import torch
    import clip
    from PIL import Image
    import requests
    from io import BytesIO
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("CLIP not available. Install with: pip install torch torchvision clip pillow")


class VisualScorer:
    """
    Provides CLIP-based visual semantic scoring for asset selection.
    
    Uses OpenAI's CLIP model to compute similarity between text queries
    and visual content (images/video thumbnails).
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """
        Initialize CLIP model for visual scoring.
        
        Args:
            model_name: CLIP model to use (ViT-B/32, ViT-L/14, etc.)
            device: Device to run on (cuda/cpu). Auto-detected if None.
        """
        if not CLIP_AVAILABLE:
            self.enabled = False
            logging.warning("Visual scorer disabled - CLIP not available")
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
            
            logging.info(f"CLIP visual scorer initialized with {model_name} on {device}")
            
        except Exception as e:
            self.enabled = False
            logging.error(f"Failed to initialize CLIP: {e}")
    
    def score_visual_similarity(
        self, 
        image_url: str, 
        text_query: str,
        use_cache: bool = True
    ) -> float:
        """
        Calculate similarity between an image and text query.
        
        Args:
            image_url: URL of the image to score
            text_query: Text to match against
            use_cache: Whether to use cached embeddings
            
        Returns:
            Similarity score between 0 and 1
        """
        if not self.enabled:
            return 0.5  # Neutral score when CLIP unavailable
        
        try:
            # Get image embedding
            image_features = self._get_image_features(image_url, use_cache)
            if image_features is None:
                return 0.0
            
            # Get text embedding
            text_features = self._get_text_features(text_query, use_cache)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(image_features, text_features)
            
            # Convert to 0-1 range (CLIP similarities are typically -1 to 1)
            return (similarity + 1.0) / 2.0
            
        except Exception as e:
            logging.warning(f"Visual scoring failed for {image_url}: {e}")
            return 0.5
    
    def batch_score_visuals(
        self,
        image_urls: List[str],
        text_queries: List[str]
    ) -> np.ndarray:
        """
        Score multiple images against multiple text queries.
        
        Args:
            image_urls: List of image URLs
            text_queries: List of text queries
            
        Returns:
            Matrix of similarity scores (images x queries)
        """
        if not self.enabled:
            return np.ones((len(image_urls), len(text_queries))) * 0.5
        
        scores = np.zeros((len(image_urls), len(text_queries)))
        
        for i, image_url in enumerate(image_urls):
            for j, text_query in enumerate(text_queries):
                scores[i, j] = self.score_visual_similarity(image_url, text_query)
        
        return scores
    
    def _get_image_features(self, image_url: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """Extract CLIP features from an image URL."""
        # Check cache
        cache_key = self._url_to_cache_key(image_url)
        if use_cache and cache_key in self._image_cache:
            return self._image_cache[cache_key]
        
        try:
            # Download image
            if image_url.startswith(('http://', 'https://')):
                response = requests.get(image_url, timeout=5)
                image = Image.open(BytesIO(response.content))
            else:
                # Local file
                image = Image.open(image_url)
            
            # Preprocess and encode
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features.cpu().numpy().flatten()
                image_features = image_features / np.linalg.norm(image_features)
            
            # Cache result
            if use_cache:
                self._image_cache[cache_key] = image_features
            
            return image_features
            
        except Exception as e:
            logging.warning(f"Failed to process image {image_url}: {e}")
            return None
    
    def _get_text_features(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Extract CLIP features from text."""
        # Check cache
        if use_cache and text in self._text_cache:
            return self._text_cache[text]
        
        # Tokenize and encode
        text_input = clip.tokenize([text], truncate=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features = text_features.cpu().numpy().flatten()
            text_features = text_features / np.linalg.norm(text_features)
        
        # Cache result
        if use_cache:
            self._text_cache[text] = text_features
        
        return text_features
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(a, b))
    
    def _url_to_cache_key(self, url: str) -> str:
        """Convert URL to cache key."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def find_best_visual_match(
        self,
        image_urls: List[str],
        text_queries: List[str],
        weights: Optional[List[float]] = None
    ) -> Tuple[int, float]:
        """
        Find the best matching image for a set of text queries.
        
        Args:
            image_urls: List of candidate image URLs
            text_queries: List of text queries to match
            weights: Optional weights for each query
            
        Returns:
            Tuple of (best_image_index, average_score)
        """
        if not self.enabled or not image_urls:
            return 0, 0.5
        
        scores = self.batch_score_visuals(image_urls, text_queries)
        
        # Apply weights if provided
        if weights:
            weights = np.array(weights) / np.sum(weights)
            weighted_scores = scores @ weights
        else:
            # Equal weight to all queries
            weighted_scores = scores.mean(axis=1)
        
        best_idx = int(np.argmax(weighted_scores))
        best_score = float(weighted_scores[best_idx])
        
        return best_idx, best_score
    
    def extract_video_thumbnail(self, video_path: str, timestamp: float = 1.0) -> Optional[str]:
        """
        Extract a thumbnail from a video for CLIP scoring.
        
        Args:
            video_path: Path to video file
            timestamp: Time in seconds to extract frame
            
        Returns:
            Path to extracted thumbnail or None
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(fps * timestamp)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Save thumbnail
                thumb_path = f"{video_path}_thumb.jpg"
                cv2.imwrite(thumb_path, frame)
                return thumb_path
            
        except Exception as e:
            logging.warning(f"Failed to extract video thumbnail: {e}")
        
        return None
    
    def score_video_similarity(
        self,
        video_path: str,
        text_query: str,
        sample_frames: int = 3
    ) -> float:
        """
        Score a video against a text query by sampling multiple frames.
        
        Args:
            video_path: Path to video file
            text_query: Text to match against
            sample_frames: Number of frames to sample
            
        Returns:
            Average similarity score
        """
        if not self.enabled:
            return 0.5
        
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            
            # Sample frames evenly throughout the video
            timestamps = np.linspace(0.5, duration - 0.5, sample_frames)
            scores = []
            
            for ts in timestamps:
                frame_number = int(fps * ts)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    # Convert frame to PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    
                    # Score this frame
                    image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        image_features = self.model.encode_image(image_input)
                        image_features = image_features.cpu().numpy().flatten()
                        image_features = image_features / np.linalg.norm(image_features)
                    
                    text_features = self._get_text_features(text_query)
                    similarity = self._cosine_similarity(image_features, text_features)
                    scores.append((similarity + 1.0) / 2.0)
            
            cap.release()
            
            # Return average score across sampled frames
            return np.mean(scores) if scores else 0.5
            
        except Exception as e:
            logging.warning(f"Failed to score video {video_path}: {e}")
            return 0.5
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        self._image_cache.clear()
        self._text_cache.clear()
        logging.info("Visual scorer cache cleared")


# Global instance for reuse
_visual_scorer = None


def get_visual_scorer(model_name: str = "ViT-B/32") -> VisualScorer:
    """
    Get or create a global VisualScorer instance.
    
    Args:
        model_name: CLIP model to use
        
    Returns:
        VisualScorer instance
    """
    global _visual_scorer
    if _visual_scorer is None:
        _visual_scorer = VisualScorer(model_name)
    return _visual_scorer


def score_asset_visual_relevance(
    asset,
    search_terms: List[str],
    segment_text: Optional[str] = None
) -> float:
    """
    Score an asset's visual relevance using CLIP.
    
    Args:
        asset: Asset object with image URL
        search_terms: Search terms used to find the asset
        segment_text: Optional full segment text for context
        
    Returns:
        Visual relevance score (0-1)
    """
    scorer = get_visual_scorer()
    
    if not scorer.enabled:
        return 0.5  # Neutral score when CLIP unavailable
    
    # Build queries from search terms and segment text
    queries = search_terms.copy()
    if segment_text:
        queries.append(segment_text)
    
    # Get preview URL or main URL
    image_url = asset.metadata.get('preview_url') or asset.url
    
    # For videos, try to use thumbnail
    if asset.type == "video" and asset.local_path:
        thumb = scorer.extract_video_thumbnail(asset.local_path)
        if thumb:
            image_url = thumb
    
    # Score against all queries and take the best match
    scores = [scorer.score_visual_similarity(image_url, query) for query in queries]
    return max(scores) if scores else 0.5
