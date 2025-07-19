"""Asset scoring module for ranking and selecting visual assets.

This module implements a sophisticated scoring system that considers
relevance, quality, diversity, and licensing to select the best assets.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from collections import Counter
import math

from .asset_types import Asset, AssetScoringConfig


def score_asset(
    asset: Asset,
    segment: Dict[str, Any],
    existing_assets: List[Asset],
    config: AssetScoringConfig
) -> float:
    """Score an asset based on multiple criteria.
    
    This is the main scoring function that combines various scoring factors
    to produce a final score for asset selection.
    
    Args:
        asset: The asset to score
        segment: The segment requiring a visual
        existing_assets: Already selected assets (for diversity calculation)
        config: Scoring configuration with weights
        
    Returns:
        Composite score between 0 and 1
    """
    # Start with base scores from the asset
    relevance = asset.relevance_score
    quality = asset.quality_score
    
    # Calculate diversity score
    diversity = calculate_diversity_score(asset, existing_assets)
    asset.diversity_score = diversity
    
    # Apply bonuses and penalties
    orientation_bonus = calculate_orientation_bonus(asset, segment, config)
    media_type_bonus = calculate_media_type_bonus(asset, segment, config)
    license_penalty = calculate_license_penalty(asset, config)
    source_bonus = calculate_source_bonus(asset, config)
    
    # Calculate weighted base score
    base_score = (
        relevance * config.relevance_weight +
        quality * config.quality_weight +
        diversity * config.diversity_weight
    )
    
    # Apply modifiers
    final_score = base_score + orientation_bonus + media_type_bonus + source_bonus - license_penalty
    
    # Ensure score stays within 0-1 range
    return max(0.0, min(1.0, final_score))


def calculate_diversity_score(asset: Asset, existing_assets: List[Asset]) -> float:
    """Calculate how different this asset is from already selected assets.
    
    Args:
        asset: The asset to evaluate
        existing_assets: Already selected assets
        
    Returns:
        Diversity score between 0 and 1
    """
    if not existing_assets:
        return 1.0  # First asset is maximally diverse
    
    diversity_factors = []
    
    # Source diversity
    source_counts = Counter(a.source for a in existing_assets)
    source_diversity = 1.0 - (source_counts.get(asset.source, 0) / len(existing_assets))
    diversity_factors.append(source_diversity * 0.3)
    
    # Orientation diversity
    orientation_counts = Counter(
        'portrait' if a.is_portrait else 'landscape' 
        for a in existing_assets
    )
    asset_orientation = 'portrait' if asset.is_portrait else 'landscape'
    orientation_diversity = 1.0 - (orientation_counts.get(asset_orientation, 0) / len(existing_assets))
    diversity_factors.append(orientation_diversity * 0.2)
    
    # Type diversity (image vs video)
    type_counts = Counter(a.type for a in existing_assets)
    type_diversity = 1.0 - (type_counts.get(asset.type, 0) / len(existing_assets))
    diversity_factors.append(type_diversity * 0.3)
    
    # Color diversity (if color metadata available)
    if 'dominant_color' in asset.metadata:
        color_diversity = calculate_color_diversity(asset, existing_assets)
        diversity_factors.append(color_diversity * 0.2)
    else:
        # No color data, assume moderate diversity
        diversity_factors.append(0.5 * 0.2)
    
    return sum(diversity_factors)


def calculate_color_diversity(asset: Asset, existing_assets: List[Asset]) -> float:
    """Calculate color-based diversity score.
    
    Args:
        asset: Asset with color metadata
        existing_assets: Existing assets to compare against
        
    Returns:
        Color diversity score
    """
    asset_color = asset.metadata.get('dominant_color')
    if not asset_color:
        return 0.5
    
    # Count how many existing assets have similar colors
    similar_count = 0
    for existing in existing_assets:
        existing_color = existing.metadata.get('dominant_color')
        if existing_color and are_colors_similar(asset_color, existing_color):
            similar_count += 1
    
    if not existing_assets:
        return 1.0
    
    return 1.0 - (similar_count / len(existing_assets))


def are_colors_similar(color1: str, color2: str, threshold: float = 0.2) -> bool:
    """Check if two colors are similar.
    
    Args:
        color1: First color (hex or name)
        color2: Second color (hex or name)
        threshold: Similarity threshold
        
    Returns:
        True if colors are similar
    """
    # Simple implementation - could be enhanced with proper color distance
    # For now, just check if they're the same color family
    color_families = {
        'red': ['red', 'crimson', 'scarlet', 'maroon'],
        'blue': ['blue', 'navy', 'azure', 'cobalt'],
        'green': ['green', 'emerald', 'lime', 'forest'],
        'yellow': ['yellow', 'gold', 'amber', 'lemon'],
        'purple': ['purple', 'violet', 'magenta', 'lavender'],
        'orange': ['orange', 'tangerine', 'coral', 'peach'],
        'brown': ['brown', 'tan', 'beige', 'bronze'],
        'gray': ['gray', 'grey', 'silver', 'charcoal'],
        'black': ['black', 'ebony', 'jet'],
        'white': ['white', 'ivory', 'cream', 'snow']
    }
    
    # Find which family each color belongs to
    family1 = None
    family2 = None
    
    for family, colors in color_families.items():
        if any(c in color1.lower() for c in colors):
            family1 = family
        if any(c in color2.lower() for c in colors):
            family2 = family
    
    return family1 == family2 and family1 is not None


def calculate_orientation_bonus(
    asset: Asset, 
    segment: Dict[str, Any], 
    config: AssetScoringConfig
) -> float:
    """Calculate bonus score for preferred orientation.
    
    Args:
        asset: The asset to evaluate
        segment: Segment data
        config: Scoring configuration
        
    Returns:
        Orientation bonus (0 to config.prefer_portrait)
    """
    # Always prefer portrait for social media
    if asset.is_portrait:
        return config.prefer_portrait
    return 0.0


def calculate_media_type_bonus(
    asset: Asset,
    segment: Dict[str, Any],
    config: AssetScoringConfig
) -> float:
    """Calculate bonus for matching preferred media type.
    
    Args:
        asset: The asset to evaluate
        segment: Segment with preferred_media field
        config: Scoring configuration
        
    Returns:
        Media type bonus
    """
    preferred_media = segment.get('preferred_media', 'image')
    
    # Exact match gets full bonus
    if asset.type == preferred_media:
        return config.prefer_video if asset.type == 'video' else 0.0
    
    # Reaction GIFs can substitute for video in some cases
    if preferred_media == 'video' and asset.type == 'reaction':
        return config.prefer_video * 0.5
    
    return 0.0


def calculate_license_penalty(asset: Asset, config: AssetScoringConfig) -> float:
    """Calculate penalty for restrictive licenses.
    
    Args:
        asset: The asset to evaluate
        config: Scoring configuration
        
    Returns:
        License penalty (positive value to subtract)
    """
    return config.license_penalties.get(asset.licence, 0.0)


def calculate_source_bonus(asset: Asset, config: AssetScoringConfig) -> float:
    """Calculate bonus for preferred sources.
    
    Args:
        asset: The asset to evaluate
        config: Scoring configuration
        
    Returns:
        Source bonus
    """
    return config.source_bonuses.get(asset.source, 0.0)


def rank_assets(
    assets: List[Asset],
    segment: Dict[str, Any],
    existing_assets: List[Asset],
    config: Optional[AssetScoringConfig] = None
) -> List[Asset]:
    """Rank a list of assets by score.
    
    Args:
        assets: Assets to rank
        segment: Segment requiring a visual
        existing_assets: Already selected assets
        config: Optional scoring configuration
        
    Returns:
        Assets sorted by score (highest first)
    """
    if config is None:
        config = AssetScoringConfig()
    
    # Score all assets
    for asset in assets:
        asset.composite_score = score_asset(asset, segment, existing_assets, config)
    
    # Sort by composite score (highest first)
    return sorted(assets, key=lambda a: a.composite_score, reverse=True)


def select_best_asset(
    assets: List[Asset],
    segment: Dict[str, Any],
    existing_assets: List[Asset],
    config: Optional[AssetScoringConfig] = None,
    min_score: float = 0.3
) -> Optional[Asset]:
    """Select the best asset from a list of candidates.
    
    Args:
        assets: Candidate assets
        segment: Segment requiring a visual
        existing_assets: Already selected assets
        config: Optional scoring configuration
        min_score: Minimum acceptable score
        
    Returns:
        Best asset or None if none meet minimum score
    """
    if not assets:
        return None
    
    ranked_assets = rank_assets(assets, segment, existing_assets, config)
    
    # Return best asset if it meets minimum score
    best_asset = ranked_assets[0]
    if best_asset.composite_score >= min_score:
        logging.info(
            f"Selected {best_asset.type} from {best_asset.source} "
            f"with score {best_asset.composite_score:.2f}"
        )
        return best_asset
    
    logging.warning(
        f"Best asset score {best_asset.composite_score:.2f} "
        f"below minimum {min_score}"
    )
    return None


def get_scoring_report(
    asset: Asset,
    segment: Dict[str, Any],
    existing_assets: List[Asset],
    config: AssetScoringConfig
) -> Dict[str, Any]:
    """Generate a detailed scoring report for an asset.
    
    Useful for debugging and understanding scoring decisions.
    
    Args:
        asset: The asset that was scored
        segment: The segment context
        existing_assets: Other selected assets
        config: Scoring configuration
        
    Returns:
        Detailed scoring breakdown
    """
    # Recalculate individual components
    diversity = calculate_diversity_score(asset, existing_assets)
    orientation_bonus = calculate_orientation_bonus(asset, segment, config)
    media_type_bonus = calculate_media_type_bonus(asset, segment, config)
    license_penalty = calculate_license_penalty(asset, config)
    source_bonus = calculate_source_bonus(asset, config)
    
    # Calculate weighted components
    relevance_weighted = asset.relevance_score * config.relevance_weight
    quality_weighted = asset.quality_score * config.quality_weight
    diversity_weighted = diversity * config.diversity_weight
    
    # Final score
    final_score = score_asset(asset, segment, existing_assets, config)
    
    return {
        'asset_id': asset.id,
        'asset_type': asset.type,
        'asset_source': asset.source,
        'scores': {
            'relevance': {
                'raw': asset.relevance_score,
                'weight': config.relevance_weight,
                'weighted': relevance_weighted
            },
            'quality': {
                'raw': asset.quality_score,
                'weight': config.quality_weight,
                'weighted': quality_weighted
            },
            'diversity': {
                'raw': diversity,
                'weight': config.diversity_weight,
                'weighted': diversity_weighted
            }
        },
        'bonuses': {
            'orientation': orientation_bonus,
            'media_type': media_type_bonus,
            'source': source_bonus
        },
        'penalties': {
            'license': license_penalty
        },
        'final_score': final_score,
        'segment_context': {
            'preferred_media': segment.get('preferred_media'),
            'intent': segment.get('intent'),
            'emotion': segment.get('emotion')
        }
    }
