"""
Entity Canonicalization System

This module implements algorithms for selecting and maintaining canonical
visual representations for entities (people, companies, products, etc).
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from .asset_types import Asset
from .advanced_cache import AdvancedCache


class EntityCanonicalizer:
    """
    Manages canonical visual representations for entities.
    
    Ensures consistent visual representation across segments by selecting
    and maintaining the best assets for each entity.
    """
    
    def __init__(self, cache: AdvancedCache):
        """
        Initialize the canonicalizer with a cache instance.
        
        Args:
            cache: Advanced cache instance for persistence
        """
        self.cache = cache
        self.logger = logging.getLogger(__name__)
        
        # Scoring weights for canonical selection
        self.weights = {
            'quality': 0.35,
            'relevance': 0.25,
            'source_trust': 0.20,
            'recency': 0.10,
            'usage_frequency': 0.10
        }
        
        # Source trust scores
        self.source_trust = {
            'wikimedia': 0.95,
            'nasa': 0.95,
            'openverse': 0.90,
            'pexels': 0.85,
            'searxng': 0.70,
            'tenor': 0.60,
            'archive_tv': 0.75,
            'gdelt': 0.70,
            'coverr': 0.80
        }
    
    def select_canonical_asset(
        self,
        entity_name: str,
        entity_type: str,
        candidates: List[Asset],
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Asset]:
        """
        Select the best canonical asset for an entity from candidates.
        
        Args:
            entity_name: Name of the entity
            entity_type: Type of entity (person, company, product, etc)
            candidates: List of candidate assets
            context: Optional context about usage
            
        Returns:
            Best canonical asset or None
        """
        if not candidates:
            return None
        
        # Check if we already have a canonical asset
        existing = self.cache.get_canonical_entity(entity_name, entity_type)
        if existing:
            # Check if it's still valid and recent
            if self._is_canonical_valid(existing, entity_name, entity_type):
                self.logger.info(f"Using existing canonical for {entity_name}")
                return existing
        
        # Score all candidates
        scored_candidates = []
        for asset in candidates:
            score = self._score_canonical_candidate(
                asset, entity_name, entity_type, context
            )
            scored_candidates.append((score, asset))
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        if scored_candidates:
            best_score, best_asset = scored_candidates[0]
            
            # Only set as canonical if score is high enough
            if best_score >= 0.6:
                self.cache.set_canonical_entity(
                    entity_name, entity_type, best_asset, best_score
                )
                self.logger.info(
                    f"Set new canonical for {entity_name}: {best_asset.id} "
                    f"(score: {best_score:.2f})"
                )
            
            return best_asset
        
        return None
    
    def _score_canonical_candidate(
        self,
        asset: Asset,
        entity_name: str,
        entity_type: str,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """
        Score a candidate asset for canonical selection.
        
        Args:
            asset: Asset to score
            entity_name: Entity name
            entity_type: Entity type
            context: Optional context
            
        Returns:
            Score between 0 and 1
        """
        scores = {}
        
        # 1. Quality score
        scores['quality'] = asset.quality_score
        
        # 2. Relevance score
        scores['relevance'] = asset.relevance_score
        
        # 3. Source trust score
        scores['source_trust'] = self.source_trust.get(asset.source, 0.5)
        
        # 4. Recency score (prefer newer assets)
        if asset.metadata.get('date_created'):
            try:
                created = datetime.fromisoformat(
                    asset.metadata['date_created'].replace('Z', '+00:00')
                )
                age_days = (datetime.now(created.tzinfo) - created).days
                scores['recency'] = max(0, 1.0 - (age_days / 365))  # Linear decay over 1 year
            except:
                scores['recency'] = 0.5
        else:
            scores['recency'] = 0.5
        
        # 5. Usage frequency (from cache stats)
        # This would require querying usage stats from cache
        scores['usage_frequency'] = 0.5  # Default for now
        
        # Entity-type specific adjustments
        if entity_type == 'person':
            # For people, prefer portrait orientation
            if asset.is_portrait:
                scores['quality'] += 0.1
            # Prefer images over videos for headshots
            if asset.type == 'image':
                scores['quality'] += 0.05
                
        elif entity_type == 'company':
            # For companies, logos are ideal
            if 'logo' in asset.metadata.get('title', '').lower():
                scores['relevance'] += 0.2
            # Prefer square or landscape for logos
            if 0.8 <= asset.aspect_ratio <= 1.2:  # Near square
                scores['quality'] += 0.1
                
        elif entity_type == 'product':
            # For products, prefer clear product shots
            if asset.type == 'image':
                scores['quality'] += 0.05
            # Good resolution is important
            if asset.dimensions[0] >= 1280 and asset.dimensions[1] >= 720:
                scores['quality'] += 0.1
        
        # Calculate weighted score
        total_score = 0.0
        for key, weight in self.weights.items():
            total_score += scores.get(key, 0.0) * weight
        
        return min(total_score, 1.0)
    
    def _is_canonical_valid(
        self,
        asset: Asset,
        entity_name: str,
        entity_type: str
    ) -> bool:
        """
        Check if existing canonical asset is still valid.
        
        Args:
            asset: Existing canonical asset
            entity_name: Entity name
            entity_type: Entity type
            
        Returns:
            True if still valid
        """
        # Check if asset still exists and is accessible
        if asset.local_path and not os.path.exists(asset.local_path):
            return False
        
        # Check if metadata indicates it's outdated
        metadata = asset.metadata
        if 'last_verified' in metadata:
            try:
                last_verified = datetime.fromisoformat(metadata['last_verified'])
                # Re-verify if older than 7 days
                if (datetime.now() - last_verified).days > 7:
                    return False
            except:
                pass
        
        # Entity-specific validity checks
        if entity_type == 'person':
            # People's appearances can change, shorter validity
            return True  # For now, accept if file exists
        elif entity_type == 'company':
            # Company logos are more stable
            return True
        
        return True
    
    def update_canonical_scores(self, video_project_id: str) -> None:
        """
        Update canonical asset scores based on usage performance.
        
        Args:
            video_project_id: ID of completed video project
        """
        # This would analyze the performance of assets used in the project
        # and update canonical selections based on engagement metrics
        self.logger.info(f"Updating canonical scores for project {video_project_id}")
        # Implementation would query usage stats and update confidence scores
    
    def get_entity_visual_consistency_report(
        self,
        entities: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """
        Generate a report on visual consistency for entities.
        
        Args:
            entities: List of (entity_name, entity_type) tuples
            
        Returns:
            Report with consistency metrics
        """
        report = {
            'total_entities': len(entities),
            'have_canonical': 0,
            'missing_canonical': [],
            'consistency_score': 0.0,
            'recommendations': []
        }
        
        for entity_name, entity_type in entities:
            canonical = self.cache.get_canonical_entity(entity_name, entity_type)
            if canonical:
                report['have_canonical'] += 1
            else:
                report['missing_canonical'].append({
                    'name': entity_name,
                    'type': entity_type
                })
        
        # Calculate consistency score
        if report['total_entities'] > 0:
            report['consistency_score'] = (
                report['have_canonical'] / report['total_entities']
            )
        
        # Generate recommendations
        if report['missing_canonical']:
            report['recommendations'].append(
                f"Search for visual assets for {len(report['missing_canonical'])} "
                f"entities without canonical representations"
            )
        
        if report['consistency_score'] < 0.8:
            report['recommendations'].append(
                "Consider reviewing and updating canonical assets for better consistency"
            )
        
        return report


