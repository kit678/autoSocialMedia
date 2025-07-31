"""
Adapter Selector for Visual Co-Generation

This module provides intelligent selection of visual adapters based on
segment type, content, emotional tone, and licensing requirements.
"""

import logging
from typing import List, Dict, Any, Tuple

# Define adapter capabilities and specialties
ADAPTER_PROFILES = {
    "pexels": {
        "name": "Pexels",
        "strengths": ["stock_photos", "stock_videos", "general_concepts", "professional_quality"],
        "weaknesses": ["specific_people", "current_events", "branded_content"],
        "licensing": "commercial",
        "quality_score": 0.9,
        "best_for": ["backgrounds", "abstract_concepts", "lifestyle", "business", "technology"]
    },
    "searxng": {
        "name": "SearXNG",
        "strengths": ["wide_coverage", "specific_entities", "current_content", "diverse_sources"],
        "weaknesses": ["quality_variance", "licensing_unclear", "rate_limits"],
        "licensing": "mixed",
        "quality_score": 0.7,
        "best_for": ["specific_companies", "products", "recent_events", "fallback_searches"]
    },
    "tenor": {
        "name": "Tenor",
        "strengths": ["reactions", "emotions", "memes", "engagement"],
        "weaknesses": ["professional_content", "serious_topics", "high_resolution"],
        "licensing": "commercial",
        "quality_score": 0.8,
        "best_for": ["emotional_moments", "reactions", "humor", "social_commentary"]
    },
    "openverse": {
        "name": "Openverse",
        "strengths": ["creative_commons", "attribution", "cultural_content", "historical"],
        "weaknesses": ["limited_modern_content", "requires_attribution"],
        "licensing": "creative_commons",
        "quality_score": 0.8,
        "best_for": ["educational", "historical", "cultural", "public_domain"]
    },
    "wikimedia": {
        "name": "Wikimedia",
        "strengths": ["encyclopedic", "historical", "educational", "verified"],
        "weaknesses": ["limited_variety", "static_images", "formal_style"],
        "licensing": "creative_commons",
        "quality_score": 0.85,
        "best_for": ["historical_events", "educational_content", "verified_facts", "public_figures"]
    },
    "nasa": {
        "name": "NASA",
        "strengths": ["space_imagery", "science", "high_quality", "authentic"],
        "weaknesses": ["limited_scope", "only_space_science"],
        "licensing": "public_domain",
        "quality_score": 0.95,
        "best_for": ["space", "astronomy", "science", "technology", "exploration"]
    },
    "coverr": {
        "name": "Coverr",
        "strengths": ["video_loops", "backgrounds", "high_quality", "professional"],
        "weaknesses": ["limited_specific_content", "mostly_generic"],
        "licensing": "commercial",
        "quality_score": 0.9,
        "best_for": ["video_backgrounds", "transitions", "ambient_footage", "b-roll"]
    },
    "gdelt_tv": {
        "name": "GDELT TV",
        "strengths": ["news_footage", "current_events", "real_footage", "global_coverage"],
        "weaknesses": ["requires_context", "quality_varies", "may_need_permission"],
        "licensing": "editorial",
        "quality_score": 0.8,
        "best_for": ["breaking_news", "current_events", "political_content", "real_footage"]
    },
    "archive_tv": {
        "name": "Archive TV",
        "strengths": ["historical_footage", "news_archives", "documentary_content"],
        "weaknesses": ["older_content", "lower_quality", "limited_recent"],
        "licensing": "mixed",
        "quality_score": 0.7,
        "best_for": ["historical_context", "retrospectives", "archive_footage", "past_events"]
    }
}

# Define selection rules based on content patterns
CONTENT_RULES = {
    # News and current events
    "breaking_news": ["gdelt_tv", "archive_tv", "searxng"],
    "current_events": ["gdelt_tv", "searxng", "pexels"],
    "political_news": ["gdelt_tv", "archive_tv", "wikimedia"],
    
    # Science and technology
    "space_science": ["nasa", "wikimedia", "pexels"],
    "technology": ["pexels", "searxng", "openverse"],
    "scientific_discovery": ["nasa", "wikimedia", "pexels"],
    
    # Emotional and social
    "emotional_reaction": ["tenor", "pexels", "searxng"],
    "social_commentary": ["tenor", "gdelt_tv", "pexels"],
    "humor": ["tenor", "searxng"],
    
    # Historical and educational
    "historical_reference": ["wikimedia", "openverse", "archive_tv"],
    "educational": ["wikimedia", "openverse", "nasa"],
    "cultural": ["openverse", "wikimedia", "searxng"],
    
    # General content
    "abstract_concept": ["pexels", "coverr", "openverse"],
    "background_footage": ["coverr", "pexels", "openverse"],
    "specific_company": ["searxng", "pexels", "wikimedia"],
    "specific_person": ["searxng", "wikimedia", "gdelt_tv"],
    "general_visual": ["pexels", "searxng", "openverse"]
}

def analyze_segment_type(segment: Dict[str, Any]) -> str:
    """
    Analyzes a segment to determine its content type for adapter selection.
    """
    # Extract segment properties
    visual_type = segment.get("visual_type", "").lower()
    intent = segment.get("intent", "inform").lower()
    emotion = segment.get("emotion", "neutral").lower()
    entities = segment.get("entities", [])
    narrative_context = segment.get("narrative_context", "").lower()
    primary_search = segment.get("primary_search_term", "").lower()
    
    # Check for space/science content
    space_keywords = ["nasa", "space", "telescope", "galaxy", "planet", "astronaut", "rocket"]
    if any(keyword in narrative_context or keyword in primary_search for keyword in space_keywords):
        return "space_science"
    
    # Check for news/current events
    news_keywords = ["breaking", "announced", "today", "yesterday", "latest", "update"]
    if any(keyword in narrative_context for keyword in news_keywords):
        if "political" in narrative_context or "government" in narrative_context:
            return "political_news"
        return "breaking_news" if "breaking" in narrative_context else "current_events"
    
    # Check for emotional content
    if emotion in ["happy", "concerned", "surprised"] or intent == "excite":
        return "emotional_reaction"
    
    # Check for historical content
    historical_keywords = ["history", "historical", "past", "century", "decade", "year ago"]
    if any(keyword in narrative_context for keyword in historical_keywords):
        return "historical_reference"
    
    # Check for specific entities
    if visual_type == "proper noun":
        if any("company" in entity.lower() or "corp" in entity.lower() for entity in entities):
            return "specific_company"
        elif any(name.count(" ") >= 1 for name in entities):  # Likely a person's name
            return "specific_person"
    
    # Check for technology content
    tech_keywords = ["ai", "technology", "software", "hardware", "digital", "cyber", "computer"]
    if any(keyword in narrative_context or keyword in primary_search for keyword in tech_keywords):
        return "technology"
    
    # Check for abstract concepts
    if visual_type == "abstract concept":
        return "abstract_concept"
    
    # Default fallback
    return "general_visual"

def score_adapter_for_segment(adapter_name: str, segment: Dict[str, Any], content_type: str) -> float:
    """
    Scores an adapter's suitability for a specific segment.
    Returns a score between 0 and 1.
    """
    adapter = ADAPTER_PROFILES.get(adapter_name, {})
    if not adapter:
        return 0.0
    
    score = 0.0
    
    # Base quality score
    score += adapter["quality_score"] * 0.3
    
    # Check if adapter is in recommended list for content type
    recommended_adapters = CONTENT_RULES.get(content_type, [])
    if adapter_name in recommended_adapters:
        # Higher score for earlier position in recommendation list
        position = recommended_adapters.index(adapter_name)
        score += (1.0 - position * 0.2) * 0.4
    
    # Check licensing compatibility
    required_license = segment.get("licence_requirement", "any")
    adapter_license = adapter["licensing"]
    
    if required_license == "any":
        score += 0.1
    elif required_license == "commercial" and adapter_license in ["commercial", "public_domain"]:
        score += 0.2
    elif required_license == "editorial" and adapter_license in ["editorial", "mixed"]:
        score += 0.2
    
    # Check if content matches adapter strengths
    segment_keywords = (segment.get("primary_search_term", "") + " " + 
                       " ".join(segment.get("secondary_keywords", []))).lower()
    
    for strength in adapter.get("best_for", []):
        if strength in segment_keywords or strength in content_type:
            score += 0.1
    
    # Penalty for weaknesses
    for weakness in adapter.get("weaknesses", []):
        if weakness in content_type:
            score -= 0.1
    
    # Ensure score is between 0 and 1
    return max(0.0, min(1.0, score))

def select_adapters_for_segment(
    segment: Dict[str, Any],
    max_adapters: int = 3,
    min_score: float = 0.3
) -> List[Tuple[str, float]]:
    """
    Selects the best adapters for a given segment.
    
    Returns:
        List of (adapter_name, score) tuples, sorted by score descending
    """
    # Determine content type
    content_type = analyze_segment_type(segment)
    
    logging.info(f"  > Segment type identified as: {content_type}")
    
    # Score all adapters
    adapter_scores = []
    for adapter_name in ADAPTER_PROFILES.keys():
        score = score_adapter_for_segment(adapter_name, segment, content_type)
        if score >= min_score:
            adapter_scores.append((adapter_name, score))
    
    # Sort by score descending
    adapter_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top adapters
    selected = adapter_scores[:max_adapters]
    
    logging.info(f"  > Selected adapters: {[f'{name} ({score:.2f})' for name, score in selected]}")
    
    return selected

def get_adapter_search_hints(adapter_name: str, segment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provides adapter-specific search hints and parameters.
    """
    hints = {
        "adapter": adapter_name,
        "primary_query": segment.get("primary_search_term", ""),
        "secondary_queries": segment.get("secondary_keywords", []),
        "filters": {}
    }
    
    # Add adapter-specific hints
    if adapter_name == "pexels":
        hints["filters"]["orientation"] = "landscape"
        hints["filters"]["size"] = "large"
        if segment.get("preferred_media") == "video":
            hints["filters"]["type"] = "video"
            
    elif adapter_name == "tenor":
        # For Tenor, focus on emotional keywords
        emotion = segment.get("emotion", "neutral")
        if emotion != "neutral":
            hints["primary_query"] = f"{emotion} reaction"
            
    elif adapter_name == "nasa":
        # For NASA, add space-related terms if not present
        if "space" not in hints["primary_query"].lower():
            hints["secondary_queries"].append("space")
            
    elif adapter_name == "gdelt_tv":
        # For GDELT, add time constraints
        hints["filters"]["timeframe"] = "recent"
        hints["filters"]["type"] = "news"
        
    elif adapter_name == "coverr":
        # For Coverr, focus on mood/atmosphere
        hints["filters"]["category"] = segment.get("emotional_tone", "neutral")
        
    return hints

def create_segment_adapter_mapping(visual_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Creates a complete adapter mapping for all segments in a visual story plan.
    """
    enhanced_segments = []
    
    for segment in visual_segments:
        # Get adapter recommendations
        selected_adapters = select_adapters_for_segment(segment)
        
        # Enhance segment with adapter info
        enhanced_segment = segment.copy()
        enhanced_segment["recommended_adapters"] = [
            {
                "name": adapter_name,
                "score": score,
                "search_hints": get_adapter_search_hints(adapter_name, segment)
            }
            for adapter_name, score in selected_adapters
        ]
        
        enhanced_segments.append(enhanced_segment)
    
    return enhanced_segments
