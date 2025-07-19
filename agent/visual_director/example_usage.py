"""
Example usage of the Enhanced Visual Director

This script demonstrates how to use the integrated visual director
to process narrative segments into visual sequences.
"""

import asyncio
import json
from datetime import datetime
from enhanced_director import create_visual_director


async def main():
    """Main example function."""
    
    # Example narrative segments (as would come from the script processor)
    segments = [
        {
            "id": "seg_001",
            "text": "OpenAI announced GPT-4, their most advanced language model yet.",
            "duration": 5.0,
            "entities": [
                {"text": "OpenAI", "type": "ORG", "start": 0, "end": 6},
                {"text": "GPT-4", "type": "PRODUCT", "start": 17, "end": 22}
            ],
            "topics": ["artificial intelligence", "language models"],
            "sentiment": 0.8
        },
        {
            "id": "seg_002", 
            "text": "The breakthrough represents a major leap forward in AI capabilities, with researchers expressing excitement about potential applications.",
            "duration": 7.0,
            "entities": [
                {"text": "AI", "type": "TECH", "start": 53, "end": 55}
            ],
            "topics": ["AI breakthrough", "research"],
            "sentiment": 0.9
        },
        {
            "id": "seg_003",
            "text": "However, concerns about safety and responsible deployment remain paramount in the AI community.",
            "duration": 5.0,
            "entities": [
                {"text": "AI community", "type": "GROUP", "start": 81, "end": 93}
            ],
            "topics": ["AI safety", "ethics"],
            "sentiment": -0.3
        }
    ]
    
    # Project metadata
    project_metadata = {
        "id": f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "title": "GPT-4 Announcement Analysis",
        "topics": ["AI", "OpenAI", "GPT-4", "technology"],
        "tone": "informative",
        "target_duration": 17.0
    }
    
    # Create visual director with configuration
    print("Initializing Enhanced Visual Director...")
    director = await create_visual_director("config_example.json")
    
    # Process the entire project
    print("\nProcessing video project...")
    result = await director.process_video_project(segments, project_metadata)
    
    # Display results
    print(f"\n=== PROJECT RESULTS ===")
    print(f"Project ID: {result['project_id']}")
    print(f"Total Segments: {result['statistics']['total_segments']}")
    print(f"Total Assets: {result['statistics']['total_assets']}")
    print(f"Unique Sources: {result['statistics']['unique_sources']}")
    
    # Show segment details
    for i, segment in enumerate(result['segments']):
        print(f"\n--- Segment {i+1} ({segment['id']}) ---")
        print(f"Text: {segment['text'][:80]}...")
        print(f"Duration: {segment['duration']}s")
        print(f"Visual Intent:")
        print(f"  - Emotions: {segment['visual_intent']['emotions']}")
        print(f"  - Scene Type: {segment['visual_intent']['scene_type']}")
        print(f"  - Emphasis: {segment['visual_intent']['emphasis_level']}")
        print(f"Assets Selected: {len(segment['assets'])}")
        
        for j, asset in enumerate(segment['assets']):
            print(f"\n  Asset {j+1}:")
            print(f"    - Type: {asset['type']}")
            print(f"    - Source: {asset['source']}")
            print(f"    - Scores: R={asset['scores']['relevance']:.2f}, "
                  f"Q={asset['scores']['quality']:.2f}, "
                  f"S={asset['scores']['semantic']:.2f}, "
                  f"C={asset['scores']['composite']:.2f}")
            if asset['local_path']:
                print(f"    - Local Path: {asset['local_path']}")
            if asset.get('metadata', {}).get('processed'):
                print(f"    - Processing: Video trimmed from {asset['metadata'].get('trim_start', 0):.1f}s")
    
    # Show attribution files
    print("\n=== ATTRIBUTION FILES ===")
    for format_type, path in result['attribution_files'].items():
        print(f"{format_type.upper()}: {path}")
    
    # Show cache performance
    cache_perf = result['statistics']['cache_performance']
    print(f"\n=== CACHE PERFORMANCE ===")
    print(f"Hits: {cache_perf['hits']}")
    print(f"Misses: {cache_perf['misses']}")
    print(f"Hit Rate: {cache_perf['hit_rate']:.2%}")
    
    # Get system statistics
    stats = director.get_system_stats()
    print(f"\n=== SYSTEM STATISTICS ===")
    print(f"Total Cached Assets: {stats['cache']['asset_cache']['total']}")
    print(f"Canonical Entities: {stats['cache']['entities']['total']}")
    print(f"Active Adapters: {stats['components']['adapters_count']}")
    print(f"CLIP Scoring: {'Enabled' if stats['components']['clip_enabled'] else 'Disabled'}")
    
    # Show popular assets
    if stats['popular_assets']:
        print(f"\n=== POPULAR ASSETS ===")
        for asset in stats['popular_assets'][:5]:
            print(f"- {asset['source']}: {asset['type']} (used {asset['usage_count']} times)")


async def test_individual_components():
    """Test individual components of the system."""
    from llm_tagger import LLMIntentTagger
    from clip_scorer import CLIPScorer
    from video_processor import VideoProcessor
    from attribution_manager import AttributionManager
    
    print("\n=== TESTING INDIVIDUAL COMPONENTS ===\n")
    
    # Test LLM Intent Tagger
    print("1. Testing LLM Intent Tagger...")
    tagger = LLMIntentTagger()
    test_segment = {
        "text": "The crowd erupted in celebration as the team scored the winning goal.",
        "entities": [{"text": "crowd", "type": "GROUP"}],
        "sentiment": 0.95
    }
    
    intent = tagger.fallback_analysis(test_segment)
    print(f"   Emotions detected: {intent.emotions}")
    print(f"   Scene type: {intent.scene_type}")
    print(f"   Emphasis level: {intent.emphasis_level}")
    
    # Test CLIP Scorer
    print("\n2. Testing CLIP Scorer...")
    scorer = CLIPScorer()
    print("   CLIP model loaded successfully")
    
    # Test Video Processor
    print("\n3. Testing Video Processor...")
    processor = VideoProcessor()
    info = await processor.get_video_info("test_video.mp4")
    if info:
        print(f"   Video info: {info['width']}x{info['height']}, {info['duration']}s")
    else:
        print("   No test video available")
    
    # Test Attribution Manager
    print("\n4. Testing Attribution Manager...")
    manager = AttributionManager()
    attr = manager.format_attribution(
        "pexels",
        "Photo by John Doe",
        "CC0",
        "https://example.com/photo"
    )
    print(f"   Sample attribution: {attr}")


if __name__ == "__main__":
    # Run main example
    asyncio.run(main())
    
    # Optionally test individual components
    # asyncio.run(test_individual_components())
