#!/usr/bin/env python3
"""
Test script to demonstrate the build_transition_chain function
"""

from agent.slideshow.create_smart_video import build_transition_chain, XFADE_TRANSITIONS
from agent.video_config import get_default_config
import random

def test_transition_chain():
    """Test the build_transition_chain function with different scenarios."""
    print("=== Testing build_transition_chain Function ===")
    
    # Get default configuration
    config = get_default_config("landscape")
    
    # Test 1: Single stream
    print("\n1. Testing single stream:")
    single_stream = ["[v0]"]
    result = build_transition_chain(single_stream, config)
    print(f"   Input: {single_stream}")
    print(f"   Output: {result}")
    
    # Test 2: Multiple streams with seed for reproducibility  
    print("\n2. Testing multiple streams with seed=42:")
    multi_stream = ["[v0]", "[v1]", "[v2]", "[v3]"]
    result = build_transition_chain(multi_stream, config, rng_seed=42)
    print(f"   Input: {multi_stream}")
    print(f"   Output: {result}")
    
    # Test 3: Same seed should produce same result
    print("\n3. Testing reproducibility with same seed=42:")
    result2 = build_transition_chain(multi_stream, config, rng_seed=42)
    print(f"   Same result: {result == result2}")
    
    # Test 4: Different seed should produce different result
    print("\n4. Testing different seed=123:")
    result3 = build_transition_chain(multi_stream, config, rng_seed=123)
    print(f"   Input: {multi_stream}")
    print(f"   Output: {result3}")
    print(f"   Different from seed=42: {result != result3}")
    
    # Test 5: No seed (random)
    print("\n5. Testing without seed (random):")
    result4 = build_transition_chain(multi_stream, config)
    result5 = build_transition_chain(multi_stream, config)
    print(f"   First call: {result4}")
    print(f"   Second call: {result5}")
    print(f"   Different results: {result4 != result5}")
    
    # Test 6: Empty stream
    print("\n6. Testing empty stream:")
    empty_stream = []
    result = build_transition_chain(empty_stream, config)
    print(f"   Input: {empty_stream}")
    print(f"   Output: '{result}'")
    
    print("\n=== Available Transition Types ===")
    print(f"Total transitions: {len(XFADE_TRANSITIONS)}")
    print(f"Transitions: {', '.join(XFADE_TRANSITIONS)}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_transition_chain()
