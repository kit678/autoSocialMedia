"""
AI Prompt Crafter for Visual Generation

This module creates detailed, context-aware prompts for AI image and video generation
based on the narrative context, emotional tone, and visual requirements of a script segment.
"""

import logging
from typing import Dict, Any, List

BASE_STYLE_PROMPT = "cinematic, dramatic lighting, high contrast, photorealistic, 4k"

BASE_NEGATIVE_PROMPT = "blurry, low quality, cartoon, anime, text, watermark, logo, people, faces, oversaturated, unrealistic"

def create_visual_prompt(segment: Dict[str, Any], full_script: str) -> (str, str):
    """
    Generates a detailed positive and negative prompt for a visual segment.

    Args:
        segment: The visual segment from the visual story plan.
        full_script: The complete narration script for overall context.

    Returns:
        A tuple containing the (positive_prompt, negative_prompt).
    """
    try:
        narrative = segment.get('narrative_context', '')
        primary_term = segment.get('primary_search_term', '')
        visual_type = segment.get('visual_type', 'Abstract Concept')
        emotion = segment.get('emotional_tone', 'neutral')
        motion_hint = segment.get('motion_hint', 'static')

        # 1. Core Subject
        core_subject = f"{primary_term}, inspired by the concept: '{narrative}'."

        # 2. Emotional and Stylistic Modifiers
        style_modifiers = [BASE_STYLE_PROMPT]
        if emotion == 'awe' or emotion == 'excitement':
            style_modifiers.append("epic scale, sense of wonder, vibrant colors")
        elif emotion == 'shock' or emotion == 'concerned':
            style_modifiers.append("dramatic, tense atmosphere, dark and moody lighting")
        elif emotion == 'hopeful' or emotion == 'optimistic':
            style_modifiers.append("bright, optimistic, clean aesthetic, warm light")

        # 3. Composition and Framing
        composition = []
        if visual_type == 'Concrete Object/Action':
            composition.append("dynamic action shot")
        elif visual_type == 'Abstract Concept':
            composition.append("symbolic and metaphorical representation")
        if motion_hint == 'slow_pan':
            composition.append("wide shot, slow panning camera movement")
        elif motion_hint == 'dynamic':
            composition.append("dynamic camera movement, multiple angles")

        # 4. Negative Prompt Construction
        negative_prompt = BASE_NEGATIVE_PROMPT
        if visual_type != 'Proper Noun': # Avoid faces unless it's a specific person
            negative_prompt += ", faces, people"

        # Assemble the final prompt
        prompt_parts = [core_subject]
        if style_modifiers:
            prompt_parts.append(", ".join(style_modifiers))
        if composition:
            prompt_parts.append(", ".join(composition))
        
        final_prompt = ". ".join(prompt_parts)
        
        logging.info(f"Generated AI prompt for '{primary_term}': {final_prompt[:100]}...")

        return final_prompt, negative_prompt

    except Exception as e:
        logging.error(f"Failed to craft prompt for segment: {segment.get('primary_search_term')}: {e}")
        # Fallback prompt
        return f"A visual representation of {segment.get('narrative_context', 'a concept')}", BASE_NEGATIVE_PROMPT

