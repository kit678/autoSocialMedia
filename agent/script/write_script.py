"""
Script Writing with Visual Story Planning

This module now uses a proactive visual story planning approach where we analyze
the article for visual opportunities FIRST, then engineer a script to match those visuals.
This creates much more coherent and engaging video content.
"""

import os
import json
import logging
import re
from typing import Dict, Any
from agent.decision_logger import get_decision_logger
from agent.visual_planning.visual_story_planner import run as create_visual_story_plan

def run(run_dir: str, article_text: str, headline: str, logger: 'DecisionLogger') -> bool:
    """
    Creates a script using the new visual story planning approach.
    
    This now:
    1. Analyzes article for visual opportunities
    2. Creates a visual story plan
    3. Engineers a script to trigger those planned visuals
    4. Saves both the plan and script for downstream components
    """
    try:
        logging.info("=== SCRIPT WRITING (WITH VISUAL STORY PLANNING) ===")
        
        # Phase 1: Create visual story plan (this is the new approach)
        visual_story_plan, engineered_script, _ = create_visual_story_plan(
            run_dir, headline, article_text, logger
        )
        
        if not engineered_script or engineered_script.startswith("Failed"):
            logging.error("Visual story planning failed, falling back to basic script generation")
            # Fallback to basic script if visual planning fails
            engineered_script = _create_fallback_script(headline, article_text)
        
        # Clean the script to ensure it's properly formatted
        clean_script = _clean_script_text(engineered_script)
        
        # Save the engineered script
        script_path = os.path.join(run_dir, 'script.txt')
        clean_script_path = os.path.join(run_dir, 'script_clean.txt')
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(clean_script)
        
        with open(clean_script_path, 'w', encoding='utf-8') as f:
            f.write(clean_script)
        
        # Convert visual story plan to creative brief format for backward compatibility
        creative_brief = _convert_plan_to_brief(visual_story_plan, clean_script)
        brief_path = os.path.join(run_dir, 'creative_brief.json')
        with open(brief_path, 'w', encoding='utf-8') as f:
            json.dump(creative_brief, f, indent=2)
        
        # Log final script creation
        logger.log_decision(
            step="script_creation_complete",
            decision=f"Successfully created script using visual story planning",
            reasoning=f"Generated {len(clean_script.split())} word script engineered to trigger specific visual opportunities",
            confidence=0.95,
            metadata={
                "script_word_count": len(clean_script.split()),
                "estimated_duration": f"{len(clean_script.split()) * 0.4:.1f} seconds",
                "visual_beats_planned": len(visual_story_plan.get('visual_story_beats', [])),
                "script_file": script_path,
                "clean_script_file": clean_script_path,
                "creative_brief_file": brief_path,
                "approach": "visual_story_planning"
            }
        )
        
        logging.info(f"  > Script created with visual story planning: {len(clean_script.split())} words")
        logging.info(f"  > Saved script: {script_path}")
        logging.info(f"  > Saved clean script: {clean_script_path}")
        logging.info(f"  > Converted to creative brief: {brief_path}")
        
        return True
        
    except Exception as e:
        logger.log_decision(
            step="script_creation_error",
            decision="Script creation failed",
            reasoning=f"Error during script creation: {str(e)}",
            confidence=0.0,
            metadata={
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        logging.error(f"Script creation failed: {e}")
        return False

def _clean_script_text(script_text: str) -> str:
    """
    Cleans the script text to remove any unwanted formatting or stage directions.
    """
    # Remove any **bold** formatting that might be spoken
    script_text = re.sub(r'\*\*Narration Text:\*\*', '', script_text)
    script_text = re.sub(r'\*\*.*?\*\*', '', script_text)
    
    # Remove [stage directions] or similar
    script_text = re.sub(r'\[.*?\]', '', script_text)
    
    # Remove bullet points or numbered lists that might be spoken
    script_text = re.sub(r'^\s*[-•]\s*', '', script_text, flags=re.MULTILINE)
    script_text = re.sub(r'^\s*\d+\.\s*', '', script_text, flags=re.MULTILINE)
    
    # CRITICAL FIX: Remove metadata in parentheses at the end (Word count, duration, notes)
    # This pattern removes content like "(Word count: ~80 / Target duration: ~35 seconds) *Note: ..."
    script_text = re.sub(r'\s*\(Word count:.*?\)\s*\*?Note:.*$', '', script_text, flags=re.DOTALL)
    script_text = re.sub(r'\s*\(Word count:.*?\).*$', '', script_text, flags=re.DOTALL)
    
    # Also remove any asterisks used for emphasis
    script_text = re.sub(r'\*([^*]+)\*', r'\1', script_text)
    
    # Remove quotes at the beginning and end if present
    script_text = script_text.strip()
    if script_text.startswith('"') and script_text.endswith('"'):
        script_text = script_text[1:-1]
    if script_text.startswith('*"') and script_text.endswith('"*'):
        script_text = script_text[2:-2]
    
    # Normalize whitespace
    script_text = ' '.join(script_text.split())
    
    return script_text.strip()

def _create_fallback_script(headline: str, article_text: str) -> str:
    """
    Creates a basic fallback script if visual story planning fails.
    """
    # Extract key points from article
    words = article_text.split()
    if len(words) > 200:
        summary = ' '.join(words[:200]) + "..."
    else:
        summary = article_text
    
    # Create basic script structure
    script = f"Here's what you need to know about {headline.lower()}. "
    script += summary[:300]  # Limit length
    script += " What are your thoughts on this development?"
    
    return script

def _convert_plan_to_brief(visual_story_plan: Dict[str, Any], script_text: str) -> Dict[str, Any]:
    """
    Converts the visual story plan to creative brief format for backward compatibility.
    This ensures existing components can still work with the new approach.
    """
    try:
        # Extract story information
        story_hook = visual_story_plan.get('story_hook', {})
        script_structure = visual_story_plan.get('script_structure', {})
        visual_beats = visual_story_plan.get('visual_story_beats', [])
        
        # Convert to creative brief format
        creative_brief = {
            "story_angle": {
                "angle_name": story_hook.get('hook_concept', 'Technology Impact Story'),
                "description": f"Visual narrative with {len(visual_beats)} planned story beats",
                "target_emotion": story_hook.get('emotional_target', 'curiosity')
            },
            "script_outline": {
                "hook": script_text.split('.')[0] + '.' if '.' in script_text else script_text[:50],
                "main_points": [
                    {
                        "topic": f"Story Beat {beat.get('beat_number', i+1)}",
                        "talking_points": [beat.get('narrative_purpose', 'Visual story beat')]
                    } for i, beat in enumerate(visual_beats[:3])  # Limit to 3 main points
                ],
                "conclusion": script_text.split('.')[-1] if '.' in script_text else "What do you think?",
                "target_duration": script_structure.get('target_duration', '30-35 seconds'),
                "narrative_style": script_structure.get('narrative_style', 'energetic'),
                "pacing_strategy": script_structure.get('pacing', 'fast-paced')
            },
            "visual_cues": [
                {
                    "type": beat.get('visual_type', 'Abstract Concept'),
                    "description": beat.get('visual_description', 'Visual content')
                } for beat in visual_beats
            ],
            "visual_strategy": {
                "pacing_strategy": script_structure.get('pacing', 'fast-paced'),
                "style_suggestion": "dynamic",
                "opening_strategy": {
                    "style": "animated_scroll",
                    "transition_out": "zoom_out",
                    "screenshot_duration": 3.5
                }
            }
        }
        
        return creative_brief
        
    except Exception as e:
        logging.error(f"Failed to convert plan to brief: {e}")
        # Return minimal creative brief
        return {
            "story_angle": {
                "angle_name": "Technology Story",
                "description": "Generated from visual story planning",
                "target_emotion": "curiosity"
            },
            "script_outline": {
                "hook": script_text[:50] if script_text else "Technology story",
                "main_points": [{"topic": "Main Content", "talking_points": ["Key information"]}],
                "conclusion": "What are your thoughts?",
                "target_duration": "30-35 seconds",
                "narrative_style": "energetic",
                "pacing_strategy": "fast-paced"
            },
            "visual_cues": [
                {"type": "Abstract Concept", "description": "Technology visualization"}
            ],
            "visual_strategy": {
                "pacing_strategy": "fast-paced",
                "style_suggestion": "dynamic",
                "opening_strategy": {
                    "style": "animated_scroll",
                    "transition_out": "zoom_out",
                    "screenshot_duration": 3.5
                }
            }
        }

 