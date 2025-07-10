"""
Visual Story Planner

This component analyzes article content to create a comprehensive visual narrative plan
BEFORE script writing. This enables reverse-engineering the script to ensure every visual
serves the story and creates coherent, engaging video content.
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
import requests

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

VISUAL_STORY_ANALYSIS_PROMPT = """
You are an expert video storyteller and visual director. Analyze this article to create a comprehensive VISUAL NARRATIVE PLAN that will guide script writing.

Your goal is to identify the best visual opportunities in the article and design a story arc that maximizes visual engagement.

ARTICLE HEADLINE: {headline}
ARTICLE TEXT:
{article_text}

Create a detailed visual story plan as JSON with the following structure:

{{
  "story_hook": {{
    "opening_visual": "<describe the perfect opening visual that grabs attention>",
    "hook_concept": "<the main hook/angle for the video>",
    "emotional_target": "<primary emotion to evoke: curiosity/urgency/excitement/concern>"
  }},
  "visual_segments": [
    {{
      "segment_number": 1,
      "narrative_purpose": "<what this segment accomplishes in the story>",
      "visual_type": "Proper Noun|Concrete Object/Action|Abstract Concept",
      "primary_search_term": "<2-4 word primary search query for this visual>",
      "secondary_keywords": ["<supporting keyword 1>", "<supporting keyword 2>"],
      "narrative_context": "<Brief sentence describing the visual's purpose for validation>",
      "script_guidance": "<how to write the narration to naturally trigger this visual>",
      "emotional_tone": "<tone for this segment>",
      "duration_target": "<suggested seconds for this segment>"
    }}
  ],
  "visual_entities": {{
    "companies": ["<list of companies/organizations with strong visual potential>"],
    "people": ["<list of key people mentioned>"],
    "products_services": ["<list of products/services that can be visualized>"],
    "locations": ["<list of relevant locations>"],
    "technologies": ["<list of technologies/concepts with visual potential>"]
  }},
  "visual_metaphors": {{
    "primary_concepts": [
      {{
        "concept": "<abstract concept from article>",
        "visual_metaphor": "<concrete visual metaphor to represent it>",
        "search_terms": ["<specific search terms for this visual>"]
      }}
    ]
  }},
  "script_structure": {{
    "target_duration": "<ideal video length in seconds>",
    "pacing": "fast-paced|medium|varied",
    "narrative_style": "<energetic|conversational|authoritative|investigative>",
    "call_to_action": "<suggested ending/CTA>"
  }},
  "visual_variety_plan": {{
    "proper_noun_count": "<number of company/person visuals planned>",
    "concrete_action_count": "<number of action/object visuals planned>", 
    "abstract_concept_count": "<number of concept visuals planned>",
    "total_visual_segments": "<total number of individual visual segments planned>"
  }}
}}

CRITICAL REQUIREMENTS:
1. Plan 8-12 distinct visual segments (NOT beats with multiple images) that tell a complete story.
2. Each segment should have ONE specific visual with its own search terms and duration.
3. For each segment, provide a short `primary_search_term` and `secondary_keywords`.
4. The `narrative_context` MUST be a concise sentence explaining the visual's story purpose.
5. Ensure good mix of visual types (not all abstract concepts).
6. Identify specific entities that have strong visual search potential.
7. Create visual metaphors that are searchable and compelling.
8. Design the story flow to build engagement and maintain interest.
9. Consider what visuals will actually be findable via Pexels/SearXNG/AI generation.
10. Each segment should be 2-8 seconds long for dynamic pacing.

Focus on creating a COHESIVE VISUAL NARRATIVE with individual, timed segments rather than grouped story beats.
"""

SCRIPT_ENGINEERING_PROMPT = """
You are a viral video scriptwriter who writes narration that intentionally creates specific visual opportunities.

You have been given a VISUAL STORY PLAN that maps out exactly what visuals should appear and when.
Your job is to write narration that naturally triggers these planned visuals.

VISUAL STORY PLAN:
{visual_story_plan}

ARTICLE CONTENT:
{article_text}

Write a compelling script that:
1. Follows the planned story beats exactly
2. Uses language that naturally triggers the planned visual types
3. Incorporates the identified entities and concepts organically
4. Maintains the specified pacing and emotional tone
5. Stays within the target duration (aim for 60-80 words for 30-35 seconds)

VISUAL TRIGGERING GUIDELINES:
- For Proper Nouns: Mention company/person names directly ("Companies like [Name]...")
- For Concrete Actions: Use active, visual language ("drowning in alerts", "AI agents automate...")  
- For Abstract Concepts: Use metaphorical language that triggers visual searches ("reclaiming time", "human-like reasoning")

Write ONLY the narration text. No stage directions or formatting.
The script should flow naturally while hitting every planned visual beat.
"""

def analyze_visual_story_potential(headline: str, article_text: str) -> Dict[str, Any]:
    """
    Analyzes article content to create a comprehensive visual narrative plan.
    This happens BEFORE script writing to ensure visuals drive the story.
    """
    logging.info("=== VISUAL STORY PLANNING ===")
    logging.info("  > Analyzing article for visual story potential...")
    
    prompt = VISUAL_STORY_ANALYSIS_PROMPT.format(
        headline=headline,
        article_text=article_text[:5000]  # Limit for API
    )
    
    try:
        response = requests.post(
            'https://api.deepseek.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'deepseek-chat',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.7,
                'response_format': {'type': 'json_object'}
            }
        )
        response.raise_for_status()
        
        visual_story_plan = json.loads(response.json()['choices'][0]['message']['content'])
        
        logging.info(f"  > Created visual story plan with {len(visual_story_plan.get('visual_segments', []))} story segments")
        logging.info(f"  > Identified {len(visual_story_plan.get('visual_entities', {}).get('companies', []))} companies for visual search")
        logging.info(f"  > Planned {visual_story_plan.get('visual_variety_plan', {}).get('total_visual_segments', 'unknown')} total visual segments")
        
        return visual_story_plan
        
    except Exception as e:
        logging.error(f"Visual story planning failed: {e}")
        # Return minimal fallback plan
        return {
            "story_hook": {
                "opening_visual": "Website screenshot with smart zoom",
                "hook_concept": "Technology impact story",
                "emotional_target": "curiosity"
            },
            "visual_segments": [
                {
                    "segment_number": 1,
                    "narrative_purpose": "Establish the problem",
                    "visual_type": "Abstract Concept", 
                    "primary_search_term": "technology challenges",
                    "secondary_keywords": ["visualization", "impact"],
                    "narrative_context": "Technology challenges visualization",
                    "script_guidance": "Start with the main challenge from the article",
                    "emotional_tone": "concern",
                    "duration_target": "8-10 seconds"
                }
            ],
            "visual_entities": {"companies": [], "people": [], "products_services": [], "locations": [], "technologies": []},
            "visual_metaphors": {"primary_concepts": []},
            "script_structure": {
                "target_duration": "30-35 seconds",
                "pacing": "fast-paced", 
                "narrative_style": "energetic",
                "call_to_action": "Thought-provoking question"
            },
            "visual_variety_plan": {
                "proper_noun_count": "1",
                "concrete_action_count": "1",
                "abstract_concept_count": "2",
                "total_visual_segments": "4"
            }
        }

def engineer_script_from_visual_plan(visual_story_plan: Dict[str, Any], article_text: str) -> str:
    """
    Writes a script that is specifically engineered to trigger the planned visuals.
    This is the reverse-engineering approach where visuals drive script content.
    """
    logging.info("  > Engineering script from visual story plan...")
    
    prompt = SCRIPT_ENGINEERING_PROMPT.format(
        visual_story_plan=json.dumps(visual_story_plan, indent=2),
        article_text=article_text[:3000]
    )
    
    try:
        response = requests.post(
            'https://api.deepseek.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'deepseek-chat',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.8
            }
        )
        response.raise_for_status()
        
        script = response.json()['choices'][0]['message']['content'].strip()
        
        # Clean the script of any stage directions or formatting
        script = re.sub(r'\*\*.*?\*\*', '', script)  # Remove **bold** formatting
        script = re.sub(r'\[.*?\]', '', script)      # Remove [stage directions]
        script = re.sub(r'^\s*-.*$', '', script, flags=re.MULTILINE)  # Remove bullet points
        script = ' '.join(script.split())            # Normalize whitespace
        
        logging.info(f"  > Generated script: {len(script.split())} words")
        
        return script
        
    except Exception as e:
        logging.error(f"Script engineering failed: {e}")
        return "Failed to generate script from visual plan. Please check the logs."

def run(run_dir: str, headline: str, article_text: str, logger: 'DecisionLogger') -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """
    Main function that creates a visual story plan and engineers a script to match.
    
    Returns:
        Tuple of (visual_story_plan, engineered_script, None) - search_strategy removed
    """
    try:
        # Log the start of visual story planning
        logger.log_decision(
            step="visual_story_planning_start",
            decision="Beginning proactive visual story planning",
            reasoning="Analyzing article to create visual narrative plan before script writing",
            confidence=1.0,
            metadata={
                "headline": headline,
                "article_length": len(article_text),
                "approach": "reverse_engineering"
            }
        )
        
        # Phase 1: Analyze visual story potential
        visual_story_plan = analyze_visual_story_potential(headline, article_text)
        
        # Log visual story plan creation
        logger.log_decision(
            step="visual_story_plan_created",
            decision=f"Created comprehensive visual story plan",
            reasoning=f"Identified {len(visual_story_plan.get('visual_segments', []))} story segments with mixed visual types",
            confidence=0.9,
            metadata={
                "story_segments": len(visual_story_plan.get('visual_segments', [])),
                "companies_identified": len(visual_story_plan.get('visual_entities', {}).get('companies', [])),
                "visual_metaphors": len(visual_story_plan.get('visual_metaphors', {}).get('primary_concepts', [])),
                "target_duration": visual_story_plan.get('script_structure', {}).get('target_duration', 'unknown')
            }
        )
        
        # Phase 2: Engineer script from visual plan
        engineered_script = engineer_script_from_visual_plan(visual_story_plan, article_text)
        
        # Log script engineering
        logger.log_decision(
            step="script_engineered",
            decision=f"Engineered script from visual plan",
            reasoning="Created narration specifically designed to trigger planned visual opportunities",
            confidence=0.9,
            metadata={
                "script_length": len(engineered_script.split()),
                "estimated_duration": f"{len(engineered_script.split()) * 0.4:.1f} seconds",
                "narrative_style": visual_story_plan.get('script_structure', {}).get('narrative_style', 'unknown')
            }
        )
        
        # Save visual story plan
        plan_path = os.path.join(run_dir, 'visual_story_plan.json')
        with open(plan_path, 'w', encoding='utf-8') as f:
            json.dump(visual_story_plan, f, indent=2)
        
        logging.info(f"  > Saved visual story plan: {plan_path}")
        
        return visual_story_plan, engineered_script, None
        
    except Exception as e:
        logger.log_decision(
            step="visual_story_planning_error",
            decision="Visual story planning failed",
            reasoning=f"Error during visual story planning: {str(e)}",
            confidence=0.0,
            metadata={
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        logging.error(f"Visual story planning failed: {e}")
        
        # Return minimal fallback
        return {}, "Failed to create visual story plan.", None 