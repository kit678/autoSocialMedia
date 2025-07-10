import os
import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
from agent.utils import http_retry_session, rate_limit_gemini
from agent.decision_logger import get_decision_logger
import base64

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

CONTENT_ASSESSMENT_PROMPT = """
You are a content strategist analyzing articles to create compelling social media video narratives.

Given an article's text content, your task is to:

1. **Extract the Core Story Angle**: Identify the single most compelling narrative from the article that will resonate on social media.

2. **Create Script Outline**: Develop a structured outline that will guide script generation with clear talking points and narrative flow.

Return a JSON response with this structure:
{
  "story_angle": {
    "angle_name": "Specific angle name",
    "description": "Why this angle is compelling for social media",
    "key_points": ["Point 1", "Point 2", "Point 3"],
    "target_emotion": "What emotion this should evoke",
    "viral_potential": "Why this will perform well",
    "rationale": "Detailed reasoning for selecting this angle over others"
  },
  "script_outline": {
    "hook": "Opening line or concept to grab attention (5-8 seconds)",
    "main_points": [
      {
        "topic": "Core message 1",
        "talking_points": ["Specific point A", "Specific point B"],
        "emotional_tone": "informative/concerned/hopeful",
        "duration_target": "8-12 seconds"
      },
      {
        "topic": "Core message 2", 
        "talking_points": ["Specific point C", "Specific point D"],
        "emotional_tone": "informative/concerned/hopeful",
        "duration_target": "8-12 seconds"
      }
    ],
    "conclusion": "Closing message or call to action (5-8 seconds)",
    "key_phrases": ["Important phrases that should appear in script for visual timing"],
    "target_duration": "25-35 seconds",
    "narrative_style": "conversational/informative/dramatic"
  },
  "article_image_analysis": [
    {
      "index": 0,
      "relevance_to_story": "How this image relates to the chosen angle",
      "potential_usage": "How this could be used in the video"
    }
  ]
}

Focus on creating a clear narrative structure that will guide both script writing and subsequent visual planning.

The script outline should provide enough detail to write a compelling 25-35 second script while identifying key phrases that will later be used for visual timing."""

ARTICLE_IMAGE_PLACEMENT_PROMPT = """
Given the article images and the script, determine the optimal placement for each article image within the video narrative.

Consider:
- Which part of the script each image best illustrates
- Natural flow and pacing
- Visual variety and engagement

Return placement recommendations for each article image.
"""

@rate_limit_gemini
def run(article_text: str, image_urls: List[str], screenshot_path: str, headline_data: Dict) -> Dict[str, Any]:
    """
    Analyzes content to determine the best story angle and visual strategy.
    
    Args:
        article_text: The article content to analyze
        image_urls: List of image URLs from the article  
        screenshot_path: Path to the webpage screenshot
        headline_data: Article headline and metadata
        
    Returns:
        Content assessment with story angle and visual strategy
    """
    # Initialize decision logger
    logger = get_decision_logger()
    logger.start_component("content_assessment")
    
    try:
        logger.log_decision(
            step="input_analysis",
            decision="Analyzing article for content assessment",
            reasoning=f"Processing {len(article_text)} characters of text with {len(image_urls)} existing images",
            input_data={
                "article_length": len(article_text),
                "image_count": len(image_urls),
                "headline": headline_data.get('title', 'No title')
            }
        )
        
        if not GEMINI_API_KEY:
            error_msg = "CRITICAL ERROR: GEMINI_API_KEY not found in environment"
            logging.error(error_msg)
            logger.log_decision(
                step="api_key_missing",
                decision="Cannot proceed without Gemini API key",
                reasoning="Content assessment requires AI analysis",
                confidence=0.0
            )
            raise Exception(error_msg + " Pipeline configured to crash instead of using fallback assessment.")

        # Configure Gemini - Create client with the new google-genai SDK
        client = genai.Client(api_key=GEMINI_API_KEY)
        model_name = 'gemini-2.5-flash-lite-preview-06-17'
        
        # Prepare content for analysis
        content_parts = []
        content_parts.append(types.Part.from_text(text=CONTENT_ASSESSMENT_PROMPT))
        content_parts.append(types.Part.from_text(text=f"\nArticle Title: {headline_data.get('title', 'No title')}"))
        content_parts.append(types.Part.from_text(text=f"\nArticle Content:\n{article_text}"))
        
        # Add screenshot if available
        if screenshot_path and os.path.exists(screenshot_path):
            try:
                with open(screenshot_path, 'rb') as f:
                    screenshot_data = f.read()
                content_parts.append(types.Part.from_bytes(
                    data=screenshot_data,
                    mime_type='image/png'
                ))
                
                logger.log_decision(
                    step="screenshot_inclusion",
                    decision="Including webpage screenshot in analysis",
                    reasoning="Screenshot provides visual context for layout and design elements",
                    confidence=0.8
                )
            except Exception as e:
                error_msg = f"Failed to load screenshot: {e}"
                logging.error(error_msg)
                logger.log_decision(
                    step="screenshot_error",
                    decision="Screenshot loading failed - aborting",
                    reasoning=error_msg,
                    confidence=0.0
                )
                raise Exception(error_msg + " Pipeline configured to crash instead of proceeding without screenshot.")
        
        # Add article images for analysis
        article_images_added = 0
        for i, img_url in enumerate(image_urls[:3]):  # Limit to first 3 images
            try:
                img_path = _download_temp_image(img_url, i)
                if img_path:
                    with open(img_path, 'rb') as f:
                        img_data = f.read()
                    content_parts.append(types.Part.from_bytes(
                        data=img_data,
                        mime_type='image/jpeg'
                    ))
                    article_images_added += 1
                    os.remove(img_path)  # Clean up
            except Exception as e:
                logging.warning(f"Could not add article image {i}: {e}")
        
        logger.log_decision(
            step="article_images_inclusion", 
            decision=f"Added {article_images_added} article images to analysis",
            reasoning=f"Including existing images helps determine visual gaps and usage strategy",
            confidence=0.8 if article_images_added > 0 else 0.3
        )

        # Request assessment from Gemini
        logger.log_decision(
            step="gemini_request",
            decision="Sending content assessment request to Gemini",
            reasoning="Using AI analysis to determine optimal story angle and visual strategy",
            input_data={
                "content_parts_count": len(content_parts),
                "has_screenshot": any(isinstance(p, dict) for p in content_parts),
                "article_images_included": article_images_added
            }
        )

        response = client.models.generate_content(
            model=model_name,
            contents=content_parts
        )
        
        if not response.text:
            logger.log_decision(
                step="api_error",
                decision="Gemini returned empty response - aborting pipeline",
                reasoning="No content returned from API",
                confidence=0.0
            )
            raise Exception("Gemini API returned empty response")

        # Parse Gemini response
        try:
            # Strip markdown formatting if present
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove ```json
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove closing ```
            response_text = response_text.strip()
            
            assessment = json.loads(response_text)
            
            # Log the story angle decision
            story_angle = assessment.get('story_angle', {})
            logger.log_decision(
                step="story_angle_selection",
                decision=story_angle.get('angle_name', 'Unknown angle'),
                reasoning=story_angle.get('rationale', 'No reasoning provided'),
                confidence=0.9,
                metadata={
                    "target_emotion": story_angle.get('target_emotion'),
                    "viral_potential": story_angle.get('viral_potential'),
                    "key_points": story_angle.get('key_points', [])
                }
            )
            
            # Log script outline decisions
            script_outline = assessment.get('script_outline', {})
            logger.log_decision(
                step="script_outline_creation",
                decision=f"Created script outline with {len(script_outline.get('main_points', []))} main points",
                reasoning="Script outline will guide both script generation and visual planning",
                confidence=0.9,
                metadata={
                    "hook": script_outline.get('hook'),
                    "conclusion": script_outline.get('conclusion'),
                    "target_duration": script_outline.get('target_duration'),
                    "narrative_style": script_outline.get('narrative_style'),
                    "key_phrases_count": len(script_outline.get('key_phrases', []))
                }
            )
            
            logger.log_decision(
                step="assessment_completion",
                decision="Successfully completed content assessment",
                reasoning=f"Generated script outline with {len(script_outline.get('main_points', []))} main points",
                confidence=0.9
            )
            
        except json.JSONDecodeError as e:
            logger.log_decision(
                step="parse_error",
                decision="Failed to parse Gemini JSON response - aborting pipeline",
                reasoning=f"JSON parsing failed: {e}",
                confidence=0.0,
                metadata={"response_text": response.text[:500] if response.text else None}
            )
            raise Exception(f"Failed to parse Gemini JSON response: {e}") from e
        
        return assessment
            
    except Exception as e:
        logger.log_decision(
            step="api_error",
            decision="Gemini API call failed - aborting pipeline",
            reasoning=f"Critical error: {e}",
            confidence=0.0,
            metadata={"error_type": type(e).__name__, "error_details": str(e)}
        )
        # Re-raise the exception to crash the pipeline as requested
        raise Exception(f"Gemini API call failed: {e}") from e
    
    finally:
        logger.finish_component()

# gather_visuals_for_angle function and related helpers REMOVED 
# Visual gathering now handled entirely by Step 5 (visual_gathering component)

def _download_temp_image(img_url: str, index: int) -> str:
    """Download and standardize image temporarily for analysis."""
    try:
        import tempfile
        from agent.utils import download_and_standardize_image
        
        # Create temporary file with standardized extension
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_file.close()  # Close file so it can be written to by download_and_standardize_image
        
        # Download and standardize the image
        if download_and_standardize_image(img_url, temp_file.name):
            return temp_file.name
        else:
            # Clean up if download failed
            import os
            try:
                os.unlink(temp_file.name)
            except:
                pass
            return None
        
    except Exception as e:
        logging.warning(f"Failed to download and standardize temp image {index}: {e}")
        return None 