import os
import logging
import time
import google.generativeai as genai

# Configure the Gemini API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logging.warning("GEMINI_API_KEY not set. Google TTS will not be available.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

TTS_MODEL = "gemini-2.5-flash-preview-tts"

def attempt_tts_generation(text: str, style_prompt: str, temperature: float, output_path: str, voice_name: str) -> bool:
    """Attempts to generate TTS audio with specific parameters, returns success status."""
    try:
        logging.info("  > [Google] Generating audio with Gemini...")
        logging.info(f"    - Voice: {voice_name}, Temperature: {temperature}")
        
        model = genai.GenerativeModel(TTS_MODEL)
        full_prompt = f"{style_prompt} {text}" if style_prompt else text
        
        # Use dictionaries for configuration to avoid import issues
        speech_config = {
            "voice_config": {
                "prebuilt_voice_config": {
                    "voice_name": voice_name
                }
            }
        }
        
        generation_config = {
            "temperature": temperature,
            "response_modalities": ["AUDIO"],
            "speech_config": speech_config
        }
        
        result = model.generate_content(
            full_prompt,
            generation_config=generation_config,
            stream=True
        )
        
        with open(output_path, "wb") as f:
            for chunk in result:
                if chunk.candidates and chunk.candidates[0].content.parts:
                    f.write(chunk.candidates[0].content.parts[0].inline_data.data)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logging.info(f"  > [Google] Audio generated successfully: {output_path}")
            return True
        else:
            logging.warning("[Google] Audio generation resulted in an empty file.")
            return False

    except Exception as e:
        logging.error(f"[Google] An unexpected error occurred during TTS generation: {e}")
    
    return False

def _analyze_content_sentiment(text: str) -> dict:
    """Analyzes text to determine voice style. Mimics logic from Kokoro module."""
    text_lower = text.lower()
    if any(p in text_lower for p in ['breaking', 'urgent', 'critical']):
        return {"sentiment": "urgent"}
    if any(p in text_lower for p in ['crisis', 'disaster', 'warning']):
        return {"sentiment": "negative"}
    if any(p in text_lower for p in ['breakthrough', 'success', 'exciting']):
        return {"sentiment": "positive"}
    return {"sentiment": "neutral"}

def _get_voice_config(sentiment: str) -> dict:
    """Selects Google TTS voice and style based on sentiment."""
    base_style = "Speak with an energetic, fast-paced, and engaging delivery. Sound like a captivating social media narrator."
    
    if sentiment == "urgent":
        return {
            "voice_name": "Kore",
            "style_prompt": f"{base_style} This is breaking news. Be authoritative and clear.",
            "temperature": 0.7
        }
    elif sentiment == "negative":
        return {
            "voice_name": "Charon",
            "style_prompt": f"{base_style} Use a serious but compelling tone for this important story.",
            "temperature": 0.6
        }
    elif sentiment == "positive":
        return {
            "voice_name": "Puck",
            "style_prompt": f"{base_style} Sound enthusiastic and upbeat for this exciting news!",
            "temperature": 0.9
        }
    else:  # neutral
        return {
            "voice_name": "Zephyr",
            "style_prompt": base_style,
            "temperature": 0.8
        }

def run(script_text: str, output_path: str) -> bool:
    """
    Generates audio from text using Google's Gemini TTS.
    Note: Free tier has very strict rate limits (3 RPM, 15 RPD).
    """
    if not GEMINI_API_KEY:
        logging.error("[Google] Cannot proceed with TTS generation because GEMINI_API_KEY is not set.")
        return False
        
    logging.warning("Using Google TTS. Note: The free tier is subject to VERY STRICT rate limits (3 RPM, 15 RPD).")
    logging.warning("If the pipeline fails with a 429 error, wait a few minutes or use the 'kokoro' provider.")

    # Analyze content to select the best voice and style
    analysis = _analyze_content_sentiment(script_text)
    config = _get_voice_config(analysis['sentiment'])

    logging.info(f"  > [Google] Sentiment: {analysis['sentiment']}. Using voice: {config['voice_name']}")

    for attempt in range(3):
        logging.info(f"  > [Google] Attempt {attempt + 1}/3...")
        success = attempt_tts_generation(
            text=script_text,
            style_prompt=config['style_prompt'],
            temperature=config['temperature'],
            output_path=output_path,
            voice_name=config['voice_name']
        )
        if success:
            return True
        
        # Exponential backoff for rate limiting
        wait_time = 10 * (2 ** attempt)
        logging.info(f"  > [Google] TTS attempt failed. Waiting for {wait_time} seconds before retrying...")
        time.sleep(wait_time)

    logging.error("[Google] All TTS generation attempts failed.")
    return False 