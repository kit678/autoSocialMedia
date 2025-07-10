import os
import logging
import soundfile as sf
from kokoro import KPipeline
from agent.utils import run_command
import numpy as np
import re

def analyze_content_sentiment(text: str) -> dict:
    """
    Analyzes text content to determine appropriate voice configuration.
    Returns voice style parameters based on content sentiment and topic.
    """
    text_lower = text.lower()
    
    # Define sentiment patterns
    negative_patterns = [
        'crisis', 'disaster', 'warning', 'danger', 'threat', 'collapse', 'decline', 
        'failure', 'unemployment', 'recession', 'war', 'conflict', 'controversy',
        'scandal', 'fraud', 'corruption', 'death', 'tragedy', 'attack', 'violence'
    ]
    
    positive_patterns = [
        'breakthrough', 'success', 'achievement', 'innovation', 'growth', 'opportunity',
        'celebration', 'victory', 'advancement', 'progress', 'solution', 'recovery',
        'benefit', 'improvement', 'exciting', 'amazing', 'fantastic', 'wonderful'
    ]
    
    urgent_patterns = [
        'breaking', 'urgent', 'immediate', 'critical', 'emergency', 'alert',
        'must know', 'happening now', 'last minute', 'just in'
    ]
    
    tech_patterns = [
        'ai', 'artificial intelligence', 'technology', 'software', 'algorithm',
        'data', 'programming', 'digital', 'cyber', 'tech', 'startup', 'innovation'
    ]
    
    # Count sentiment indicators
    negative_count = sum(1 for pattern in negative_patterns if pattern in text_lower)
    positive_count = sum(1 for pattern in positive_patterns if pattern in text_lower)
    urgent_count = sum(1 for pattern in urgent_patterns if pattern in text_lower)
    tech_count = sum(1 for pattern in tech_patterns if pattern in text_lower)
    
    # Determine dominant sentiment
    if urgent_count > 0:
        sentiment = "urgent"
    elif negative_count > positive_count:
        sentiment = "negative"
    elif positive_count > negative_count:
        sentiment = "positive"
    else:
        sentiment = "neutral"
    
    # Determine topic category
    if tech_count > 2:
        topic = "tech"
    else:
        topic = "general"
    
    return {
        "sentiment": sentiment,
        "topic": topic,
        "negative_count": negative_count,
        "positive_count": positive_count,
        "urgent_count": urgent_count,
        "tech_count": tech_count
    }

def get_voice_config(content_analysis: dict, script_text: str) -> dict:
    """
    Returns optimized Kokoro voice configuration based on content analysis.
    Maps content sentiment to appropriate Kokoro voices.
    """
    sentiment = content_analysis["sentiment"]
    topic = content_analysis["topic"]
    
    voice_config = {
        "voice_name": "af_heart",  # Default warm, professional voice
        "language_code": "a",      # American English
        "speed": 1.0               # Default speech rate
    }

    # Determine voice based on content
    if sentiment == "negative":
        voice_config["voice_name"] = "am_adam" # Serious, deep voice
        voice_config["speed"] = 0.95
    elif topic == "tech":
        voice_config["voice_name"] = "af_sarah" # Clear, professional tech voice
        voice_config["speed"] = 1.15
    elif topic == "entertainment":
        voice_config["voice_name"] = "af_sky" # Energetic and lively
        voice_config["speed"] = 1.2
    elif sentiment == "positive":
        voice_config["voice_name"] = "af_nicole" # Upbeat and friendly
        voice_config["speed"] = 1.1
    else:
        voice_config["voice_name"] = "af_sarah" # Default neutral/informative voice
        voice_config["speed"] = 1.1

    return voice_config

def run(script_text: str, output_mp3_path: str) -> bool:
    """
    Generates audio from text using Kokoro TTS and saves it as an MP3 file.
    """
    wav_path = output_mp3_path.replace('.mp3', '_temp.wav')
    try:
        content_analysis = analyze_content_sentiment(script_text)
        voice_config = get_voice_config(content_analysis, script_text)
        
        logging.info(f"  > [Kokoro] Content analysis: {content_analysis['sentiment']} sentiment, {content_analysis['topic']} topic")
        logging.info(f"  > [Kokoro] Selected voice: {voice_config['voice_name']} with speed {voice_config['speed']}")

        pipeline = KPipeline(lang_code=voice_config['language_code'])
        
        generator = pipeline(
            script_text, 
            voice=voice_config['voice_name'],
            speed=voice_config['speed']
        )
        
        audio_chunks = [audio for i, (gs, ps, audio) in enumerate(generator) if audio is not None]
        
        if not audio_chunks:
            logging.error("[Kokoro] No audio chunks generated")
            return False
        
        full_audio = np.concatenate(audio_chunks)
        sf.write(wav_path, full_audio, 24000)
        
        ffmpeg_cmd = ['ffmpeg', '-i', wav_path, '-c:a', 'libmp3lame', '-b:a', '128k', output_mp3_path, '-y']
        success, _, stderr = run_command(ffmpeg_cmd)
        
        if not success:
            logging.error(f"[Kokoro] FFmpeg conversion failed. Stderr: {stderr}")
            return False

        logging.info(f"  > [Kokoro] TTS completed successfully.")
        return True
        
    except Exception as e:
        logging.error(f"[Kokoro] TTS generation failed: {str(e)}")
        return False
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path) 