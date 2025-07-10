import logging
from . import google_tts
from . import kokoro_tts

def run(script_text: str, output_path: str, tts_provider: str) -> bool:
    """
    Dispatcher function for TTS generation.
    
    Args:
        script_text (str): The text to synthesize.
        output_path (str): The path to save the final .mp3 file.
        tts_provider (str): The TTS provider to use ('google' or 'kokoro').
        
    Returns:
        bool: True on success, False on failure.
    """
    logging.info(f"--- Running Audio Generation with provider: {tts_provider} ---")
    
    if tts_provider == 'google':
        return google_tts.run(script_text, output_path)
    elif tts_provider == 'kokoro':
        return kokoro_tts.run(script_text, output_path)
    else:
        logging.error(f"Unknown TTS provider specified: '{tts_provider}'. Please use 'google' or 'kokoro'.")
        return False 