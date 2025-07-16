# AutoSocialMedia - AI-Powered Social Media Video Generation

## Overview

AutoSocialMedia is an end-to-end pipeline for automatically generating professional social media videos from trending news articles. The system combines AI-powered content discovery, visual direction, and video assembly to create engaging short-form videos optimized for various social platforms.

## Features

- **Automated Content Discovery**: Finds trending headlines from multiple news sources
- **Intelligent Visual Direction**: AI-powered visual asset selection and timing
- **Multi-Platform Optimization**: Portrait, landscape, and square video formats
- **Word-Level Synchronization**: Precise audio-visual timing with Whisper integration
- **Professional Effects**: Ken Burns effects, smooth transitions, and modern captions
- **Flexible TTS Integration**: Support for Kokoro and Google TTS providers

## Pipeline Architecture

The pipeline consists of 9 sequential components:

1. **discover** → Find trending news headlines
2. **scrape** → Extract article content and metadata
3. **screenshot** → Capture webpage visuals and opening videos
4. **script** → Generate AI-powered narration scripts
5. **audio** → Convert scripts to speech using TTS
6. **timing_extraction** → Extract word-level timestamps
7. **visual_director** → Orchestrate visual asset acquisition
8. **slideshow** → Assemble final video with effects
9. **captions** → Add synchronized word-level captions

## Orientation Support

### Portrait Mode (1080x1920)
- **Platforms**: TikTok, Instagram Reels, YouTube Shorts
- **Aspect Ratio**: 9:16
- **Use Case**: Mobile-first vertical content

### Landscape Mode (1920x1080)
- **Platforms**: YouTube, Facebook, LinkedIn
- **Aspect Ratio**: 16:9
- **Use Case**: Desktop and traditional video platforms

### Square Mode (1080x1080)
- **Platforms**: Instagram posts, Twitter videos
- **Aspect Ratio**: 1:1
- **Use Case**: Social media posts and previews

## Timeline Expectations

- **Video Duration**: 30-60 seconds based on script length
- **Opening Sequence**: 3-second webpage capture video
- **Visual Segments**: 2-8 seconds each with AI-synchronized timing
- **Transition Duration**: 1.0 second between segments
- **Audio Synchronization**: Word-level timestamp alignment (2.5 words/second)

## Configuration

### Required Files

- **config.ini**: Main configuration file
  ```ini
  [audio]
  tts_provider = kokoro

  [api_keys]
  # Add your API keys here
  ```

### Key Modules

- **agent/video_config.py**: Video parameters and orientation settings
- **agent/component_runner.py**: Pipeline orchestration and execution
- **agent/decision_logger.py**: AI decision tracking and logging
- **agent/slideshow/**: Video assembly and effects processing

## Quick Start

```python
from agent import create_smart_video

# Generate a video with default settings
video_path = create_smart_video()

if video_path:
    print(f"Video generated: {video_path}")
else:
    print("Pipeline failed - check logs for details")
```

## Development Setup

### Prerequisites

- Python 3.8+
- FFmpeg (installed and in PATH)
- Valid API keys for visual sources
- ~500MB disk space for temporary files
- (Optional) AntiCaptcha API key for CAPTCHA solving

### Installation

```bash
# Clone repository
git clone [repository-url]
cd AutoSocialMedia

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp config.ini.example config.ini
# Edit config.ini with your API keys
```

## Error Handling

The pipeline includes comprehensive error handling:

- **Component Isolation**: Failures don't cascade between components
- **Graceful Fallbacks**: Non-critical components (like timing extraction) can fail
- **Detailed Logging**: All decisions and errors are logged for debugging
- **Validation**: Input/output validation at each pipeline stage
- **Verification Bypass**: Automatic handling of Cloudflare and CAPTCHA challenges using open-source Buster extension

## Contributing

### Code Style

- Follow existing patterns for module documentation
- Use Google-style docstrings for all public functions
- Include type hints for function parameters and returns
- Add comprehensive error handling with specific exception types

### Testing Individual Components

```python
from agent.component_runner import ComponentRunner
from agent.decision_logger import DecisionLogger

# Test a specific component
logger = DecisionLogger("runs/test")
runner = ComponentRunner("runs/test", logger)
success = runner.run_component("discover")
```

### Adding New Components

1. Create component module in appropriate subdirectory
2. Implement `run()` function with proper return values
3. Add component specification to `agent/component_registry.py`
4. Update `ComponentRunner._execute_component_by_name()` dispatcher
5. Add comprehensive docstrings and error handling

## File Structure

```
runs/current/                 # Current run directory
├── headline.json            # Discovered headline data
├── article.txt             # Scraped article content
├── script_clean.txt        # Generated narration script
├── voice.mp3              # TTS audio output
├── transcript_data.json   # Word-level timestamps
├── visual_map.json        # Visual timeline and assets
├── slideshow.mp4          # Video without captions
└── final_video.mp4        # Complete video with captions
```

## Performance Considerations

- **Parallel Processing**: Visual acquisition runs in parallel where possible
- **Caching**: Intermediate files are preserved for debugging and reuse
- **Memory Management**: Large files are processed in chunks
- **Timeout Handling**: Network operations have reasonable timeouts

## Known Limitations

- Requires stable internet connection for content discovery and visual assets
- Processing time: 2-5 minutes per video depending on complexity
- FFmpeg dependency for video processing
- Limited to English content currently

## License

[Add your license information here]

## Support

For issues and questions:
1. Check the logs in `runs/current/run.log`
2. Verify all dependencies are installed
3. Ensure API keys are configured correctly
4. Review component-specific documentation in source files
