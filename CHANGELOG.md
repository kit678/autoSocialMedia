# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Updated `create_smart_video.run()` call signature to accept optional `config` parameter
  - **Migration Note**: Call sites of `create_smart_video.run()` should now pass a `config` parameter or explicitly use `get_default_config("landscape")` as the default
  - The internal function already uses `get_default_config("landscape")` if no config is passed
  - Updated call sites in:
    - `agent/component_runner.py`: Added `config=get_default_config("landscape")` 
    - `test_slideshow.py`: Added `config=get_default_config("landscape")`
    - `tests/slideshow/test_slideshow.py`: Added `config=get_default_config("landscape")`

### Improved
- Video configuration is now more explicit at call sites, making it easier to customize video output settings

## Example Migration

Before:
```python
video_path = create_smart_video(
    visual_analysis=visual_analysis,
    all_image_paths=all_image_paths,
    audio_path=audio_path,
    audio_duration=audio_duration,
    output_path=output_path
)
```

After:
```python
from agent.video_config import get_default_config

video_path = create_smart_video(
    visual_analysis=visual_analysis,
    all_image_paths=all_image_paths,
    audio_path=audio_path,
    audio_duration=audio_duration,
    output_path=output_path,
    config=get_default_config("landscape")  # Or pass custom VideoConfig
)
```
