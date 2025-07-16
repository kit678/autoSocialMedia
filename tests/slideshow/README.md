# Slideshow Component Tests

This directory contains pytest unit and integration tests for the slideshow component of the AutoSocialMedia pipeline.

## Test Coverage

### `test_slideshow.py`

The test file includes the following test cases:

1. **`test_constant_config()`** - Verifies that video configuration (width/height) is properly propagated through the system. Tests that landscape mode correctly sets 1920x1080 resolution.

2. **`test_build_filter_single()`** - Tests that a single-image timeline builds a valid filter graph. Validates that the timeline processing works correctly for a simple slideshow with one image.

3. **`test_build_filter_multi()`** - Tests multi-image timelines with transitions. Ensures that multiple images are processed correctly and that each segment has valid duration frames.

4. **`test_mixed_media()`** - Verifies that the slideshow component can handle mixed media (images and videos) without NameError exceptions and that the FFmpeg command is assembled correctly. This test mocks the `run_command` to avoid actual FFmpeg execution.

5. **`test_ffmpeg_command_structure()`** - Comprehensive test of the FFmpeg command structure. Verifies that:
   - The command starts with 'ffmpeg'
   - Contains proper flags (-y, -filter_complex)
   - Includes audio input mapping
   - Has correct video/audio codec settings (libx264, aac)
   - Maps outputs correctly ([vout])

6. **`test_slideshow_component_adapted()`** - Integration test adapted from the existing `test_slideshow_component`. Tests the complete flow with mocked dependencies to ensure backward compatibility with the existing API.

## Running the Tests

To run all slideshow tests:
```bash
python -m pytest tests/slideshow/test_slideshow.py -v
```

To run a specific test:
```bash
python -m pytest tests/slideshow/test_slideshow.py::test_mixed_media -v
```

## Test Implementation Notes

- All tests use mocking to avoid file system dependencies and actual FFmpeg execution
- Temporary files are properly cleaned up using try/finally blocks to avoid Windows file locking issues
- The tests validate both the internal logic and the external command generation

## Known Issues

The actual slideshow component has a bug where it tries to use `-loop 1` with video files (e.g., webpage_capture.mp4), which is not valid. The `-loop` option only works with image files in FFmpeg. This needs to be fixed in the main component code.
