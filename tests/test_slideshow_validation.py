"""
Unit tests for slideshow validation and error handling.

Tests the validation layer for missing files, zero duration, 
mismatched audio, and other error conditions.
"""

import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock

from agent.slideshow.exceptions import (
    SlideshowError,
    ValidationError,
    MissingFileError,
    InvalidDurationError,
    AudioMismatchError,
    FFmpegNotFoundError,
    TimelineValidationError
)
from agent.slideshow.validation import (
    validate_ffmpeg_availability,
    validate_file_exists,
    validate_timeline_segment,
    validate_timeline_segments,
    validate_audio_compatibility,
    validate_slideshow_inputs,
    validate_mixed_media_inputs
)


class TestExceptions:
    """Test custom exception classes."""
    
    def test_missing_file_error(self):
        """Test MissingFileError with custom message."""
        error = MissingFileError("/path/to/missing.jpg", "test file")
        assert error.file_path == "/path/to/missing.jpg"
        assert "test file" in str(error)
        assert "/path/to/missing.jpg" in str(error)
    
    def test_invalid_duration_error(self):
        """Test InvalidDurationError with segment info."""
        error = InvalidDurationError(-1.0, "segment_01")
        assert error.duration == -1.0
        assert error.segment_id == "segment_01"
        assert "segment_01" in str(error)
        assert "-1.0" in str(error)
    
    def test_audio_mismatch_error(self):
        """Test AudioMismatchError with expected and actual durations."""
        error = AudioMismatchError(10.0, 8.5)
        assert error.expected_duration == 10.0
        assert error.actual_duration == 8.5
        assert "10.0" in str(error)
        assert "8.5" in str(error)
    
    def test_timeline_validation_error(self):
        """Test TimelineValidationError with multiple issues."""
        issues = ["Missing file", "Invalid duration", "Bad format"]
        error = TimelineValidationError(issues)
        assert error.issues == issues
        for issue in issues:
            assert issue in str(error)


class TestValidation:
    """Test validation functions."""
    
    def test_validate_ffmpeg_availability_success(self):
        """Test FFmpeg availability validation when tools are available."""
        with patch('agent.slideshow.validation.run_command') as mock_run:
            mock_run.return_value = (True, "ffmpeg version", "")
            # Should not raise exception
            validate_ffmpeg_availability()
            assert mock_run.call_count == 2  # ffmpeg and ffprobe
    
    def test_validate_ffmpeg_availability_missing_ffmpeg(self):
        """Test FFmpeg availability validation when ffmpeg is missing."""
        with patch('agent.slideshow.validation.run_command') as mock_run:
            mock_run.side_effect = [(False, "", "not found"), (True, "", "")]
            with pytest.raises(FFmpegNotFoundError) as exc_info:
                validate_ffmpeg_availability()
            assert "ffmpeg" in str(exc_info.value)
    
    def test_validate_ffmpeg_availability_missing_ffprobe(self):
        """Test FFmpeg availability validation when ffprobe is missing."""
        with patch('agent.slideshow.validation.run_command') as mock_run:
            mock_run.side_effect = [(True, "", ""), (False, "", "not found")]
            with pytest.raises(FFmpegNotFoundError) as exc_info:
                validate_ffmpeg_availability()
            assert "ffprobe" in str(exc_info.value)
    
    def test_validate_file_exists_success(self):
        """Test file existence validation for existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test data")
            tmp_name = tmp.name
        try:
            # Should not raise exception
            validate_file_exists(tmp_name, "test file")
        finally:
            try:
                os.unlink(tmp_name)
            except (FileNotFoundError, PermissionError):
                pass
    
    def test_validate_file_exists_missing(self):
        """Test file existence validation for missing file."""
        with pytest.raises(MissingFileError) as exc_info:
            validate_file_exists("/nonexistent/file.jpg", "test file")
        assert "/nonexistent/file.jpg" in str(exc_info.value)
        assert "test file" in str(exc_info.value)
    
    def test_validate_timeline_segment_valid(self):
        """Test timeline segment validation for valid segment."""
        segment = {
            'cue_id': 'visual_01',
            'duration_frames': 90,
            'start_time': 0.0,
            'end_time': 3.0
        }
        issues = validate_timeline_segment(segment, 0)
        assert issues == []
    
    def test_validate_timeline_segment_missing_cue_id(self):
        """Test timeline segment validation for missing cue_id."""
        segment = {
            'duration_frames': 90,
            'start_time': 0.0,
            'end_time': 3.0
        }
        issues = validate_timeline_segment(segment, 0)
        assert len(issues) == 1
        assert "missing 'cue_id'" in issues[0]
    
    def test_validate_timeline_segment_invalid_duration(self):
        """Test timeline segment validation for invalid duration."""
        segment = {
            'cue_id': 'visual_01',
            'duration_frames': 0,  # Invalid: zero duration
            'start_time': 0.0,
            'end_time': 3.0
        }
        issues = validate_timeline_segment(segment, 0)
        assert len(issues) == 1
        assert "invalid duration_frames" in issues[0]
    
    def test_validate_timeline_segment_negative_duration(self):
        """Test timeline segment validation for negative duration."""
        segment = {
            'cue_id': 'visual_01',
            'duration_frames': -30,  # Invalid: negative duration
            'start_time': 0.0,
            'end_time': 3.0
        }
        issues = validate_timeline_segment(segment, 0)
        assert len(issues) == 1
        assert "invalid duration_frames" in issues[0]
    
    def test_validate_timeline_segments_success(self):
        """Test timeline segments validation for valid timeline."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test image data")
            tmp_name = tmp.name
        try:
            timeline = [
                {'cue_id': 'visual_01', 'duration_frames': 90},
                {'cue_id': 'visual_02', 'duration_frames': 120}
            ]
            all_image_paths = {
                'visual_01': tmp_name,
                'visual_02': tmp_name
            }
            # Should not raise exception
            validate_timeline_segments(timeline, all_image_paths)
        finally:
            try:
                os.unlink(tmp_name)
            except (FileNotFoundError, PermissionError):
                pass
    
    def test_validate_timeline_segments_missing_file(self):
        """Test timeline segments validation for missing file."""
        timeline = [
            {'cue_id': 'visual_01', 'duration_frames': 90}
        ]
        all_image_paths = {
            'visual_01': '/nonexistent/file.jpg'
        }
        with pytest.raises(TimelineValidationError) as exc_info:
            validate_timeline_segments(timeline, all_image_paths)
        assert "file not found" in str(exc_info.value)
    
    def test_validate_timeline_segments_missing_cue_id(self):
        """Test timeline segments validation for missing cue_id in paths."""
        timeline = [
            {'cue_id': 'visual_01', 'duration_frames': 90}
        ]
        all_image_paths = {}  # Empty - missing cue_id
        with pytest.raises(TimelineValidationError) as exc_info:
            validate_timeline_segments(timeline, all_image_paths)
        assert "not found in image paths" in str(exc_info.value)
    
    def test_validate_timeline_segments_empty_timeline(self):
        """Test timeline segments validation for empty timeline."""
        timeline = []
        all_image_paths = {}
        with pytest.raises(TimelineValidationError) as exc_info:
            validate_timeline_segments(timeline, all_image_paths)
        assert "Timeline is empty" in str(exc_info.value)
    
    def test_validate_audio_compatibility_success(self):
        """Test audio compatibility validation for matching duration."""
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp.write(b"fake audio data")
            tmp_name = tmp.name
        try:
            with patch('agent.slideshow.validation.get_audio_duration') as mock_duration:
                mock_duration.return_value = 10.0
                actual_duration = validate_audio_compatibility(tmp_name, 10.0)
                assert actual_duration == 10.0
        finally:
            try:
                os.unlink(tmp_name)
            except (FileNotFoundError, PermissionError):
                pass
    
    def test_validate_audio_compatibility_missing_file(self):
        """Test audio compatibility validation for missing audio file."""
        with pytest.raises(MissingFileError) as exc_info:
            validate_audio_compatibility('/nonexistent/audio.mp3', 10.0)
        assert "audio.mp3" in str(exc_info.value)
    
    def test_validate_audio_compatibility_duration_mismatch(self):
        """Test audio compatibility validation for duration mismatch."""
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp.write(b"fake audio data")
            tmp_name = tmp.name
        try:
            with patch('agent.slideshow.validation.get_audio_duration') as mock_duration:
                mock_duration.return_value = 8.0  # Different from expected 10.0
                with pytest.raises(AudioMismatchError) as exc_info:
                    validate_audio_compatibility(tmp_name, 10.0)
                assert "10.0" in str(exc_info.value)
                assert "8.0" in str(exc_info.value)
        finally:
            try:
                os.unlink(tmp_name)
            except (FileNotFoundError, PermissionError):
                pass
    
    def test_validate_audio_compatibility_zero_duration(self):
        """Test audio compatibility validation for zero duration audio."""
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp.write(b"fake audio data")
            tmp_name = tmp.name
        try:
            with patch('agent.slideshow.validation.get_audio_duration') as mock_duration:
                mock_duration.return_value = 0.0  # Zero duration
                with pytest.raises(AudioMismatchError) as exc_info:
                    validate_audio_compatibility(tmp_name, 10.0)
                assert "Could not determine audio duration" in str(exc_info.value)
        finally:
            try:
                os.unlink(tmp_name)
            except (FileNotFoundError, PermissionError):
                pass
    
    def test_validate_mixed_media_inputs_success(self):
        """Test mixed media inputs validation for valid inputs."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test data")
            tmp_name = tmp.name
        try:
            timeline = [
                {'image_source': 'visual_01', 'start_time': 0.0, 'end_time': 3.0},
                {'image_source': 'visual_02', 'start_time': 3.0, 'end_time': 6.0}
            ]
            all_paths = {
                'visual_01': tmp_name,
                'visual_02': tmp_name
            }
            # Should not raise exception
            validate_mixed_media_inputs(timeline, all_paths)
        finally:
            try:
                os.unlink(tmp_name)
            except (FileNotFoundError, PermissionError):
                pass
    
    def test_validate_mixed_media_inputs_missing_source(self):
        """Test mixed media inputs validation for missing image_source."""
        timeline = [
            {'start_time': 0.0, 'end_time': 3.0}  # Missing image_source
        ]
        all_paths = {}
        with pytest.raises(TimelineValidationError) as exc_info:
            validate_mixed_media_inputs(timeline, all_paths)
        assert "missing 'image_source'" in str(exc_info.value)
    
    def test_validate_mixed_media_inputs_invalid_duration(self):
        """Test mixed media inputs validation for invalid duration."""
        timeline = [
            {'image_source': 'visual_01', 'start_time': 3.0, 'end_time': 1.0}  # Invalid: end < start
        ]
        all_paths = {'visual_01': 'some/path.jpg'}
        with pytest.raises(TimelineValidationError) as exc_info:
            validate_mixed_media_inputs(timeline, all_paths)
        assert "invalid duration" in str(exc_info.value)
    
    def test_validate_mixed_media_inputs_zero_duration(self):
        """Test mixed media inputs validation for zero duration."""
        timeline = [
            {'image_source': 'visual_01', 'start_time': 3.0, 'end_time': 3.0}  # Zero duration
        ]
        all_paths = {'visual_01': 'some/path.jpg'}
        with pytest.raises(TimelineValidationError) as exc_info:
            validate_mixed_media_inputs(timeline, all_paths)
        assert "invalid duration" in str(exc_info.value)


class TestSlideshowInputValidation:
    """Test comprehensive slideshow input validation."""
    
    def test_validate_slideshow_inputs_success(self):
        """Test full slideshow input validation for valid inputs."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test data")
            tmp_name = tmp.name
        try:
            visual_analysis = {
                'segments': [
                    {'cue_id': 'visual_01'},
                    {'cue_id': 'visual_02'}
                ]
            }
            all_image_paths = {
                'visual_01': tmp_name,
                'visual_02': tmp_name
            }
            
            with patch('agent.slideshow.validation.run_command') as mock_run:
                mock_run.return_value = (True, "", "")
                with patch('agent.slideshow.validation.get_audio_duration') as mock_duration:
                    mock_duration.return_value = 10.0
                    with patch('agent.slideshow.create_smart_video._calculate_segment_durations') as mock_calc:
                        mock_calc.return_value = [
                            {'cue_id': 'visual_01', 'duration_frames': 150},
                            {'cue_id': 'visual_02', 'duration_frames': 150}
                        ]
                        
                        timeline, actual_duration = validate_slideshow_inputs(
                            visual_analysis, all_image_paths, tmp_name, 10.0
                        )
                        
                        assert len(timeline) == 2
                        assert actual_duration == 10.0
        finally:
            try:
                os.unlink(tmp_name)
            except (FileNotFoundError, PermissionError):
                pass
    
    def test_validate_slideshow_inputs_empty_analysis(self):
        """Test slideshow input validation for empty visual analysis."""
        with pytest.raises(TimelineValidationError) as exc_info:
            validate_slideshow_inputs({}, {}, '/fake/audio.mp3', 10.0)
        assert "Visual analysis is empty" in str(exc_info.value)
    
    def test_validate_slideshow_inputs_no_segments(self):
        """Test slideshow input validation for no segments."""
        visual_analysis = {'segments': []}
        with pytest.raises(TimelineValidationError) as exc_info:
            validate_slideshow_inputs(visual_analysis, {}, '/fake/audio.mp3', 10.0)
        assert "No segments found" in str(exc_info.value)
    
    def test_validate_slideshow_inputs_ffmpeg_missing(self):
        """Test slideshow input validation when FFmpeg is missing."""
        visual_analysis = {'segments': [{'cue_id': 'visual_01'}]}
        with patch('agent.slideshow.validation.run_command') as mock_run:
            mock_run.return_value = (False, "", "not found")
            with pytest.raises(FFmpegNotFoundError):
                validate_slideshow_inputs(visual_analysis, {}, '/fake/audio.mp3', 10.0)


class TestIntegrationWithSlideshowModule:
    """Integration tests with the main slideshow module."""
    
    def test_slideshow_run_with_validation_error(self):
        """Test slideshow run function handles validation errors gracefully."""
        from agent.slideshow.create_smart_video import run
        
        # Empty visual analysis should trigger validation error
        visual_analysis = {}
        all_image_paths = {}
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp.write(b"fake audio")
            tmp_name = tmp.name
        try:
            result = run(visual_analysis, all_image_paths, tmp_name, 10.0, '/fake/output.mp4')
            # Should return None on validation error
            assert result is None
        finally:
            try:
                os.unlink(tmp_name)
            except (FileNotFoundError, PermissionError):
                pass
    
    def test_slideshow_run_with_missing_file_error(self):
        """Test slideshow run function handles missing file errors."""
        from agent.slideshow.create_smart_video import run
        
        visual_analysis = {
            'segments': [{'cue_id': 'visual_01'}]
        }
        all_image_paths = {
            'visual_01': '/nonexistent/file.jpg'  # Missing file
        }
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp.write(b"fake audio")
            tmp_name = tmp.name
        try:
            result = run(visual_analysis, all_image_paths, tmp_name, 10.0, '/fake/output.mp4')
            # Should return None on missing file error
            assert result is None
        finally:
            try:
                os.unlink(tmp_name)
            except (FileNotFoundError, PermissionError):
                pass
    
    def test_slideshow_run_with_audio_mismatch_error(self):
        """Test slideshow run function handles audio mismatch errors."""
        from agent.slideshow.create_smart_video import run
        
        with tempfile.NamedTemporaryFile(delete=False) as img_tmp:
            img_tmp.write(b"fake image")
            img_tmp_name = img_tmp.name
        try:
            visual_analysis = {
                'segments': [{'cue_id': 'visual_01'}]
            }
            all_image_paths = {
                'visual_01': img_tmp_name
            }
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as audio_tmp:
                audio_tmp.write(b"fake audio")
                audio_tmp_name = audio_tmp.name
            try:
                with patch('agent.slideshow.validation.get_audio_duration') as mock_duration:
                    mock_duration.return_value = 5.0  # Different from expected 10.0
                    result = run(visual_analysis, all_image_paths, audio_tmp_name, 10.0, '/fake/output.mp4')
                    # Should return None on audio mismatch error
                    assert result is None
            finally:
                try:
                    os.unlink(audio_tmp_name)
                except (FileNotFoundError, PermissionError):
                    pass
        finally:
            try:
                os.unlink(img_tmp_name)
            except (FileNotFoundError, PermissionError):
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
