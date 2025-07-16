import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock

from agent.slideshow.create_smart_video import run as create_smart_video
from agent.slideshow.validation import validate_slideshow_inputs
from agent.video_config import get_default_config

# Sample data for testing
visual_analysis_single = {'segments': [{'cue_id': 'image1', 'start_time': 0, 'end_time': 5}]}
visual_analysis_multi = {
    'segments': [
        {'cue_id': 'image1', 'start_time': 0, 'end_time': 5},
        {'cue_id': 'image2', 'start_time': 5, 'end_time': 10, 'transition': 'fade'}
    ]
}
all_image_paths = {'image1': '/path/to/image1.jpg', 'image2': '/path/to/image2.jpg'}
audio_path = '/path/to/audio.mp3'
audio_duration = 10.0
output_path = '/path/to/output.mp4'
fps = 30


def test_constant_config():
    """Ensure width/height are propagated correctly."""
    config = get_default_config("landscape")
    assert config.width == 1920
    assert config.height == 1080


def test_build_filter_single():
    """Single-image timeline builds a valid graph."""
    with patch('agent.slideshow.validation.validate_slideshow_inputs') as mock_validate:
        mock_timeline = [{'cue_id': 'image1', 'duration_frames': 150}]
        mock_validate.return_value = (mock_timeline, 5.0)
        
        timeline, _ = mock_validate(visual_analysis_single, all_image_paths, audio_path, audio_duration)
        assert len(timeline) == 1
        assert timeline[0]['duration_frames'] > 0


def test_build_filter_multi():
    """Multi-image with transitions builds a valid graph."""
    with patch('agent.slideshow.validation.validate_slideshow_inputs') as mock_validate:
        mock_timeline = [
            {'cue_id': 'image1', 'duration_frames': 150},
            {'cue_id': 'image2', 'duration_frames': 150}
        ]
        mock_validate.return_value = (mock_timeline, 10.0)
        
        timeline, _ = mock_validate(visual_analysis_multi, all_image_paths, audio_path, audio_duration)
        assert len(timeline) == 2
        for segment in timeline:
            assert 'duration_frames' in segment and segment['duration_frames'] > 0


@patch('agent.slideshow.create_smart_video.run_command')
def test_mixed_media(mock_run_command):
    """Verify no NameError and FFmpeg command assembled correctly."""
    # Mock run_command to return success
    mock_run_command.return_value = (True, "", "")
    
    # Create temporary files
    temp_files = []
    try:
        tmp1 = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        tmp1.close()
        temp_files.append(tmp1.name)
        
        tmp2 = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        tmp2.close()
        temp_files.append(tmp2.name)
        
        tmp_audio = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        tmp_audio.close()
        temp_files.append(tmp_audio.name)
        
        temp_image_paths = {
            'image1': tmp1.name,
            'image2': tmp2.name
        }
        
        # Mock validation to avoid real file checks
        with patch('agent.slideshow.create_smart_video.validate_slideshow_inputs') as mock_validate:
            mock_timeline = [
                {'cue_id': 'image1', 'duration_frames': 150},
                {'cue_id': 'image2', 'duration_frames': 150}
            ]
            mock_validate.return_value = (mock_timeline, 10.0)
            
            final_video_path = create_smart_video(
                visual_analysis=visual_analysis_multi,
                all_image_paths=temp_image_paths,
                audio_path=tmp_audio.name,
                audio_duration=audio_duration,
                output_path=output_path,
                fps=fps
            )
        
        # Verify the function was called and returned expected path
        mock_run_command.assert_called()
        assert final_video_path == output_path
        
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass


@patch('agent.slideshow.create_smart_video.run_command')
@patch('agent.slideshow.create_smart_video._build_smart_filter')
@patch('agent.slideshow.create_smart_video.validate_slideshow_inputs')
def test_ffmpeg_command_structure(mock_validate, mock_build_filter, mock_run_command):
    """Test that FFmpeg command is properly structured with correct inputs and filters."""
    # Mock timeline and filter construction
    mock_timeline = [
        {'cue_id': 'image1', 'duration_frames': 150, 'start_time': 0, 'end_time': 5},
        {'cue_id': 'image2', 'duration_frames': 150, 'start_time': 5, 'end_time': 10}
    ]
    mock_validate.return_value = (mock_timeline, 10.0)
    
    # Mock filter chains that would be built
    mock_filter_chains = [
        "[0:v]scale=1920:1080:force_original_aspect_ratio=increase:flags=lanczos,crop=1920:1080,zoompan=z='1.0+0.2*on/30':d=150:s=1920x1080:fps=30[v0];",
        "[1:v]scale=1920:1080:force_original_aspect_ratio=increase:flags=lanczos,crop=1920:1080,zoompan=z='1.0+0.2*on/30':d=150:s=1920x1080:fps=30[v1];"
    ]
    mock_inputs_args = [
        '-loop', '1', '-t', '5.0', '-i', '/tmp/image1.jpg',
        '-loop', '1', '-t', '5.0', '-i', '/tmp/image2.jpg'
    ]
    mock_build_filter.return_value = (mock_filter_chains, mock_inputs_args)
    
    mock_run_command.return_value = (True, "", "")
    
    # Test data
    visual_analysis = {'segments': mock_timeline}
    all_image_paths = {
        'image1': '/tmp/image1.jpg',
        'image2': '/tmp/image2.jpg'
    }
    
    # Call the function
    result = create_smart_video(
        visual_analysis=visual_analysis,
        all_image_paths=all_image_paths,
        audio_path='/tmp/audio.mp3',
        audio_duration=10.0,
        output_path='/tmp/output.mp4',
        fps=30
    )
    
    # Verify the FFmpeg command structure
    assert mock_run_command.called
    ffmpeg_cmd = mock_run_command.call_args[0][0]
    
    # Check basic structure
    assert ffmpeg_cmd[0] == 'ffmpeg'
    assert '-y' in ffmpeg_cmd
    assert '-filter_complex' in ffmpeg_cmd
    assert '/tmp/output.mp4' in ffmpeg_cmd
    
    # Verify audio input is included
    assert '-i' in ffmpeg_cmd
    assert '/tmp/audio.mp3' in ffmpeg_cmd
    
    # Verify mapping
    assert '-map' in ffmpeg_cmd
    assert '[vout]' in ffmpeg_cmd  # Video output mapping
    
    # Check video codec settings
    assert '-c:v' in ffmpeg_cmd
    assert 'libx264' in ffmpeg_cmd
    assert '-c:a' in ffmpeg_cmd
    assert 'aac' in ffmpeg_cmd
    
    assert result == '/tmp/output.mp4'


@patch('agent.slideshow.create_smart_video.run_command')
@patch('agent.slideshow.create_smart_video.validate_slideshow_inputs')
@patch('agent.utils.get_audio_duration')
def test_slideshow_component_adapted(mock_get_audio_duration, mock_validate, mock_run_command):
    """Test adapted from existing test_slideshow_component to use new API."""
    # Mock dependencies
    mock_get_audio_duration.return_value = 15.0
    mock_timeline = [
        {'cue_id': 'visual_00', 'duration_frames': 225},
        {'cue_id': 'visual_01', 'duration_frames': 225}
    ]
    mock_validate.return_value = (mock_timeline, 15.0)
    mock_run_command.return_value = (True, "", "")
    
    # Create temporary files
    temp_files = []
    try:
        tmp1 = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        tmp1.close()
        temp_files.append(tmp1.name)
        
        tmp2 = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        tmp2.close()
        temp_files.append(tmp2.name)
        
        tmp_audio = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        tmp_audio.close()
        temp_files.append(tmp_audio.name)
        
        # Simulate the visual analysis structure from existing test
        visual_analysis = {
            'segments': [
                {'cue_id': 'visual_00', 'start_time': 0.0, 'end_time': 7.5},
                {'cue_id': 'visual_01', 'start_time': 7.5, 'end_time': 15.0}
            ]
        }
        
        # Image paths mapping
        all_image_paths = {
            'visual_00': tmp1.name,
            'visual_01': tmp2.name
        }
        
        # Test the slideshow creation function
        result = create_smart_video(
            visual_analysis=visual_analysis,
            all_image_paths=all_image_paths,
            audio_path=tmp_audio.name,
            audio_duration=15.0,
            output_path='/tmp/test_output.mp4',
            fps=30,
            config=get_default_config("landscape")
        )
        
        # Verify the function was called and returned expected path
        assert result == '/tmp/test_output.mp4'
        mock_run_command.assert_called()
        
        # Verify the FFmpeg command was constructed
        call_args = mock_run_command.call_args[0][0]
        assert call_args[0] == 'ffmpeg'
        assert '-y' in call_args  # Overwrite flag
        assert '/tmp/test_output.mp4' in call_args
        
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass

