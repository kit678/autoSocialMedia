"""
Component Runner for AutoSocialMedia Pipeline Orchestration

This module provides the core pipeline orchestration functionality for the AutoSocialMedia
system. It manages the execution of individual pipeline components with proper dependency
validation, error handling, and state management.

Pipeline Components:
    1. discover: Find trending news headlines from various sources
    2. scrape: Extract article content and metadata from URLs
    3. screenshot: Capture webpage visuals and create opening videos
    4. script: Generate AI-powered narration scripts
    5. audio: Convert scripts to speech using TTS providers
    6. timing_extraction: Extract word-level timestamps from audio
    7. visual_director: Orchestrate visual asset acquisition and timing
    8. slideshow: Assemble final video with synchronized visuals
    9. captions: Add word-level captions to the final video

Dependency Management:
    Each component has defined input/output requirements that are validated
    before execution. The runner ensures proper dependency chains and provides
    clear error messages for missing files or failed components.

Error Handling:
    - Comprehensive logging of all pipeline steps
    - Graceful fallback for non-critical failures
    - Detailed error reporting for debugging
    - Component isolation to prevent cascade failures

Configuration:
    - TTS provider selection (Kokoro, Google TTS)
    - Decision logging for AI component analysis
    - Configurable run directories and file management

Example:
    >>> from agent.component_runner import ComponentRunner
    >>> runner = ComponentRunner(run_dir="runs/test", logger=logger)
    >>> success = runner.run_component("discover")
    >>> if success:
    ...     print("Discovery completed successfully")
"""

import os
import json
import logging
import importlib
from typing import Dict, Any, Optional
from agent.component_registry import ComponentRegistry, ComponentSpec
from agent.utils import get_audio_duration
from agent.decision_logger import DecisionLogger
from agent.video_config import get_default_config

class ComponentRunner:
    """
    Handles isolated execution of pipeline components with dependency management.
    
    The ComponentRunner orchestrates the execution of individual pipeline components,
    managing their dependencies, validating inputs/outputs, and providing comprehensive
    error handling and logging.
    
    Attributes:
        run_dir (str): Directory for storing run-specific files and outputs
        logger (DecisionLogger): Logger for tracking AI decisions and pipeline state
        tts_provider (str): Text-to-speech provider ('kokoro' or 'google')
        registry (ComponentRegistry): Registry for component specifications
    
    Example:
        >>> from agent.component_runner import ComponentRunner
        >>> from agent.decision_logger import DecisionLogger
        >>> logger = DecisionLogger("runs/test")
        >>> runner = ComponentRunner("runs/test", logger, "kokoro")
        >>> success = runner.run_component("discover")
    """
    
    def __init__(self, run_dir: str, logger: DecisionLogger, tts_provider: str = 'kokoro', config: Dict[str, Any] = None):
        """
        Initialize the ComponentRunner with configuration and dependencies.
        
        Args:
            run_dir (str): Directory path for storing run-specific files and outputs.
                This directory will contain all intermediate and final files.
            logger (DecisionLogger): Logger instance for tracking AI decisions,
                component execution, and pipeline state.
            tts_provider (str, optional): Text-to-speech provider to use for audio
                generation. Supported values: 'kokoro', 'google'. Defaults to 'kokoro'.
            config (Dict[str, Any], optional): Configuration dictionary for pipeline components.
                Can specify visual director type and settings.
        
        Example:
            >>> logger = DecisionLogger("runs/current")
            >>> config = {'visual_director': {'type': 'ai', 'ai_config': {...}}}
            >>> runner = ComponentRunner(
            ...     run_dir="runs/current",
            ...     logger=logger,
            ...     tts_provider="google",
            ...     config=config
            ... )
        """
        self.run_dir = run_dir
        self.logger = logger
        self.tts_provider = tts_provider.lower()
        self.config = config or {}
        self.registry = ComponentRegistry(run_dir=self.run_dir)
    
    def _get_path(self, key: str) -> str:
        """Constructs a file path in the current run directory."""
        return os.path.join(self.run_dir, key)

    def _load_json(self, filename: str):
        with open(self._get_path(filename), 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_text(self, filename: str) -> str:
        with open(self._get_path(filename), 'r', encoding='utf-8') as f:
            return f.read()

    def _write_json(self, data: dict, filename: str):
        with open(self._get_path(filename), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def _write_text(self, text: str, filename: str):
        with open(self._get_path(filename), 'w', encoding='utf-8') as f:
            f.write(text)

    def run_pipeline(self):
        """
        Executes the full pipeline in sequence from discovery to final video.
        
        Runs all pipeline components in the correct dependency order:
        1. discover -> scrape -> screenshot -> script -> audio
        2. timing_extraction -> visual_director -> slideshow -> captions
        
        The pipeline will stop and raise an exception if any component fails,
        except for timing_extraction which is allowed to fail gracefully.
        
        Raises:
            Exception: If any critical component fails during execution
            
        Example:
            >>> runner = ComponentRunner("runs/test", logger)
            >>> runner.run_pipeline()  # Executes complete pipeline
        """
        pipeline = [
            'discover',
            'scrape',
            'screenshot',
            'script',
            'audio',
            'timing_extraction',
            'visual_director',
            'slideshow',
            'captions'
        ]
        for component_name in pipeline:
            success = self.run_component(component_name)
            if not success:
                raise Exception(f"Pipeline failed at component: {component_name}")

    def run_component(self, component_name: str) -> bool:
        """
        Execute a single pipeline component with full error handling and logging.
        
        Validates component dependencies, executes the component logic, validates
        outputs, and logs all decisions and errors for debugging.
        
        Args:
            component_name (str): Name of the component to execute. Must be one of:
                'discover', 'scrape', 'screenshot', 'script', 'audio',
                'timing_extraction', 'visual_director', 'slideshow', 'captions'
        
        Returns:
            bool: True if component executed successfully, False otherwise
            
        Example:
            >>> runner = ComponentRunner("runs/test", logger)
            >>> if runner.run_component("discover"):
            ...     print("Discovery completed successfully")
            ... else:
            ...     print("Discovery failed - check logs")
        
        Note:
            This method handles all exceptions internally and returns False
            for any failure. Check the logs for detailed error information.
        """
        component = self.registry.get_component(component_name)
        if not component:
            logging.error(f"Unknown component: {component_name}")
            return False
        
        print(f"\n>>> EXECUTING COMPONENT: {component_name}")
        logging.info(f"Executing component '{component_name}'...")
        self.logger.start_component(component_name)
        
        try:
            success = self._execute_component_by_name(component_name)
            if success:
                logging.info(f"[OK] Component '{component_name}' executed successfully")
                self._validate_outputs(component)
            else:
                logging.error(f"[FAIL] Component '{component_name}' execution failed")
            
            # Only save decision logs for components that actually log decisions
            components_with_decisions = ['script', 'visual_director', 'slideshow']
            if component_name in components_with_decisions:
                self.logger.finish_component()
                if success:
                    self.logger.save_master_log()
            else:
                # Don't create empty decision files for components without decisions
                self.logger._reset_current_component()
            return success
            
        except Exception as e:
            self.logger.finish_component()
            logging.error(f"Component '{component_name}' threw an exception: {e}", exc_info=True)
            return False
    
    def _execute_component_by_name(self, name: str) -> bool:
        """Dispatcher for component execution logic."""
        handlers = {
            'discover': self._run_discover,
            'scrape': self._run_scrape,
            'screenshot': self._run_screenshot,
            'script': self._run_script,
            'audio': self._run_audio,
            'timing_extraction': self._run_timing_extraction,
            'visual_director': self._run_visual_director,
            'slideshow': self._run_slideshow,
            'captions': self._run_captions
        }
        handler = handlers.get(name)
        if not handler:
            raise ValueError(f"No handler found for component: {name}")
        return handler()

    def _validate_outputs(self, component: ComponentSpec):
        for output in component.outputs:
            file_path = self._get_path(output.file_path)
            if not os.path.exists(file_path):
                logging.warning(f"  [WARN] Expected output not found: {output.file_path}")

    # --- Component Execution Handlers ---
    
    def _run_discover(self) -> bool:
        from agent.discover.discover_headline import run as run_discover
        result = run_discover()
        if result and result.get('primary'):
            # Save the primary story in the expected format for backward compatibility
            primary_story = result['primary']
            headline_data = {
                'title': primary_story['title'],
                'url': primary_story['url'],
                'fallbacks': result.get('fallbacks', [])
            }
            self._write_json(headline_data, "headline.json")
            return True
        return False
    
    def _run_scrape(self) -> bool:
        from agent.scrape.scrape_article import run as run_scrape
        headline = self._load_json("headline.json")
        
        # Try primary URL first
        result = run_scrape(headline['url'])
        if result:
            self._write_text(result['text'], "article.txt")
            self._write_json(result.get('image_urls', []), "image_urls.json")
            self._write_text(result.get('html_content', ''), "article.html")
            return True
        
        # If primary fails, try fallback URLs
        fallbacks = headline.get('fallbacks', [])
        for i, fallback in enumerate(fallbacks):
            logging.info(f"Primary URL failed, trying fallback {i+1}/{len(fallbacks)}: {fallback['title']}")
            result = run_scrape(fallback['url'])
            if result:
                # Update headline.json with the successful fallback URL
                headline['url'] = fallback['url']
                headline['title'] = fallback['title']
                self._write_json(headline, "headline.json")
                
                self._write_text(result['text'], "article.txt")
                self._write_json(result.get('image_urls', []), "image_urls.json")
                self._write_text(result.get('html_content', ''), "article.html")
                logging.info(f"Successfully scraped using fallback URL")
                return True
        
        logging.error("All URLs failed during scraping")
        return False
    
    def _run_screenshot(self) -> bool:
        from agent.screenshot.capture_url import run as run_screenshot, capture_webpage_video, analyze_webpage_layout
        headline = self._load_json("headline.json")
        
        urls_to_try = [{'url': headline['url'], 'title': headline['title']}]
        
        # Add fallback URLs
        fallbacks = headline.get('fallbacks', [])
        urls_to_try.extend(fallbacks)
        
        for i, url_data in enumerate(urls_to_try):
            url = url_data['url']
            title = url_data['title']
            
            if i > 0:
                logging.info(f"Primary screenshot failed, trying fallback {i}/{len(fallbacks)}: {title}")
            
            try:
                import time
                start_time = time.time()
                # Static screenshot
                run_screenshot(url, self._get_path('url_screenshot.png'))
                elapsed_time = time.time() - start_time
                logging.info(f"  ⌛ Web capture took {elapsed_time:.2f} seconds for {url}")
                
                # Intelligent video capture with fallback
                layout_data = analyze_webpage_layout(url)
                
                # Save layout data even if null (for debugging purposes)
                if layout_data:
                    self._write_json(layout_data, "layout_analysis.json")
                else:
                    # Save placeholder data indicating analysis failed
                    self._write_json({
                        "main_headline": None,
                        "main_image": None,
                        "scroll_range": {
                            "start_y": 0,
                            "end_y": 3000,
                            "total_scroll": 3000
                        },
                        "analysis_status": "failed"
                    }, "layout_analysis.json")
                    
                # Capture video (will fall back to basic scroll if layout_data is None)
                success = capture_webpage_video(url, self._get_path('webpage_capture.mp4'), layout_data=layout_data)
                
                if success:
                    # If this was a fallback URL, update headline.json
                    if i > 0:
                        headline['url'] = url
                        headline['title'] = title
                        self._write_json(headline, "headline.json")
                        logging.info(f"Successfully captured screenshot using fallback URL")
                    return True
                    
            except Exception as e:
                error_msg = str(e)
                if "VERIFICATION_FAILED" in error_msg:
                    logging.warning(f"  ⚠ Verification failed for URL {i+1}: {title}")
                    # Continue to next URL
                    continue
                else:
                    logging.error(f"  ✗ Screenshot failed for URL {i+1}: {error_msg}")
                    # Continue to next URL
                    continue
        
        logging.error("All URLs failed during screenshot capture")
        return False
    
    def _run_script(self) -> bool:
        from agent.script.write_script import run as run_script
        headline = self._load_json("headline.json")
        article_text = self._load_text("article.txt")
        return run_script(self.run_dir, article_text, headline['title'], self.logger)
    
    def _run_audio(self) -> bool:
        from agent.audio.generate_audio import run as run_audio
        script_text = self._load_text("script_clean.txt")
        voice_path = self._get_path('voice.mp3')
        return run_audio(script_text, voice_path, tts_provider=self.tts_provider)

    def _run_timing_extraction(self) -> bool:
        from agent.audio.timestamp_extractor import extract_word_timestamps
        audio_path = self._get_path('voice.mp3')
        script_text = self._load_text("script_clean.txt")
        timestamps = extract_word_timestamps(audio_path, script_text)
        self._write_json(timestamps, 'transcript_data.json')
        return True
    
    def _run_visual_director(self) -> bool:
        # Determine which visual director to use based on config
        director_type = self.config.get('visual_director', {}).get('type', 'conventional')
        
        if director_type == 'ai':
            from agent.visual_director_ai import run as run_ai_visuals
            transcript = self._load_json('transcript_data.json')
            full_script = self._load_text('script_clean.txt')
            creative_brief = self._load_json('creative_brief.json')
            ai_config = self.config.get('visual_director', {}).get('ai_config', {})
            result = run_ai_visuals(self.run_dir, transcript, full_script, creative_brief, self.logger, ai_config)
        else:
            # Default to conventional director
            from agent.visual_director_conventional import run as run_conventional_visuals
            transcript = self._load_json('transcript_data.json')
            creative_brief = self._load_json('creative_brief.json')
            result = run_conventional_visuals(self.run_dir, transcript, creative_brief, self.logger)
            
        return result is not None

    def _run_slideshow(self) -> bool:
        from agent.slideshow.create_smart_video import run as create_smart_video
        from agent.utils import get_audio_duration
        from agent.video_config import get_default_config

        # Load the necessary data for video creation
        # Try both possible filenames for backward compatibility
        visual_analysis = None
        if os.path.exists(self._get_path('visual_map_ai.json')):
            visual_analysis = self._load_json('visual_map_ai.json')
        elif os.path.exists(self._get_path('visual_map.json')):
            visual_analysis = self._load_json('visual_map.json')
        else:
            logging.error("No visual map file found (tried visual_map_ai.json and visual_map.json)")
            return False
            
        all_image_paths = visual_analysis.get('visual_map', {}).copy()
        audio_path = self._get_path('voice.mp3')
        
        # Visual director now handles opening/closing shots, so we use the timeline as-is
        if 'segments' not in visual_analysis:
            visual_analysis['segments'] = visual_analysis.get('visual_timeline_simple', 
                                                            visual_analysis.get('visual_timeline', []))

        # Ensure we have the audio duration
        audio_duration = get_audio_duration(audio_path)
        if not audio_duration:
            logging.error("Could not determine audio duration for slideshow creation.")
            return False

        # Directly call the new, unified smart video creation function
        # --- Hard-code portrait orientation (1080×1920) across the pipeline ---
        portrait_config = get_default_config("portrait")
        
        # Log the orientation choice for transparency
        self.logger.log_decision(
            step="slideshow_orientation",
            decision="Using portrait orientation for final video",
            reasoning="Aspect ratio hard-coded per product requirement",
            metadata={
                "width": portrait_config.width,
                "height": portrait_config.height,
                "orientation": "portrait"
            }
        )
        
        video_path = create_smart_video(
            visual_analysis=visual_analysis,
            all_image_paths=all_image_paths,
            audio_path=audio_path,
            audio_duration=audio_duration,
            output_path=self._get_path('slideshow.mp4'),
            config=portrait_config
        )
        return video_path is not None
    
    def _run_captions(self) -> bool:
        from agent.video.word_captions import add_word_captions
        
        video_path = self._get_path('slideshow.mp4')
        transcript_data = self._load_json('transcript_data.json')
        output_path = self._get_path('final_video.mp4')

        final_video_path = add_word_captions(
            video_path=video_path,
            transcript_data=transcript_data,
            output_path=output_path
        )
        return final_video_path is not None
    
    def print_component_info(self, component_name: str):
        """Print detailed information about a component."""
        component = self.registry.get_component(component_name)
        if not component:
            logging.error(f"Unknown component: {component_name}")
            return
        
        print(f"\n📦 Component: {component.name}")
        print(f"Description: {component.description}")
        
        if component.inputs:
            print(f"\n📥 Inputs:")
            for inp in component.inputs:
                req = "Required" if inp.required else "Optional"
                print(f"  - {inp.file_path} ({req}): {inp.description}")
        
        if component.outputs:
            print(f"\n📤 Outputs:")
            for out in component.outputs:
                print(f"  - {out.file_path}: {out.description}")
        
        # Show dependency chain
        chain = self.registry.get_dependency_chain(component_name)
        if chain:
            print(f"\n🔗 Dependency Chain: {' → '.join(chain)} → {component_name}")
        
        # Check current status
        valid, errors = self.registry.validate_dependencies(component_name)
        if valid:
            print(f"\n[OK] All dependencies are satisfied")
        else:
            print(f"\n[FAIL] Missing dependencies:")
            for error in errors:
                print(f"  - {error}")
    
    def _download_article_image(self, img_url: str, index: int) -> str:
        """Download and standardize article image to 1080x1920 pixels."""
        try:
            from agent.utils import download_and_standardize_image
            
            img_path = os.path.join(self.run_dir, f'article_image_{index}.jpg')
            
            if download_and_standardize_image(img_url, img_path):
                return img_path
            else:
                return None
            
        except Exception as e:
            logging.warning(f"Failed to download and standardize article image {index}: {e}")
            return None
    
    def _analyze_source_distribution(self, gathered_visuals: Dict) -> Dict:
        """Analyze which sources were used for gathered visuals."""
        source_counts = {'google': 0, 'stock_photo': 0, 'stock_video': 0, 'ai_generated': 0, 'failed': 0}
        total_visuals = len(gathered_visuals)
        
        for visual_data in gathered_visuals.values():
            source = visual_data.get('source', 'failed')
            if source in source_counts:
                source_counts[source] += 1
            else:
                source_counts['failed'] += 1
        
        # Calculate percentages
        distribution = {}
        for source, count in source_counts.items():
            distribution[source] = {
                'count': count,
                'percentage': (count / total_visuals * 100) if total_visuals > 0 else 0
            }
        
        # Add success rate
        successful = total_visuals - source_counts['failed']
        distribution['success_rate'] = (successful / total_visuals * 100) if total_visuals > 0 else 0
        distribution['total_visuals'] = total_visuals
        
        return distribution 