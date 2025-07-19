"""
Component Registry System for Pipeline Testing

Defines all components with their inputs, outputs, and dependencies
to enable isolated component testing and validation.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

@dataclass
class ComponentDependency:
    """Represents a single file dependency for a component."""
    file_path: str
    required: bool = True
    description: str = ""
    validator: Optional[callable] = None

@dataclass
class ComponentOutput:
    """Represents a file output from a component."""
    file_path: str
    description: str = ""
    validator: Optional[callable] = None

@dataclass
class ComponentSpec:
    """Complete specification for a pipeline component."""
    name: str
    description: str
    inputs: List[ComponentDependency]
    outputs: List[ComponentOutput]
    function_module: str
    function_name: str

class ComponentRegistry:
    """Registry of all pipeline components with dependency management."""
    
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.components = self._register_components()
    
    def _register_components(self) -> Dict[str, ComponentSpec]:
        """Register all pipeline components with their specifications."""
        
        def json_validator(file_path: str) -> bool:
            """Validate JSON file can be loaded."""
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)
                return True
            except:
                return False
        
        def text_validator(file_path: str) -> bool:
            """Validate text file has content."""
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                return len(content) > 0
            except:
                return False
        
        def image_validator(file_path: str) -> bool:
            """Validate image file exists and has reasonable size."""
            try:
                return os.path.exists(file_path) and os.path.getsize(file_path) > 1000
            except:
                return False
        
        components = {
            "discover": ComponentSpec(
                name="discover",
                description="Discover latest AI/automation headline from Hacker News",
                inputs=[],
                outputs=[
                    ComponentOutput("headline.json", "Headline data with title and URL", json_validator)
                ],
                function_module="agent.discover.discover_headline",
                function_name="run"
            ),
            
            "scrape": ComponentSpec(
                name="scrape", 
                description="Scrape article content and extract images",
                inputs=[
                    ComponentDependency("headline.json", True, "Headline with URL to scrape", json_validator)
                ],
                outputs=[
                    ComponentOutput("article.txt", "Scraped article text", text_validator),
                    ComponentOutput("image_urls.json", "List of image URLs found", json_validator),
                    ComponentOutput("article.html", "Raw HTML content", text_validator)
                ],
                function_module="agent.scrape.scrape_article",
                function_name="run"
            ),
            
            "screenshot": ComponentSpec(
                name="screenshot",
                description="Capture webpage screenshot and optional video",
                inputs=[
                    ComponentDependency("headline.json", True, "Headline with URL to capture", json_validator)
                ],
                outputs=[
                    ComponentOutput("url_screenshot.png", "Webpage screenshot", image_validator),
                    ComponentOutput("webpage_capture.mp4", "Webpage scroll video", os.path.exists)
                ],
                function_module="agent.screenshot.capture_url",
                function_name="run"  # Will need custom execution logic
            ),
            
            "script": ComponentSpec(
                name="script",
                description="Generate creative brief and video script",
                inputs=[
                    ComponentDependency("headline.json", True, "Article title", json_validator),
                    ComponentDependency("article.txt", True, "Article content", text_validator),
                ],
                outputs=[
                    ComponentOutput("creative_brief.json", "Creative brief with story angle and visual strategy", json_validator),
                    ComponentOutput("script.txt", "Video script with visual cues", text_validator),
                    ComponentOutput("script_clean.txt", "Clean script for TTS", text_validator)
                ],
                function_module="agent.script.write_script",
                function_name="run"
            ),
            
            "audio": ComponentSpec(
                name="audio",
                description="Generate audio narration from script",
                inputs=[
                    ComponentDependency("script_clean.txt", True, "Clean script text", text_validator)
                ],
                outputs=[
                    ComponentOutput("voice.mp3", "Generated audio narration", os.path.exists)
                ],
                function_module="agent.audio.generate_audio",
                function_name="run"
            ),
            
            "timing_extraction": ComponentSpec(
                name="timing_extraction",
                description="Extract word-level timestamps from audio using whisper",
                inputs=[
                    ComponentDependency("voice.mp3", True, "Generated audio narration", os.path.exists),
                    ComponentDependency("script_clean.txt", True, "Clean script text", text_validator),
                ],
                outputs=[
                    ComponentOutput("transcript_data.json", "Word-level timestamp data from whisper", json_validator),
                ],
                function_module="agent.audio.timestamp_extractor",
                function_name="extract_word_timestamps"
            ),
            
            "visual_director": ComponentSpec(
                name="visual_director",
                description="Directs visuals by classifying script segments and sourcing media.",
                inputs=[
                    ComponentDependency("creative_brief.json", True, "The overall creative direction", json_validator),
                    ComponentDependency("transcript_data.json", True, "Word-level timestamp data", json_validator),
                    ComponentDependency("visual_story_plan.json", True, "Visual story plan from script component", json_validator),
                    ComponentDependency("article.txt", False, "Article text for LLM tagging", text_validator),
                    ComponentDependency("headline.json", False, "Headline for context", json_validator)
                ],
                outputs=[
                    ComponentOutput("visual_map.json", "The final map of visual cues to media files", json_validator)
                ],
                function_module="agent.visual_director.enhanced_director",
                function_name="run"
            ),
            
            "slideshow": ComponentSpec(
                name="slideshow",
                description="Assemble all visual and audio assets into the final video.",
                inputs=[
                    ComponentDependency("visual_map.json", True, "The master visual plan", json_validator),
                    ComponentDependency("voice.mp3", True, "Audio narration", os.path.exists),
                    ComponentDependency("webpage_capture.mp4", True, "The opening video shot", os.path.exists)
                ],
                outputs=[
                    ComponentOutput("slideshow.mp4", "Video slideshow", os.path.exists)
                ],
                function_module="agent.video.smart_assembler",
                function_name="create_smart_video"
            ),
            
            "captions": ComponentSpec(
                name="captions",
                description="Add word-level captions to the final video",
                inputs=[
                    ComponentDependency("slideshow.mp4", True, "Video file without captions", os.path.exists),
                    ComponentDependency("transcript_data.json", True, "Word-level timestamp data", json_validator)
                ],
                outputs=[
                    ComponentOutput("final_video.mp4", "Final video with captions", os.path.exists)
                ],
                function_module="agent.video.word_captions",
                function_name="add_word_captions"
            ),
            
            "merge": ComponentSpec(
                name="merge",
                description="Merge audio with video slideshow",
                inputs=[
                    ComponentDependency("slideshow.mp4", True, "Video slideshow", os.path.exists),
                    ComponentDependency("voice.mp3", True, "Audio narration", os.path.exists)
                ],
                outputs=[
                    ComponentOutput("merged.mp4", "Audio+video merged", os.path.exists)
                ],
                function_module="agent.video.generate_video", 
                function_name="transcribe_and_caption"
            ),
        }
        
        return components
    
    def get_component(self, name: str) -> Optional[ComponentSpec]:
        """Get component specification by name."""
        return self.components.get(name)
    
    def list_components(self) -> List[str]:
        """Get list of all component names."""
        return list(self.components.keys())
    
    def validate_dependencies(self, component_name: str) -> Tuple[bool, List[str]]:
        """
        Validate that all dependencies for a component exist and are valid.
        
        Returns:
            Tuple of (all_valid, list_of_errors)
        """
        component = self.get_component(component_name)
        if not component:
            return False, [f"Unknown component: {component_name}"]
        
        errors = []
        
        for dep in component.inputs:
            file_path = os.path.join(self.run_dir, dep.file_path)
            
            # Check if file exists
            if not os.path.exists(file_path):
                if dep.required:
                    errors.append(f"Required dependency missing: {dep.file_path} - {dep.description}")
                else:
                    logging.info(f"Optional dependency missing: {dep.file_path}")
                continue
            
            # Run validator if provided
            if dep.validator and not dep.validator(file_path):
                errors.append(f"Invalid dependency: {dep.file_path} - Failed validation")
        
        return len(errors) == 0, errors
    
    def get_dependency_chain(self, component_name: str) -> List[str]:
        """Get ordered list of components that must run before this component."""
        if component_name not in self.components:
            return []
        
        # Simple dependency resolution based on file outputs
        chain = []
        component = self.components[component_name]
        
        for dep in component.inputs:
            if not dep.required:
                continue
                
            # Find which component produces this file
            for other_name, other_comp in self.components.items():
                if other_name == component_name:
                    continue
                    
                for output in other_comp.outputs:
                    if output.file_path == dep.file_path:
                        if other_name not in chain:
                            # Recursively get dependencies
                            sub_chain = self.get_dependency_chain(other_name)
                            for item in sub_chain:
                                if item not in chain:
                                    chain.append(item)
                            chain.append(other_name)
                        break
        
        return chain
    
    def validate_component_chain(self, component_name: str) -> Tuple[bool, List[str]]:
        """Validate entire dependency chain for a component."""
        chain = self.get_dependency_chain(component_name)
        chain.append(component_name)
        
        all_errors = []
        
        for comp_name in chain:
            valid, errors = self.validate_dependencies(comp_name)
            if not valid:
                all_errors.extend([f"[{comp_name}] {err}" for err in errors])
        
        return len(all_errors) == 0, all_errors 