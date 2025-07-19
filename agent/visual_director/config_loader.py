"""
Configuration Loader for Visual Director

This module handles loading and managing configuration from YAML files
with environment variable substitution and validation.
"""

import os
import re
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yaml


class ConfigLoader:
    """
    Handles configuration loading for the Visual Director system.
    
    Features:
    - YAML file parsing
    - Environment variable substitution
    - Default value handling
    - Configuration validation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or self._find_config_file()
        self.config = {}
        self.defaults = self._get_default_config()
    
    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in standard locations."""
        search_paths = [
            Path.cwd() / "config.yaml",
            Path.cwd() / "visual_director" / "config.yaml",
            Path(__file__).parent / "config.yaml",
            Path.home() / ".autosocialmedia" / "config.yaml",
        ]
        
        for path in search_paths:
            if path.exists():
                self.logger.info(f"Found config file at: {path}")
                return str(path)
        
        return None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            'visual_director': {
                'allow_paid': False,
                'max_results_per_source': 20,
                'portrait_width': 1080,
                'portrait_height': 1920,
                'min_score': 0.45,
                'reaction_min_conf': 0.6,
                'cache_dir': './cache',
                'project_dir': './projects',
                'enable_clip': True,
                'scoring': {
                    'relevance_weight': 0.4,
                    'quality_weight': 0.3,
                    'diversity_weight': 0.2,
                    'semantic_weight': 0.1
                },
                'providers': {
                    'searxng': {
                        'endpoint': 'http://localhost:8888/search',
                        'engines': ['wikimedia', 'duckduckgo_images']
                    },
                    'openverse': {
                        'api_key': None  # Public API
                    },
                    'wikimedia': {
                        'enabled': True
                    },
                    'nasa': {
                        'api_key': 'DEMO_KEY'
                    }
                }
            }
        }
    
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from file and environment.
        
        Returns:
            Loaded configuration dictionary
        """
        # Start with defaults
        config = self.defaults.copy()
        
        # Load from file if exists
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config = self._deep_merge(config, file_config)
                        self.logger.info(f"Loaded config from: {self.config_path}")
            except Exception as e:
                self.logger.error(f"Failed to load config file: {e}")
        
        # Substitute environment variables
        config = self._substitute_env_vars(config)
        
        # Validate configuration
        self._validate_config(config)
        
        self.config = config
        return config
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in config.
        
        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Pattern for ${VAR} or ${VAR:default}
            pattern = r'\${(\w+)(?::([^}]*))?}'
            
            def replacer(match):
                var_name = match.group(1)
                default_value = match.group(2)
                value = os.getenv(var_name)
                
                if value is None:
                    if default_value is not None:
                        return default_value
                    else:
                        self.logger.warning(f"Environment variable {var_name} not set")
                        return match.group(0)  # Return original
                
                return value
            
            return re.sub(pattern, replacer, config)
        else:
            return config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration values."""
        vd_config = config.get('visual_director', {})
        
        # Validate numeric ranges
        if not 0 <= vd_config.get('min_score', 0.45) <= 1:
            raise ValueError("min_score must be between 0 and 1")
        
        if not 0 <= vd_config.get('reaction_min_conf', 0.6) <= 1:
            raise ValueError("reaction_min_conf must be between 0 and 1")
        
        # Validate scoring weights sum to 1
        scoring = vd_config.get('scoring', {})
        weights = [
            scoring.get('relevance_weight', 0.4),
            scoring.get('quality_weight', 0.3),
            scoring.get('diversity_weight', 0.2),
            scoring.get('semantic_weight', 0.1)
        ]
        
        if abs(sum(weights) - 1.0) > 0.01:
            self.logger.warning(f"Scoring weights sum to {sum(weights)}, normalizing to 1.0")
            total = sum(weights)
            if total > 0:
                for key in ['relevance_weight', 'quality_weight', 'diversity_weight', 'semantic_weight']:
                    if key in scoring:
                        scoring[key] = scoring[key] / total
        
        # Validate dimensions
        if vd_config.get('portrait_width', 1080) <= 0:
            raise ValueError("portrait_width must be positive")
        
        if vd_config.get('portrait_height', 1920) <= 0:
            raise ValueError("portrait_height must be positive")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated path (e.g., 'visual_director.providers.pexels.api_key')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if not self.config:
            self.load()
        
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated path
            value: Value to set
        """
        if not self.config:
            self.load()
        
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set value
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            path: Path to save to (uses original path if not specified)
        """
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No path specified for saving configuration")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        self.logger.info(f"Saved configuration to: {save_path}")


# Global config instance
_config_loader = None


def get_config_loader(config_path: Optional[str] = None) -> ConfigLoader:
    """Get or create global configuration loader."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configuration dictionary
    """
    loader = get_config_loader(config_path)
    return loader.load()
