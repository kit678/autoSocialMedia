"""
Configuration Manager for Visual Director

This module handles configuration loading, API key management, and fallback
strategies for when certain adapters are unavailable.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import yaml
from pathlib import Path


class ConfigManager:
    """
    Manages configuration for the Visual Director system.
    
    Handles loading configs, managing API keys, and providing fallback
    strategies when certain adapters are unavailable.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config = self._load_config()
        
        # Track available adapters
        self.available_adapters = set()
        self.unavailable_adapters = set()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'visual_director': {
                'allow_paid': False,
                'max_results_per_source': 20,
                'portrait_width': 1080,
                'portrait_height': 1920,
                'min_score': 0.35,  # Lowered threshold
                'reaction_min_conf': 0.6,
                'fallback_enabled': True,
                'enable_clip': False,  # Disabled by default
                'cache_dir': './cache/visual_assets',
                'project_dir': './projects',
                'scoring': {
                    'relevance_weight': 0.5,
                    'quality_weight': 0.3,
                    'diversity_weight': 0.2,
                    'prefer_portrait': 0.1
                },
                'providers': {
                    'openverse': {
                        'enabled': True,
                        'priority': 1
                    },
                    'wikimedia': {
                        'enabled': True,
                        'priority': 2
                    },
                    'nasa': {
                        'enabled': True,
                        'priority': 3
                    },
                    'archive_tv': {
                        'enabled': True,
                        'priority': 4
                    },
                    'gdelt': {
                        'enabled': True,
                        'priority': 5
                    },
                    'searxng': {
                        'enabled': True,
                        'priority': 6,
                        'requires_api_key': False,
                        'config': {
                            'instance_url': 'https://searx.be'
                        }
                    },
                    'pexels': {
                        'enabled': True,
                        'priority': 7,
                        'requires_api_key': True,
                        'api_key_env': 'PEXELS_API_KEY',
                        'config': {
                            'api_key': '${PEXELS_API_KEY}'
                        }
                    },
                    'tenor': {
                        'enabled': True,
                        'priority': 8,
                        'requires_api_key': True,
                        'api_key_env': 'TENOR_API_KEY',
                        'config': {
                            'api_key': '${TENOR_API_KEY}'
                        }
                    },
                    'coverr': {
                        'enabled': True,
                        'priority': 9,
                        'requires_api_key': True,
                        'api_key_env': 'COVERR_API_KEY',
                        'config': {
                            'api_key': '${COVERR_API_KEY}'
                        }
                    }
                },
                'fallback_queries': [
                    'technology',
                    'science',
                    'innovation',
                    'research',
                    'future',
                    'digital',
                    'artificial intelligence',
                    'computer',
                    'data'
                ]
            }
        }
        
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    # Deep merge with defaults
                    config = self._deep_merge(default_config, loaded_config)
                    return config['visual_director']
            except Exception as e:
                self.logger.warning(f"Failed to load config from {self.config_path}: {e}")
        
        return default_config['visual_director']
    
    def _deep_merge(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = default.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider configuration with resolved environment variables
        """
        providers = self.config.get('providers', {})
        provider_config = providers.get(provider_name, {})
        
        # Resolve environment variables
        if 'config' in provider_config:
            resolved_config = {}
            for key, value in provider_config['config'].items():
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    resolved_value = os.getenv(env_var)
                    if resolved_value:
                        resolved_config[key] = resolved_value
                    else:
                        self.logger.warning(f"Environment variable {env_var} not set for {provider_name}")
                else:
                    resolved_config[key] = value
            
            provider_config['config'] = resolved_config
        
        return provider_config
    
    def check_provider_availability(self, provider_name: str) -> bool:
        """
        Check if a provider is available (API key present if required).
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            True if provider is available
        """
        provider_config = self.get_provider_config(provider_name)
        
        # Check if provider is enabled
        if not provider_config.get('enabled', True):
            return False
        
        # Check API key requirement
        if provider_config.get('requires_api_key', False):
            api_key_env = provider_config.get('api_key_env')
            if api_key_env and not os.getenv(api_key_env):
                return False
            
            # Also check resolved config
            config = provider_config.get('config', {})
            if 'api_key' in config and not config['api_key']:
                return False
        
        return True
    
    def get_available_providers(self) -> List[str]:
        """
        Get list of available providers sorted by priority.
        
        Returns:
            List of provider names that are available
        """
        providers = self.config.get('providers', {})
        available = []
        
        for provider_name, provider_config in providers.items():
            if self.check_provider_availability(provider_name):
                available.append((provider_name, provider_config.get('priority', 999)))
                self.available_adapters.add(provider_name)
            else:
                self.unavailable_adapters.add(provider_name)
        
        # Sort by priority (lower number = higher priority)
        available.sort(key=lambda x: x[1])
        
        return [name for name, _ in available]
    
    def get_fallback_queries(self) -> List[str]:
        """
        Get fallback queries for when specific search terms don't work.
        
        Returns:
            List of fallback query terms
        """
        return self.config.get('fallback_queries', [])
    
    def is_fallback_enabled(self) -> bool:
        """Check if fallback mode is enabled."""
        return self.config.get('fallback_enabled', True)
    
    def get_min_score_threshold(self) -> float:
        """Get minimum score threshold for asset selection."""
        return self.config.get('min_score', 0.35)
    
    def should_use_fallback_threshold(self) -> bool:
        """Check if we should use a lower threshold when few providers are available."""
        return len(self.available_adapters) < 3
    
    def get_fallback_threshold(self) -> float:
        """Get lowered threshold for when few providers are available."""
        return max(0.2, self.get_min_score_threshold() - 0.15)
    
    def log_provider_status(self):
        """Log the status of all providers."""
        self.logger.info(f"Available providers: {', '.join(self.available_adapters)}")
        if self.unavailable_adapters:
            self.logger.warning(f"Unavailable providers: {', '.join(self.unavailable_adapters)}")
            self.logger.info("To enable more providers, set the following environment variables:")
            
            for provider in self.unavailable_adapters:
                provider_config = self.get_provider_config(provider)
                api_key_env = provider_config.get('api_key_env')
                if api_key_env:
                    self.logger.info(f"  - {api_key_env} (for {provider})")
    
    def create_sample_config(self, output_path: str):
        """
        Create a sample configuration file.
        
        Args:
            output_path: Path to save the sample config
        """
        sample_config = {
            'visual_director': {
                'allow_paid': False,
                'max_results_per_source': 20,
                'min_score': 0.35,
                'fallback_enabled': True,
                'enable_clip': False,
                'providers': {
                    'pexels': {
                        'enabled': True,
                        'priority': 1,
                        'config': {
                            'api_key': '${PEXELS_API_KEY}'
                        }
                    },
                    'tenor': {
                        'enabled': True,
                        'priority': 2,
                        'config': {
                            'api_key': '${TENOR_API_KEY}'
                        }
                    },
                    'searxng': {
                        'enabled': True,
                        'priority': 3,
                        'config': {
                            'instance_url': 'https://searx.be'
                        }
                    }
                }
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Sample configuration saved to {output_path}")


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get a singleton configuration manager.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigManager instance
    """
    if not hasattr(get_config_manager, '_instance'):
        get_config_manager._instance = ConfigManager(config_path)
    
    return get_config_manager._instance
