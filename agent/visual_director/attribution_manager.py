"""
Attribution Management System

This module handles license tracking, attribution requirements, and
generates appropriate attribution overlays and credit files.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from .asset_types import Asset


class AttributionManager:
    """
    Manages attribution requirements for visual assets used in videos.
    
    Tracks licenses, generates attribution text, creates credit files,
    and provides overlay specifications for required attributions.
    """
    
    # License types and their attribution requirements
    LICENSE_REQUIREMENTS = {
        'CC0': {
            'requires_attribution': False,
            'name': 'Creative Commons Zero',
            'description': 'No rights reserved'
        },
        'CC-BY': {
            'requires_attribution': True,
            'name': 'Creative Commons Attribution',
            'format': '{title} by {author} is licensed under CC BY {version}',
            'url_required': True
        },
        'CC-BY-SA': {
            'requires_attribution': True,
            'name': 'Creative Commons Attribution-ShareAlike',
            'format': '{title} by {author} is licensed under CC BY-SA {version}',
            'url_required': True,
            'share_alike': True
        },
        'CC-BY-NC': {
            'requires_attribution': True,
            'name': 'Creative Commons Attribution-NonCommercial',
            'format': '{title} by {author} is licensed under CC BY-NC {version}',
            'url_required': True,
            'commercial_use': False
        },
        'Pexels': {
            'requires_attribution': False,  # Optional but appreciated
            'name': 'Pexels License',
            'format': 'Photo by {author} on Pexels',
            'suggested': True
        },
        'Unsplash': {
            'requires_attribution': False,  # Optional but appreciated
            'name': 'Unsplash License',
            'format': 'Photo by {author} on Unsplash',
            'suggested': True
        },
        'Pixabay': {
            'requires_attribution': False,
            'name': 'Pixabay License',
            'format': 'Image by {author} from Pixabay',
            'suggested': True
        },
        'Tenor': {
            'requires_attribution': True,
            'name': 'Tenor',
            'format': 'GIF via Tenor',
            'minimal': True  # Just needs "via Tenor"
        },
        'editorial': {
            'requires_attribution': True,
            'name': 'Editorial Use',
            'format': '{source} - Editorial Use Only',
            'restrictions': ['editorial', 'non-commercial']
        }
    }
    
    def __init__(self, project_dir: str):
        """
        Initialize attribution manager.
        
        Args:
            project_dir: Directory to store attribution files
        """
        self.project_dir = project_dir
        self.attributions = []
        self.attribution_cache = {}
        
        # Create attribution directory
        self.attribution_dir = os.path.join(project_dir, 'attributions')
        os.makedirs(self.attribution_dir, exist_ok=True)
    
    def add_asset_attribution(self, asset: Asset, segment_info: Dict[str, Any]) -> None:
        """
        Add attribution for an asset used in the video.
        
        Args:
            asset: The asset requiring attribution
            segment_info: Information about where/how the asset is used
        """
        attribution_data = {
            'asset_id': asset.id,
            'source': asset.source,
            'license': asset.licence,
            'url': asset.url,
            'attribution_text': self.format_attribution(asset),
            'requires_display': self.requires_visual_attribution(asset),
            'segment': {
                'start_time': segment_info.get('start_time', 0),
                'end_time': segment_info.get('end_time', 0),
                'duration': segment_info.get('end_time', 0) - segment_info.get('start_time', 0)
            },
            'metadata': asset.metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        self.attributions.append(attribution_data)
        
        # Cache by asset ID for quick lookup
        self.attribution_cache[asset.id] = attribution_data
    
    def format_attribution(self, asset: Asset) -> str:
        """
        Format attribution text for an asset.
        
        Args:
            asset: The asset to create attribution for
            
        Returns:
            Formatted attribution string
        """
        license_info = self.LICENSE_REQUIREMENTS.get(asset.licence, {})
        
        if not license_info.get('requires_attribution') and not license_info.get('suggested'):
            return ""
        
        # Use custom attribution if provided
        if asset.attribution:
            return asset.attribution
        
        # Use license-specific format
        if 'format' in license_info:
            format_str = license_info['format']
            
            # Extract metadata
            metadata = asset.metadata or {}
            author = metadata.get('author', metadata.get('creator', 'Unknown'))
            title = metadata.get('title', 'Untitled')
            version = metadata.get('license_version', '4.0')
            source = asset.source
            
            # Format the attribution
            try:
                attribution = format_str.format(
                    title=title,
                    author=author,
                    version=version,
                    source=source
                )
                return attribution
            except KeyError:
                # Fallback if formatting fails
                return f"{title} - {asset.licence}"
        
        # Generic attribution
        return f"Asset from {asset.source} - {asset.licence}"
    
    def requires_visual_attribution(self, asset: Asset) -> bool:
        """
        Check if an asset requires visual attribution overlay.
        
        Args:
            asset: The asset to check
            
        Returns:
            True if visual attribution is required
        """
        license_info = self.LICENSE_REQUIREMENTS.get(asset.licence, {})
        
        # Always show attribution for these licenses
        if asset.licence in ['CC-BY', 'CC-BY-SA', 'CC-BY-NC', 'editorial']:
            return True
        
        # Show if specifically required
        if license_info.get('requires_attribution'):
            return True
        
        # Don't show for optional attributions by default
        return False
    
    def get_overlay_specifications(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get overlay specifications for displaying attribution.
        
        Args:
            asset_id: ID of the asset
            
        Returns:
            Overlay specification dict or None
        """
        attribution = self.attribution_cache.get(asset_id)
        if not attribution or not attribution['requires_display']:
            return None
        
        # Determine overlay style based on license
        license_type = attribution['license']
        
        if license_type == 'Tenor':
            # Minimal overlay for Tenor
            return {
                'text': attribution['attribution_text'],
                'position': 'bottom-right',
                'style': 'minimal',
                'font_size': 24,
                'opacity': 0.8,
                'padding': 10,
                'duration': 2.0,  # Show for 2 seconds
                'fade_in': 0.2,
                'fade_out': 0.3
            }
        elif license_type in ['CC-BY', 'CC-BY-SA', 'CC-BY-NC']:
            # Full attribution for CC licenses
            return {
                'text': attribution['attribution_text'],
                'position': 'bottom-left',
                'style': 'full',
                'font_size': 28,
                'opacity': 0.9,
                'padding': 15,
                'background': True,
                'background_opacity': 0.7,
                'duration': attribution['segment']['duration'],  # Show for entire segment
                'fade_in': 0.3,
                'fade_out': 0.3
            }
        else:
            # Default attribution style
            return {
                'text': attribution['attribution_text'],
                'position': 'bottom-center',
                'style': 'standard',
                'font_size': 26,
                'opacity': 0.85,
                'padding': 12,
                'duration': min(3.0, attribution['segment']['duration']),
                'fade_in': 0.2,
                'fade_out': 0.2
            }
    
    def generate_attribution_file(self, format: str = 'all') -> str:
        """
        Generate attribution file in various formats.
        
        Args:
            format: Output format ('txt', 'srt', 'json', 'html', 'all')
            
        Returns:
            Path to generated file(s)
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f'attributions_{timestamp}'
        
        files_created = []
        
        if format in ['txt', 'all']:
            txt_path = self._generate_txt_attribution(base_filename)
            files_created.append(txt_path)
        
        if format in ['srt', 'all']:
            srt_path = self._generate_srt_attribution(base_filename)
            files_created.append(srt_path)
        
        if format in ['json', 'all']:
            json_path = self._generate_json_attribution(base_filename)
            files_created.append(json_path)
        
        if format in ['html', 'all']:
            html_path = self._generate_html_attribution(base_filename)
            files_created.append(html_path)
        
        return files_created[0] if len(files_created) == 1 else self.attribution_dir
    
    def _generate_txt_attribution(self, base_filename: str) -> str:
        """Generate plain text attribution file."""
        output_path = os.path.join(self.attribution_dir, f'{base_filename}.txt')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== VIDEO ATTRIBUTIONS ===\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Group by license type
            by_license = {}
            for attr in self.attributions:
                license_type = attr['license']
                if license_type not in by_license:
                    by_license[license_type] = []
                by_license[license_type].append(attr)
            
            # Write attributions by license
            for license_type, attrs in by_license.items():
                license_info = self.LICENSE_REQUIREMENTS.get(license_type, {})
                f.write(f"\n{license_info.get('name', license_type)} Assets:\n")
                f.write("-" * 50 + "\n")
                
                for attr in attrs:
                    if attr['attribution_text']:
                        f.write(f"• {attr['attribution_text']}\n")
                        if attr['url']:
                            f.write(f"  Source: {attr['url']}\n")
                        f.write(f"  Used at: {attr['segment']['start_time']:.1f}s - {attr['segment']['end_time']:.1f}s\n")
                        f.write("\n")
        
        return output_path
    
    def _generate_srt_attribution(self, base_filename: str) -> str:
        """Generate SRT subtitle file with attributions."""
        output_path = os.path.join(self.attribution_dir, f'{base_filename}.srt')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            srt_index = 1
            
            for attr in self.attributions:
                if attr['requires_display'] and attr['attribution_text']:
                    # Format timestamps
                    start_time = self._seconds_to_srt_time(attr['segment']['start_time'])
                    end_time = self._seconds_to_srt_time(
                        attr['segment']['start_time'] + 
                        min(3.0, attr['segment']['duration'])  # Show for max 3 seconds
                    )
                    
                    f.write(f"{srt_index}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{attr['attribution_text']}\n\n")
                    
                    srt_index += 1
        
        return output_path
    
    def _generate_json_attribution(self, base_filename: str) -> str:
        """Generate JSON attribution file."""
        output_path = os.path.join(self.attribution_dir, f'{base_filename}.json')
        
        attribution_data = {
            'generated': datetime.now().isoformat(),
            'video_project': self.project_dir,
            'total_assets': len(self.attributions),
            'attributions': self.attributions,
            'license_summary': self._get_license_summary()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(attribution_data, f, indent=2)
        
        return output_path
    
    def _generate_html_attribution(self, base_filename: str) -> str:
        """Generate HTML attribution file."""
        output_path = os.path.join(self.attribution_dir, f'{base_filename}.html')
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Video Attributions</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        .attribution {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
        .time {{ color: #888; font-size: 0.9em; }}
        .license {{ font-weight: bold; color: #0066cc; }}
        a {{ color: #0066cc; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>Video Attributions</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Assets Used</h2>
"""
        
        # Group by license
        by_license = {}
        for attr in self.attributions:
            license_type = attr['license']
            if license_type not in by_license:
                by_license[license_type] = []
            by_license[license_type].append(attr)
        
        for license_type, attrs in by_license.items():
            license_info = self.LICENSE_REQUIREMENTS.get(license_type, {})
            html_content += f"<h3 class='license'>{license_info.get('name', license_type)}</h3>\n"
            
            for attr in attrs:
                if attr['attribution_text']:
                    html_content += "<div class='attribution'>\n"
                    html_content += f"<p>{attr['attribution_text']}</p>\n"
                    if attr['url']:
                        html_content += f"<p><a href='{attr['url']}' target='_blank'>View Source</a></p>\n"
                    html_content += f"<p class='time'>Used at: {attr['segment']['start_time']:.1f}s - {attr['segment']['end_time']:.1f}s</p>\n"
                    html_content += "</div>\n"
        
        html_content += """
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
    
    def _get_license_summary(self) -> Dict[str, Any]:
        """Get summary of licenses used."""
        summary = {}
        
        for attr in self.attributions:
            license_type = attr['license']
            if license_type not in summary:
                summary[license_type] = {
                    'count': 0,
                    'requires_attribution': False,
                    'info': self.LICENSE_REQUIREMENTS.get(license_type, {})
                }
            
            summary[license_type]['count'] += 1
            if attr['requires_display']:
                summary[license_type]['requires_attribution'] = True
        
        return summary
    
    def check_commercial_compatibility(self) -> Tuple[bool, List[str]]:
        """
        Check if all assets are compatible with commercial use.
        
        Returns:
            Tuple of (is_compatible, list_of_issues)
        """
        issues = []
        
        for attr in self.attributions:
            license_info = self.LICENSE_REQUIREMENTS.get(attr['license'], {})
            
            # Check for non-commercial licenses
            if license_info.get('commercial_use') is False:
                issues.append(
                    f"Asset {attr['asset_id']} has non-commercial license: {attr['license']}"
                )
            
            # Check for editorial restrictions
            if 'editorial' in license_info.get('restrictions', []):
                issues.append(
                    f"Asset {attr['asset_id']} is for editorial use only"
                )
        
        return len(issues) == 0, issues
    
    def generate_end_credits(self) -> List[str]:
        """
        Generate text for end credits.
        
        Returns:
            List of credit lines
        """
        credits = ["Visual Assets:"]
        
        # Group by source
        by_source = {}
        for attr in self.attributions:
            source = attr['source']
            if source not in by_source:
                by_source[source] = []
            if attr['attribution_text']:
                by_source[source].append(attr['attribution_text'])
        
        # Format credits by source
        for source, attributions in by_source.items():
            if attributions:
                credits.append("")
                credits.append(f"From {source}:")
                for attribution in set(attributions):  # Remove duplicates
                    credits.append(f"  {attribution}")
        
        return credits
