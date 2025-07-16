# Slideshow Codebase Audit - Hard-coded Dimensions & Orientation Strategy

## Executive Summary
The codebase currently has **inconsistent orientation handling** with hard-coded dimensions scattered throughout multiple functions. There is a clear conflict between landscape (1920×1080) and portrait (1080×1920) orientations that needs resolution.

## Hard-coded Dimensions Found

| Location | Line | Value | Usage Context | Issues |
|----------|------|-------|---------------|--------|
| `create_smart_video.py:47` | 47 | `width, height = 1920, 1080` | Main video output dimensions | **CONFLICT: Landscape** |
| `create_smart_video.py:109` | 109 | `width, height` parameters | Ken Burns effect scaling | Uses above landscape values |
| `create_smart_video.py:143` | 143 | `scale={width}:{height}` | FFmpeg scale filter | Uses landscape values |
| `create_smart_video.py:148` | 148 | `s=1080x1920` | Zoompan filter size | **CONFLICT: Portrait** |
| `create_smart_video.py:342` | 342 | `scale={width}:{height}` | Video segment scaling | Uses landscape values |
| `create_smart_video.py:353` | 353 | `scale=1080:1920` | Image segment scaling | **CONFLICT: Portrait** |
| `create_smart_video.py:354` | 354 | `s=1080x1920` | Zoompan size | **CONFLICT: Portrait** |
| `create_smart_video.py:467` | 467 | `scale={width}:{height}` | Webpage video scaling | Uses landscape values |
| `create_smart_video.py:489` | 489 | `scale=1080:1920` | Image scaling | **CONFLICT: Portrait** |
| `create_smart_video.py:492` | 492 | `s=1080x1920` | Zoompan size | **CONFLICT: Portrait** |

## Functions Touching Dimensions

| Function | Purpose | Dimension Dependencies | Global Variable Usage |
|----------|---------|------------------------|----------------------|
| `run()` | Main entry point | Sets `width, height = 1920, 1080` | **YES** - Creates globals |
| `_get_ken_burns_params_simple()` | Ken Burns effects | Takes `width, height` parameters | **YES** - Uses passed globals |
| `_create_with_video_segments()` | Mixed media slideshow | Hard-codes `1080:1920` | **NO** - Uses literals |
| `create_with_webpage_video()` | Webpage integration | Hard-codes `1080:1920` | **NO** - Uses literals |
| `_build_smart_filter()` | Filter chain builder | Passes `width, height` | **NO** - Parameter passing |
| `_assemble_ffmpeg_command()` | Command assembly | No direct dimension usage | **NO** |

## Critical Issues Identified

### 1. **Orientation Conflict**
- Main function sets **landscape** (1920×1080) 
- Multiple functions hard-code **portrait** (1080×1920)
- This creates inconsistent video output

### 2. **Inconsistent Global Usage**
- `run()` creates global variables but many functions ignore them
- Mixed parameter passing and hard-coded literals
- No centralized configuration

### 3. **Function Coupling**
- `_get_ken_burns_params_simple()` depends on global variables
- Other functions ignore globals and use literals
- Inconsistent architecture patterns

## Caller Module Analysis

### `agent/component_runner.py`
- Lines 257-307: Calls `create_smart_video()` 
- No dimension parameters passed
- Relies on slideshow module defaults

### `agent/__init__.py`
- Lines 22, 146: Imports but doesn't use slideshow directly
- References `smart_assembler` (missing module)

### `agent/component_registry.py`
- Lines 172-184: References slideshow component
- Points to wrong module path (`agent.video.smart_assembler`)

## Recommendation: Orientation Strategy

### **Decision: Portrait 1080×1920**

**Rationale:**
1. **Social Media Standard**: Portrait is dominant on TikTok, Instagram Reels, YouTube Shorts
2. **Current Codebase Majority**: Most functions already use 1080×1920
3. **Mobile-First**: Optimized for mobile viewing
4. **Engagement**: Portrait videos typically have higher engagement rates

### **Configuration Strategy: Non-Configurable**

**Rationale:**
1. **Simplicity**: Single orientation reduces complexity
2. **Consistency**: Ensures all outputs match platform requirements
3. **Maintenance**: Easier to maintain and debug
4. **Performance**: No conditional logic overhead

## Recommended Implementation

```python
# Constants at module level
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
ASPECT_RATIO = "9:16"  # Portrait
```

## Next Steps

1. **Standardize all hard-coded dimensions** to 1080×1920
2. **Create module-level constants** for consistent usage
3. **Refactor functions** to use centralized constants
4. **Update component registry** to point to correct module
5. **Test output consistency** across all video creation paths

---
**Final Decision: Portrait 1080×1920, Non-configurable**
