# AutoSocialMedia Pipeline - Detailed End-to-End Analysis

## Executive Summary

The AutoSocialMedia pipeline is an automated system that transforms trending news articles into professional social media videos. It consists of 9 sequential components that work together to discover content, generate scripts, create visuals, and assemble final videos with captions. The pipeline is designed for portrait orientation (1080x1920) optimized for platforms like TikTok, Instagram Reels, and YouTube Shorts.

## Pipeline Architecture Overview

```
main.py → ComponentRunner → [9 Components] → final_video.mp4
```

### Component Execution Order:
1. **discover** → Find trending headlines
2. **scrape** → Extract article content
3. **screenshot** → Capture webpage visuals
4. **script** → Generate AI scripts
5. **audio** → Text-to-speech conversion
6. **timing_extraction** → Word-level timestamps
7. **visual_director** → Source and validate visuals
8. **slideshow** → Assemble video with effects
9. **captions** → Add synchronized captions

## Detailed Component Analysis

### 1. Entry Point: main.py

**Purpose**: Initialize the pipeline and set up the execution environment.

**Step-by-Step Process**:
1. **UTF-8 Configuration** (Lines 13-24)
   - Detects Windows platform
   - Reconfigures stdout/stderr to UTF-8 encoding
   - Prevents UnicodeEncodeError for special characters

2. **Run Directory Setup** (Lines 79-88)
   - Creates `runs/current` directory structure
   - Clears previous run data (Lines 30-60)
   - Creates decisions subdirectory for AI logging

3. **Logging Configuration** (Lines 61-70)
   - Sets up dual logging (file + console)
   - UTF-8 encoding for log files
   - Suppresses verbose library warnings

4. **Pipeline Initialization** (Lines 93-95)
   - Creates DecisionLogger for AI decision tracking
   - Initializes ComponentRunner with TTS provider
   - Default TTS: Kokoro (configurable via config.ini)

5. **Pipeline Execution** (Lines 98-102)
   - Calls `runner.run_pipeline()`
   - Handles exceptions with detailed logging

**Key Files Created**:
- `runs/current/run.log` - Main execution log
- `runs/current/decisions/` - AI decision tracking directory

**Configuration Used**:
- `config.ini` - TTS provider selection (kokoro/google)

---

### 2. Component: discover

**Purpose**: Find trending AI/tech headlines from Hacker News.

**Module**: `agent/discover/discover_headline.py`

**Step-by-Step Process**:

1. **API Query** (Lines 189-194)
   - Queries Hacker News Algolia API
   - Search term: "artificial intelligence"
   - Fetches 30 most recent stories

2. **Paywall Detection** (Lines 5-112)
   - Checks against known paywall domains (NYTimes, WSJ, etc.)
   - Tests for Cloudflare challenges
   - Makes HEAD/GET requests to verify accessibility
   - Detects paywall keywords in content

3. **Story Validation** (Lines 202-222)
   - Iterates through fetched stories
   - Validates each URL for paywall/bot detection
   - Collects up to 3 valid stories (primary + 2 fallbacks)

4. **Fallback Strategy** (Lines 114-178)
   - If AI search fails, tries broader tech searches
   - Search terms: "technology", "programming", "software"
   - Last resort: hardcoded GitHub trending page

**Output Files**:
- `headline.json`:
  ```json
  {
    "title": "Article headline",
    "url": "https://example.com/article",
    "fallbacks": [
      {"title": "Fallback 1", "url": "..."},
      {"title": "Fallback 2", "url": "..."}
    ]
  }
  ```

**Error Handling**:
- Network failures → broader search fallback
- All searches fail → hardcoded tech story
- Rate limiting → exponential backoff

---

### 3. Component: scrape

**Purpose**: Extract article content and images from the discovered URL.

**Module**: `agent/scrape/scrape_article.py`

**Step-by-Step Process**:

1. **HTTP Request** (Lines 31-36)
   - Uses browser-like headers to avoid bot detection
   - 20-second timeout with retry logic
   - Handles compressed responses (gzip, deflate, br)

2. **Content Extraction** (Lines 35-39)
   - Uses Readability library for article extraction
   - Extracts main content, removes ads/navigation
   - Converts to clean plain text

3. **Image Discovery** (Lines 42-74)
   - Finds all `<img>` tags in article content
   - Handles relative URLs → absolute conversion
   - Filters out tracking pixels and data URLs
   - Extracts OpenGraph (og:image) metadata
   - Removes duplicates while preserving order

4. **Fallback URL Handling** (Lines 259-277)
   - If primary URL fails, tries fallback URLs
   - Updates headline.json with successful URL
   - Logs fallback usage for transparency

**Output Files**:
- `article.txt` - Clean article text
- `image_urls.json` - List of discovered image URLs
- `article.html` - Raw HTML content for analysis

**Error Handling**:
- Request failures → try fallback URLs
- No valid content → component fails
- Encoding issues → handled by Readability

---

### 4. Component: screenshot

**Purpose**: Capture webpage screenshot and create opening video sequence.

**Module**: `agent/screenshot/capture_url.py`

**Step-by-Step Process**:

1. **Browser Setup** (Lines 1187-1292)
   - Launches Chromium with anti-detection settings
   - Mobile viewport: 375x812 (iPhone standard)
   - Overrides webdriver detection scripts
   - Custom user agent for iOS Safari

2. **Static Screenshot** (Lines 24-121)
   - Navigates to URL with enhanced headers
   - Handles verification/CAPTCHA pages
   - Dismisses cookie banners automatically
   - Takes full-page PNG screenshot
   - Standardizes to 1080x1920 portrait

3. **Layout Analysis** (Lines 164-219, 1049-1159)
   - Uses Gemini Vision API for intelligent analysis
   - Identifies main headline coordinates
   - Locates primary article image
   - Calculates optimal scroll range
   - Falls back gracefully if analysis fails

4. **Video Capture** (Lines 221-372, 704-744)
   - Creates two-phase scroll sequence:
     - Phase 1 (2s): Original webpage view
     - Phase 2 (8s): Smooth scroll to headline
   - Captures at 15 FPS for efficiency
   - Applies easing curves for natural motion
   - Handles popup dismissal during recording

5. **Video Processing** (Lines 675-702)
   - Converts frames to MP4 using FFmpeg
   - Scales to 1080x1920 portrait
   - H.264 encoding with CRF 23
   - Optimized for mobile viewing

**Output Files**:
- `url_screenshot.png` - Static webpage capture
- `webpage_capture.mp4` - 10-second scroll video
- `layout_analysis.json` - Gemini Vision analysis data

**Verification Handling**:
- Cloudflare challenges → Buster extension simulation
- CAPTCHA detection → AntiCaptcha API fallback
- Cookie banners → Automated dismissal
- Membership popups → Skip/close automation

**Error Handling**:
- Verification fails → try fallback URLs
- Gemini fails → basic scroll without intelligence
- Frame capture fails → retry with state reset
- All URLs fail → component fails

---

### 5. Component: script

**Purpose**: Generate AI-powered narration script with visual planning.

**Module**: `agent/script/write_script.py`

**Step-by-Step Process**:

1. **Visual Story Planning** (Lines 31-33)
   - Calls `visual_story_planner.run()`
   - Analyzes article for visual opportunities
   - Creates structured visual narrative plan

2. **DeepSeek API Analysis** (`visual_story_planner.py` Lines 121-194)
   - Sends article to DeepSeek LLM
   - Identifies 8-12 visual segments
   - Maps abstract concepts to searchable visuals
   - Plans story pacing and emotional tone
   - Creates visual metaphors for complex ideas

3. **Script Engineering** (Lines 196-237)
   - Reverse-engineers script from visual plan
   - Uses trigger words for planned visuals
   - Maintains 60-80 word target (30-35 seconds)
   - Incorporates entity names naturally
   - Applies specified narrative style

4. **Script Cleaning** (Lines 97-130)
   - Removes formatting markers (*bold*, [stage directions])
   - Strips metadata (word count, duration notes)
   - Normalizes whitespace
   - Removes quotation marks if present

5. **Creative Brief Generation** (Lines 150-229)
   - Converts visual plan to legacy format
   - Maintains backward compatibility
   - Includes story angle and emotions
   - Maps visual cues to segments

**Output Files**:
- `script.txt` - Full script with visual cues
- `script_clean.txt` - TTS-ready clean script
- `creative_brief.json` - Visual strategy metadata
- `visual_story_plan.json` - Detailed visual planning

**AI Decision Points**:
- Story hook selection
- Visual segment identification  
- Narrative style choice
- Pacing strategy
- Emotional targeting

**Error Handling**:
- API failure → basic script fallback
- JSON parsing errors → minimal brief
- Empty response → retry with timeout

---

### 6. Component: audio

**Purpose**: Convert script to speech using TTS providers.

**Module**: `agent/audio/generate_audio.py`

**Step-by-Step Process**:

1. **Provider Selection** (Lines 5-25)
   - Reads TTS provider from config
   - Routes to Kokoro or Google TTS
   - Validates provider availability

2. **Kokoro TTS** (`kokoro_tts.py`):
   - **Content Analysis** (Lines 9-68)
     - Detects sentiment (urgent/negative/positive/neutral)
     - Identifies topic category (tech/general)
     - Counts emotional indicators
   
   - **Voice Selection** (Lines 70-101)
     - Maps sentiment to voice personas
     - Adjusts speech rate (0.95-1.2x)
     - Tech content → Sarah voice (clear, professional)
     - Negative → Adam voice (serious, deep)
     - Positive → Nicole voice (upbeat, friendly)
   
   - **Audio Generation** (Lines 103-147)
     - Generates 24kHz WAV audio
     - Concatenates audio chunks
     - Converts to MP3 (128k bitrate)
     - Cleans up temporary files

3. **Google TTS** (`google_tts.py`):
   - **Gemini 2.5 Flash API** (Lines 15-61)
     - Uses latest TTS model
     - Configurable temperature (0.6-0.9)
     - Voice selection based on sentiment
     - Streaming audio generation
   
   - **Rate Limit Handling** (Lines 120-136)
     - 3 RPM / 15 RPD limits
     - Exponential backoff retry
     - Warning messages for limits

**Output Files**:
- `voice.mp3` - Generated narration audio

**Voice Configuration**:
- Kokoro: 7 voice options with speed control
- Google: 4 voice presets with style prompts
- Dynamic selection based on content

**Error Handling**:
- TTS failure → retry with backoff
- Invalid provider → error message
- File write failure → cleanup attempt

---

### 7. Component: timing_extraction

**Purpose**: Extract word-level timestamps using Whisper.

**Module**: `agent/audio/timestamp_extractor.py`

**Step-by-Step Process**:

1. **Whisper Model Loading**
   - Loads OpenAI Whisper model
   - Configures for timestamp extraction
   - Sets language to English

2. **Audio Processing**
   - Loads voice.mp3 audio file
   - Extracts audio waveform
   - Normalizes audio levels

3. **Transcription with Timestamps**
   - Processes audio through Whisper
   - Generates word-level timing data
   - Includes confidence scores
   - Handles punctuation attachment

4. **Data Structure Creation**
   - Creates word_timings dictionary
   - Builds segments array
   - Includes all_words list
   - Preserves original formatting

**Output Files**:
- `transcript_data.json`:
  ```json
  {
    "word_timings": {
      "word": {
        "start_time": 0.52,
        "end_time": 0.86,
        "confidence": 0.995,
        "original_text": "word"
      }
    },
    "segments": [...],
    "all_words": [...]
  }
  ```

**Key Features**:
- Sub-second precision
- Confidence scoring
- Punctuation handling
- Multi-segment support

**Error Handling**:
- Non-critical component (can fail gracefully)
- Falls back to estimated timing
- Logs warnings but continues

---

### 8. Component: visual_director

**Purpose**: Source and validate visual assets for each script segment.

**Module**: `agent/visual_director/direct_visuals.py`

**Step-by-Step Process**:

1. **Visual Plan Loading** (Lines 30-39)
   - Loads visual_story_plan.json
   - Validates plan structure
   - Extracts visual segments

2. **Timeline Enhancement** (Lines 114-226)
   - **Transcript Synchronization** (Lines 140-159)
     - Matches keywords to transcript timing
     - Calculates segment durations
     - Ensures minimum 2-second segments
   
   - **Visual Search Execution** (Lines 173-180)
     - Priority: SearXNG → Pexels → AI Generation
     - Uses narrative context for validation
     - Executes planned searches

3. **Visual Acquisition** (`_execute_planned_search`):
   - **Search Strategy**:
     - Primary search term + secondary keywords
     - Multiple source fallbacks
     - Gemini validation for relevance
   
   - **Source Priority**:
     1. SearXNG (Google Images)
     2. Pexels Photos (Stock)
     3. AI Generation (Fallback)
     4. Pexels Videos (Motion)

4. **Media Standardization** (Lines 182-194)
   - Images → 1080x1920 portrait
   - Videos → 1080x1920 portrait
   - Maintains aspect ratios
   - High-quality scaling

5. **Validation & Quality Control** (Lines 196-201)
   - Verifies file existence
   - Checks file sizes
   - **Critical**: Crashes if visual missing
   - Logs source attribution

**Output Files**:
- `visual_map.json`:
  ```json
  {
    "visual_timeline": [...],
    "visual_map": {
      "visual_00": "/path/to/image1.jpg",
      "visual_01": "/path/to/image2.jpg"
    },
    "segments": [...],
    "visual_strategy": {...}
  }
  ```

**Visual Files**:
- `visual_XX.jpg/mp4` - Downloaded/generated visuals
- All standardized to 1080x1920

**AI Decision Points**:
- Visual relevance validation
- Source selection priority
- Fallback trigger conditions
- Quality assessment

**Error Handling**:
- Missing visual → **FATAL ERROR** (by design)
- API failures → source fallbacks
- Invalid dimensions → standardization
- Timeout → skip to next source

---

### 9. Component: slideshow

**Purpose**: Assemble final video with synchronized visuals.

**Module**: `agent/slideshow/create_smart_video.py`

**Step-by-Step Process**:

1. **Opening Segment Addition** (ComponentRunner Lines 385-405)
   - Inserts webpage_capture.mp4 as first visual
   - 3-second duration
   - Shifts all other segments by 3 seconds
   - Creates opening_video entry

2. **Input Validation** (Lines 63-77)
   - Validates visual timeline structure
   - Checks audio duration
   - Caps timeline to match audio
   - Loads portrait configuration (1080x1920)

3. **Filter Chain Building** (Lines 119-185)
   - **For Each Segment**:
     - Videos: Trim, deinterlace, scale to portrait
     - Images: Ken Burns effect with zoom/pan
     - Random motion parameters
     - Consistent FPS normalization

4. **Ken Burns Effects** (Lines 187-221)
   - Start zoom: 1.0-1.1x
   - End zoom: 1.2-1.4x
   - Random pan directions
   - Smooth easing curves
   - 30 FPS frame generation

5. **Transition Assembly** (Lines 25-46)
   - Random transition selection per segment
   - 1-second transition duration
   - Available transitions:
     - fade, dissolve, distance
     - wipes (left/right/up/down)
     - slides, circles, radial
   - Seamless blending

6. **FFmpeg Command** (Lines 223-260)
   - Inputs: All visuals + audio
   - Complex filter graph
   - H.264 encoding
   - AAC audio
   - 30 FPS output
   - YUV420p pixel format

**Output Files**:
- `slideshow.mp4` - Video without captions

**Configuration**:
- **Hard-coded**: Portrait 1080x1920
- Transition duration: 1.0s
- Output FPS: 30
- Video codec: libx264
- Audio codec: AAC

**Error Handling**:
- Missing files → skip segment
- FFmpeg failure → return None
- Invalid duration → adjust timeline
- Zero frames → error message

---

### 10. Component: captions

**Purpose**: Add word-level synchronized captions.

**Module**: `agent/video/word_captions.py`

**Step-by-Step Process**:

1. **FFmpeg Setup** (Lines 73-89)
   - Locates full-featured FFmpeg build
   - Overrides system FFmpeg
   - Ensures drawtext filter availability

2. **ASS Subtitle Generation** (Lines 211-262)
   - Creates karaoke-style subtitles
   - Groups words (max 5 per line)
   - Calculates display timing
   - Two styles: Default + Highlight

3. **Caption Styling**:
   - Font: Arial, 38pt
   - Colors: White text, black outline
   - Position: Bottom center (margin 50)
   - Bold weight for visibility
   - 2.5pt outline thickness

4. **Burn-in Methods** (Lines 164-189)
   - Primary: drawtext filter (Windows)
   - Fallback: subtitles filter
   - Automatic fallback on failure
   - Validates filter availability

5. **Word Grouping Logic**:
   - Maximum 5 words per line
   - Natural phrase boundaries
   - Synchronized with speech timing
   - Smooth transitions

**Output Files**:
- `final_video.mp4` - Complete video with captions
- `*_words.ass` - Temporary subtitle file (deleted)

**Technical Details**:
- ASS format version 4.00+
- UTF-8 encoding throughout
- PlayResX: 1080, PlayResY: 1920
- Alignment: Bottom center (2)

**Error Handling**:
- Missing drawtext → subtitles fallback
- Both fail → exception raised
- Temp file cleanup on success/failure
- Path sanitization for Windows

---

## Data Flow Summary

### File Dependencies

```
headline.json → scrape → article.txt
                      ↓
                    script → script_clean.txt → audio → voice.mp3
                                            ↓
                                    timing_extraction → transcript_data.json
                                            ↓
creative_brief.json ←─────────────── visual_director → visual_map.json
        ↓                                   ↓
webpage_capture.mp4 → slideshow ←───────────┘
                          ↓
                    slideshow.mp4 → captions → final_video.mp4
```

### Critical Files

1. **headline.json** - Entry point data
2. **script_clean.txt** - TTS input
3. **voice.mp3** - Audio track
4. **transcript_data.json** - Timing data
5. **visual_map.json** - Visual assets
6. **webpage_capture.mp4** - Opening video
7. **slideshow.mp4** - Assembled video
8. **final_video.mp4** - Final output

---

## Areas for Optimization

### 1. Unused Code
- `agent/__init__.py` references missing `smart_assembler` module
- `component_registry.py` has unused "merge" component
- Multiple deprecated video assembly functions

### 2. Efficiency Improvements
- **Parallel Processing**:
  - Visual search could run concurrent to audio generation
  - Multiple image downloads in parallel
  
- **Caching**:
  - Whisper model loaded repeatedly
  - FFmpeg path detection repeated
  
- **Memory Usage**:
  - Large video files loaded entirely
  - Could use streaming processing

### 3. Error Recovery
- Visual director crashes on missing files (by design?)
- Could implement partial video generation
- Better fallback for API failures

### 4. Configuration
- Hard-coded portrait orientation
- Could support multiple aspect ratios
- Runtime configuration changes

### 5. Code Duplication
- Screenshot and video capture share logic
- Multiple FFmpeg command builders
- Repeated standardization code

---

## Pipeline Characteristics

### Strengths
1. **Modular Design** - Clear component separation
2. **Error Handling** - Comprehensive fallbacks
3. **AI Integration** - Multiple LLM/Vision APIs
4. **Quality Control** - Validation at each step
5. **Debugging** - Extensive logging

### Weaknesses
1. **Sequential Processing** - No parallelization
2. **Memory Intensive** - Full file loading
3. **API Dependencies** - Multiple points of failure
4. **Fixed Orientation** - No flexibility
5. **Complex Setup** - Many dependencies

### Performance Metrics
- **Total Duration**: 2-5 minutes per video
- **Bottlenecks**:
  - Visual search/download (30-60s)
  - Video encoding (20-30s each)
  - API calls (variable)
- **Disk Usage**: ~500MB temporary files
- **Memory Peak**: ~2GB during video assembly

---

## Recommendations

### Immediate Improvements
1. Remove unused `smart_assembler` references
2. Fix component registry paths
3. Implement visual search parallelization
4. Add configuration for orientation
5. Cache Whisper model globally

### Long-term Enhancements
1. Streaming video processing
2. Distributed component execution
3. Local vision models option
4. Real-time progress tracking
5. Web UI for monitoring

### Code Cleanup
1. Remove deprecated functions
2. Consolidate FFmpeg utilities
3. Standardize error handling
4. Unify media processing
5. Extract common constants

---

## Conclusion

The AutoSocialMedia pipeline is a sophisticated system that successfully transforms news articles into engaging social media videos. While the architecture is sound and includes comprehensive error handling, there are opportunities for optimization in parallel processing, memory usage, and code organization. The modular design allows for easy maintenance and extension, making it a solid foundation for future enhancements.

The pipeline's strength lies in its AI integration and quality control mechanisms, ensuring professional output despite the complexity of coordinating multiple services and APIs. With the recommended optimizations, the system could achieve better performance while maintaining its current reliability and output quality.
