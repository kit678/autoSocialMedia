# Script & Visual Co-Generation Plan

## The Problem We're Solving

Currently, we generate scripts first and then try to find visuals that match. This approach has limitations:
- The script writer doesn't know what visuals are available
- We end up with mismatched or generic visuals
- No variety in visual sources across segments
- Script segments aren't optimized for visual storytelling

## The Solution: Generate Scripts and Visuals Together

Instead of treating script and visuals as separate steps, we'll create them as a team effort where they inform each other.

## How It Works

### Step 1: Article Analysis
- Take the article content and break it down into key points
- Identify important people, places, concepts, and emotions
- Understand the timeline and flow of information

### Step 2: Smart Script Generation
When writing the script, the AI will create each segment with special metadata:

```json
{
  "segment_id": "seg_03",
  "narration_text": "NASA's Webb telescope just spotted something amazing in deep space...",
  "segment_type": "scientific_explanation", 
  "emotional_tone": "wonder",
  "key_entities": ["NASA", "James Webb Space Telescope", "deep space"],
  "visual_needs": ["telescope_imagery", "space_scenes"],
  "preferred_media": "video"
}
```

### Step 3: Visual Adapter Selection
For each segment, we automatically choose the best visual sources:

**Example Mapping Rules:**
- **Breaking news** → Use GDELT TV or Archive TV (real news footage)
- **Science/space topics** → Use NASA adapter (authentic space imagery)  
- **Emotional moments** → Use Tenor (reaction GIFs)
- **Historical references** → Use Wikimedia or Openverse (proper attribution)
- **Background ambiance** → Use Coverr (professional loops) or Pexels
- **General concepts** → Use SearXNG (wide web search) then Pexels

### Step 4: Asset Search & Selection
- Search the chosen adapters with smart queries
- Score results based on relevance, quality, and licensing
- Pick the best assets for each segment
- Store the chosen visual with each script segment

### Step 5: Quality Check & Refinement
- If no good visuals are found, the script generator can:
  - Rewrite the segment to be more visual-friendly
  - Mark it as "voice-only" 
  - Suggest alternative phrasing

## Key Benefits

1. **Better Visual Variety**: Each segment can use different visual sources based on content type
2. **Smarter Script Writing**: The narrator knows what visuals will be available
3. **Higher Quality Matches**: Visuals are selected based on segment context, not just keywords
4. **Flexible Licensing**: Automatically chooses appropriate sources (public domain for science, fair use for news, etc.)
5. **Consistent Quality**: Both script and visuals are optimized together

## Visual Adapter Selection Strategy

### Our 9 Visual Sources:
1. **Pexels** - High-quality stock photos/videos
2. **SearXNG** - Web image search
3. **Tenor** - Reaction GIFs and emotions  
4. **Openverse** - Creative Commons content
5. **Wikimedia** - Educational/historical content
6. **NASA** - Space and science imagery
7. **Coverr** - Professional video loops
8. **GDELT TV** - Current news clips
9. **Archive TV** - Historical news footage

### Smart Selection Rules:
- **News segments** → GDELT TV → Archive TV → SearXNG
- **Science topics** → NASA → Wikimedia → Pexels
- **Emotional beats** → Tenor → Pexels (close-ups) 
- **Background footage** → Coverr → Pexels → Openverse
- **Historical references** → Wikimedia → Openverse → Archive TV
- **General concepts** → SearXNG → Pexels → Openverse

## Implementation Steps

1. **Create segment metadata format** - Define the JSON structure for script segments
2. **Build Adapter Selector** - Smart logic to choose the right visual sources
3. **Integrate with script generation** - Modify script writer to output metadata
4. **Add visual search coordination** - Connect segment metadata to adapter searches
5. **Create feedback loop** - Allow script refinement if visuals aren't good enough

## Expected Outcome

- More engaging videos with varied, contextually appropriate visuals
- Faster production pipeline (no manual visual hunting)
- Better licensing compliance (automatic source selection)
- Scripts written with visual storytelling in mind
- Higher viewer engagement through better script-visual harmony

---

*This approach transforms video creation from a linear process (script → visuals) into a collaborative process (script ↔ visuals) where both elements enhance each other.*
