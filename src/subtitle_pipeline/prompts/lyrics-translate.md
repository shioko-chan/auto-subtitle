# Lyrics Translation Prompt

<!-- SYSTEM_PROMPT_START -->
You translate published Japanese song lyrics into natural Simplified Chinese lyrics.
<!-- SYSTEM_PROMPT_END -->

<!-- USER_PROMPT_START -->
Translate every numbered lyric line. Preserve imagery, voice, repetition, and
line-to-line correspondence. Do not merge, split, omit, or reorder lines. Names
and established terms must follow REFERENCE.

Return exactly one JSON object and no explanation:
{"lines":[{"line_id":0,"text":"中文字幕"}]}

SONG: {{SONG_TITLE}}
ARTIST: {{ARTIST}}
REFERENCE:
{{REFERENCE_TEXT}}

LYRICS:
{{LYRICS_TEXT}}
<!-- USER_PROMPT_END -->
