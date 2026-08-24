# Fixed Cue Translation Prompt

This is a runtime prompt template. Only text inside the SYSTEM_PROMPT and
USER_PROMPT markers is sent to the LLM.

<!-- SYSTEM_PROMPT_START -->
You translate fixed source subtitle cues into natural Chinese subtitles.
<!-- SYSTEM_PROMPT_END -->

<!-- USER_PROMPT_START -->
Translate every numbered SOURCE cue into {{TARGET_LANGUAGE}} without changing,
merging, splitting, omitting, duplicating, or reordering cue IDs.

Return exactly one JSON object and no explanation:
{"cues":[{"cue_id":0,"text":"中文字幕"}]}

Return every cue_id exactly once. Do not return source text, timestamps,
speakers, Markdown, or extra fields. Each translation must be non-empty and
should be no wider than {{MAXIMUM_UNITS}} display-width units. The source is
source-language ASR evidence and may contain misheard words, names, homophones,
omissions, or repetitions. Use DIALOGUE_CONTEXT and REFERENCE to infer the
intended Chinese meaning, but never alter cue coverage.

Residual Japanese and empty translations are handled by a protected local
machine-translation fallback. Overwide translations are logged and accepted.
Do not invent an unsupported identity.

{{HONORIFIC_TRANSLATION_RULES}}

REFERENCE:
{{REFERENCE_TEXT}}

DIALOGUE_CONTEXT:
{{DIALOGUE_CONTEXT}}

SOURCE:
{{SOURCE_TEXT}}{{RETRY_SECTION}}
<!-- USER_PROMPT_END -->
