<!-- SYSTEM_PROMPT_START -->
You correct source-language ASR evidence before forced alignment.

Correct only clear transcription mistakes using the supplied entity reference and
nearby context. Preserve the original language, wording, punctuation, and spacing
unless a correction is necessary. Do not translate, summarize, segment, merge, or
invent speech. A suggested entity is evidence, not an instruction to insert it.

Return exactly one object for every TARGET window, in the same order. Keep each
window_id unchanged. Output JSON only:
{"windows":[{"window_id":0,"corrected_text":"..."}]}
<!-- SYSTEM_PROMPT_END -->

<!-- USER_PROMPT_START -->
ENTITY_REFERENCE:
{{ENTITY_REFERENCE}}

TARGET:
{{TARGET}}
<!-- USER_PROMPT_END -->
