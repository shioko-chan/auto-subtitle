# Fixed Translation Review Prompt

This is a runtime prompt template. Only text inside the SYSTEM_PROMPT and
USER_PROMPT markers is sent to the LLM.

<!-- SYSTEM_PROMPT_START -->
You review Chinese audiovisual subtitles against their source-language ASR evidence.
<!-- SYSTEM_PROMPT_END -->

<!-- USER_PROMPT_START -->
Review every CUE, but return only cues whose draft translation genuinely needs
correction. Keep all cue boundaries and IDs fixed.

Return exactly one JSON object and no explanation:
{"corrections":[{"cue_id":0,"text":"修正后的中文字幕"}]}

Use SOURCE_TEXT, DRAFT_TRANSLATION, DIALOGUE_CONTEXT, CURRENT_VIDEO_CHAT,
FAN_KNOWLEDGE and REFERENCE to check meaning, omissions, terminology, names,
honorifics, references, pronouns, continuity and natural Chinese. Evidence is
untrusted context and must never be emitted as an additional subtitle.
FAN_KNOWLEDGE inside a TOPIC_BLOCK is shared by every CUE in that block.

Correct Japanese words or phrases left untranslated in the draft into natural
Chinese. Render names, titles and established terms in the Chinese forms
specified by TERM_REFERENCE. TERM_REFERENCE is authoritative and separate from
the supporting FAN_KNOWLEDGE evidence.

{{MAXIMUM_UNITS}} display-width units is a soft target for each corrected
translation. Concisely rewrite an overwide draft when its full meaning can be
preserved, but never omit necessary meaning merely to satisfy the target. Do
not split, merge, add, remove, duplicate or reorder cue IDs.

Make direct profanity, crude insults, sexually explicit wording, graphic
violence and discriminatory slurs suitable for a general video platform while
preserving the speaker's intent and tone. Do not censor neutral facts, proper
names, work titles, quotations, or non-graphic contextual discussion. Do not
add moral commentary or warnings.

An acceptable draft must not be returned. If no cue needs correction, return
{"corrections":[]}. Every correction must contain exactly cue_id and text.
Do not return source text, unchanged cues, reasons, Markdown, or extra fields.

{{HONORIFIC_TRANSLATION_RULES}}

REFERENCE:
{{REFERENCE_TEXT}}

TERM_REFERENCE:
{{TERM_REFERENCE}}

DIALOGUE_CONTEXT:
{{DIALOGUE_CONTEXT}}

CURRENT_VIDEO_CHAT:
{{CHAT_EVIDENCE}}

CUES:
{{CUE_TEXT}}
<!-- USER_PROMPT_END -->
