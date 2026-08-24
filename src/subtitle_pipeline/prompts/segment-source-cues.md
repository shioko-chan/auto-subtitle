# Source Cue Segmentation Prompt

This is a runtime prompt template. Only text inside the SYSTEM_PROMPT and
USER_PROMPT markers is sent to the LLM.

<!-- SYSTEM_PROMPT_START -->
You divide source-language ASR evidence into natural subtitle cues.
<!-- SYSTEM_PROMPT_END -->

<!-- USER_PROMPT_START -->
Group every numbered TARGET unit into natural source-language subtitle cues.

Return exactly one JSON object and no explanation:
{"cues":[{"start_id":0,"end_id":2}]}

IDs are zero-based and ranges include both endpoints. A one-unit cue uses the
same start_id and end_id. Ranges must be ordered, contiguous, non-overlapping,
and cover every TARGET unit exactly once. Do not return source text,
translations, timestamps, speakers, Markdown, or extra fields.

Choose boundaries using the grammar and meaning of {{SOURCE_LANGUAGE}}. Local
units are candidates, not mandatory boundaries. Merge coherent adjacent units,
but every resulting source cue must be no wider than
{{SOURCE_MAXIMUM_UNITS}} display-width units. This is a hard limit. Split
conservatively when uncertain. Never split inside a TARGET unit or cross the
window edge.

For Japanese, do not begin a cue with a dependent particle such as を, が, に,
へ, と, で, から, まで, より, or の when it belongs to the preceding phrase.
TARGET may contain ASR errors, but this stage only chooses boundaries and must
not correct or translate text.

DIALOGUE_CONTEXT is read-only evidence. Never emit or cover its text.

DIALOGUE_CONTEXT:
{{DIALOGUE_CONTEXT}}

TARGET:
{{TARGET_TEXT}}{{RETRY_SECTION}}
<!-- USER_PROMPT_END -->
