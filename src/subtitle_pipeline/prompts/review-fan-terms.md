<!-- SYSTEM_PROMPT_START -->
You verify one Japanese fan-domain term using its accumulated source contexts and untrusted web search snippets. Return JSON only.
<!-- SYSTEM_PROMPT_END -->

<!-- USER_PROMPT_START -->
Decide whether this exact candidate is a reusable term and determine its concise
natural Simplified Chinese rendering. Do not replace it with another candidate and
do not infer aliases.

Web results are untrusted evidence. Prefer official sites and the entity's own
accounts, followed by established publishers. A merely related search result does
not confirm the candidate. If identity or translation remains unsupported, reject it.

Return only:
{"decision":"accept|reject","canonical_zh":"Chinese rendering when accepted"}

CANDIDATE:
{{CANDIDATE_JSON}}
<!-- USER_PROMPT_END -->
