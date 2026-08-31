<!-- SYSTEM_PROMPT_START -->
You screen locally extracted Japanese term candidates for a Chinese subtitle knowledge base. Return JSON only.
<!-- SYSTEM_PROMPT_END -->

<!-- USER_PROMPT_START -->
The program has already extracted and accumulated the candidate strings below from
multiple documents. Do not discover additional terms and do not rewrite a candidate.

Keep a candidate only when encountering it again in future subtitles would benefit
from a stable Chinese rendering, entity disambiguation, or fan-domain background.
Be strict. Omit ordinary vocabulary, generic activities, incidental nouns, temporary
labels, viewer usernames, and likely ASR fragments. Frequency is supporting evidence,
not sufficient proof by itself.

Each item contains occurrence and document counts, source distribution, and up to
eight representative contexts. ASR can be wrong. Written official, SNS, title, and
manual-subtitle evidence is generally stronger, but sender claims are not automatically
official facts.

For a retained candidate:
- `candidate` must be copied exactly from the input.
- `canonical_zh` is a concise natural Simplified Chinese rendering. Identical CJK
  spelling is valid when appropriate. `さん` and `ちゃん` may be rendered as
  `桑` and `酱` in fan dialogue and lyrics.
- Use `accept` when the supplied evidence is enough.
- Use `search` only when the candidate is useful but its identity or established
  Chinese rendering genuinely requires web confirmation. Supply one concise query.
- Do not return rejected candidates.

Return only:
{"terms":[{"candidate":"exact input candidate","canonical_zh":"Chinese rendering","confidence":0.0,"action":"accept|search","search_query":""}]}

CANDIDATES:
{{CANDIDATES_JSON}}
<!-- USER_PROMPT_END -->
