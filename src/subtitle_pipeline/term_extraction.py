from __future__ import annotations

import json
import os
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from .fan_knowledge import (
    ExtractedTerm,
    FanKnowledgeRetriever,
    PendingTermDocument,
    TermCandidateOccurrence,
    TermExtractionChunk,
)
from .llm_response import (
    finish_reason,
    parse_json_object,
    structured_request_body,
    structured_response_content,
)
from .prompt_budget import estimate_prompt_tokens
from .prompt_templates import render_user_prompt
from .translate import LLMHTTPError, is_llm_quota_exhausted

_DEFAULT_CONTEXT_SIZE = 262144
_DEFAULT_TARGET_INPUT_TOKENS = 220000
_DEFAULT_MAX_OUTPUT_TOKENS = 16384
_AUDIT_LOCK = threading.Lock()
_WEB_RESULT_LIMIT = 4
_MAX_BACKGROUND_ZIPF = 4.8
_MAX_ENGLISH_ZIPF = 4.0
_LOCAL_SCAN_WORKERS = min(4, os.cpu_count() or 1)
_SC_LINE_RE = re.compile(r"^SC\s+\S+\s+(.+?):\s*(.*)$")
_KATAKANA_TERM_RE = re.compile(r"[ァ-ヶー][ァ-ヶー・]{2,31}")
_LATIN_TERM_RE = re.compile(r"(?<![\w])[@#＃]?[A-Za-z][A-Za-z0-9+_.-]{1,31}")
_QUOTED_TERM_RE = re.compile(r"[「『【]([^」』】\n]{2,40})[」』】]")
_JAPANESE_HASHTAG_RE = re.compile(r"[#＃][ぁ-んァ-ヶー一-龯々A-Za-z0-9_]{2,40}")
_MIXED_NAME_RE = re.compile(
    r"(?:[一-龯々]{2,}[ぁ-んー]{2,16}(?=の|を|が|は|で|に|と|、|。|$)"
    r"|[ぁ-んー]{2,16}[ァ-ヶー]{2,16}(?=[一-龯々]|の|を|が|は|で|に|と|、|。|$))"
)
_SOURCE_URL_RE = re.compile(
    r"(?i)(?:(?:https?://|www\.)[^\s<>\"'）】]+|"
    r"\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?:/[^\s<>\"'）】]*)?)"
)
_SOURCE_EMAIL_RE = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_SOURCE_QUERY_RE = re.compile(r"(?:^|\s)[?&](?:[\w.-]+=[^\s&]+&?)+")
_DATE_ONLY_RE = re.compile(
    r"^(?:(?:令和|平成|昭和)\s*)?\d{1,4}(?:[-/.年月日]\d{1,2})*日?$"
)
_NUMERIC_SYMBOLIC_NAME_RE = re.compile(r"^\d{1,3}/\d{1,3}$")
_NUMERIC_PREFIX_NAME_RE = re.compile(r"^\d{1,3}[A-Z][A-Za-z0-9]*")
_MARKUP_RE = re.compile(
    r"(?:<[^>]+>|&(?:#\d+|#x[0-9a-f]+|[a-z]+);|[{}<>])", re.IGNORECASE
)
_NOISY_EDGE_RE = re.compile(
    r"^(?:[^\wぁ-んァ-ヶー一-龯々@#＃]{2,})|"
    r"(?:[^\wぁ-んァ-ヶー一-龯々]{2,})$"
)
_GINZA_TERM_LABELS = frozenset(
    {
        "Person",
        "Company",
        "Organization_Other",
        "Show_Organization",
        "Game",
        "Music",
        "Movie",
        "Book",
        "Magazine",
        "Character",
        "Event_Other",
        "Conference",
        "Festival",
        "Sports_Event",
    }
)
_LOCAL_TERM_TOKENIZER = threading.local()


class TermExtractionOutputTooLong(RuntimeError):
    pass


@dataclass(frozen=True)
class TermExtractionSummary:
    documents: int = 0
    batches: int = 0
    candidates: int = 0
    stored: int = 0
    published: int = 0


@dataclass(frozen=True)
class TermPreparationSummary:
    documents: int = 0
    local_candidates: int = 0
    screenable_candidates: int = 0
    known_candidates: int = 0
    pending_candidates: int = 0


@dataclass(frozen=True)
class LocalTermOccurrence:
    document_id: str
    logical_document_id: str
    chunk_id: str
    context: str
    source_type: str
    reliability: float
    strong_name_evidence: bool = False
    occurrence_count: int = 1


@dataclass(frozen=True)
class LocalTermCandidate:
    surface: str
    occurrences: tuple[LocalTermOccurrence, ...]

    @property
    def document_count(self) -> int:
        return len({value.logical_document_id for value in self.occurrences})

    @property
    def source_counts(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for occurrence in self.occurrences:
            counts[occurrence.source_type] += occurrence.occurrence_count
        return counts



def extract_pending_terms(
    retriever: FanKnowledgeRetriever,
    *,
    request: Callable[[dict[str, object]], dict[str, object]],
    model: str,
    max_tokens: int,
    thinking: str | None,
    max_retries: int,
    maximum_documents: int | None = None,
    maximum_candidates: int | None = None,
    maximum_new_terms: int | None = None,
    include_backfill: bool = False,
    audit_path: Path | None = None,
    max_concurrency: int = 3,
    search_web: Callable[[str], dict[str, object]] | None = None,
    context_size: int = _DEFAULT_CONTEXT_SIZE,
    target_input_tokens: int = _DEFAULT_TARGET_INPUT_TOKENS,
) -> TermExtractionSummary:
    preparation = prepare_pending_terms(
        retriever,
        maximum_documents=maximum_documents,
        include_backfill=include_backfill,
    )
    validation = validate_pending_terms(
        retriever,
        request=request,
        model=model,
        max_tokens=max_tokens,
        thinking=thinking,
        max_retries=max_retries,
        audit_path=audit_path,
        max_concurrency=max_concurrency,
        search_web=search_web,
        context_size=context_size,
        target_input_tokens=target_input_tokens,
        include_backfill=include_backfill,
        maximum_candidates=maximum_candidates,
        maximum_new_terms=maximum_new_terms,
    )
    return replace(validation, documents=preparation.documents)


def validate_pending_terms(
    retriever: FanKnowledgeRetriever,
    *,
    request: Callable[[dict[str, object]], dict[str, object]],
    model: str,
    max_tokens: int,
    thinking: str | None,
    max_retries: int,
    maximum_candidates: int | None = None,
    maximum_new_terms: int | None = None,
    include_backfill: bool = False,
    audit_path: Path | None = None,
    max_concurrency: int = 3,
    search_web: Callable[[str], dict[str, object]] | None = None,
    context_size: int = _DEFAULT_CONTEXT_SIZE,
    target_input_tokens: int = _DEFAULT_TARGET_INPUT_TOKENS,
) -> TermExtractionSummary:
    queued_forms = retriever.pending_term_review_forms(
        maximum_candidates, include_backfill=include_backfill
    )
    candidates = _candidates_from_stored_occurrences(
        retriever.term_candidate_occurrences(queued_forms)
    )
    candidate_order = {value: index for index, value in enumerate(queued_forms)}
    candidates.sort(
        key=lambda candidate: candidate_order[
            _normalize_candidate(candidate.surface)
        ]
    )
    eligible_forms = {
        _normalize_candidate(candidate.surface) for candidate in candidates
    }
    retriever.complete_term_reviews(
        value for value in queued_forms if value not in eligible_forms
    )
    if not candidates:
        return TermExtractionSummary()
    accepted: list[ExtractedTerm] = []
    known_surfaces: set[str] = set()
    pending_candidates: list[LocalTermCandidate] = []
    for candidate in candidates:
        known = retriever.known_term_mapping(candidate.surface, ())
        if known is None:
            pending_candidates.append(candidate)
            continue
        known_surfaces.add(_normalize_candidate(candidate.surface))
        accepted.append(
            ExtractedTerm(
                surface=candidate.surface,
                canonical_zh=known.canonical_zh,
                aliases=(),
                reading=known.reading,
                relation="name",
                evidence_chunk_ids=tuple(
                    dict.fromkeys(
                        occurrence.chunk_id
                        for occurrence in candidate.occurrences
                    )
                ),
                confidence=known.confidence,
            )
        )
        _write_audit(
            audit_path,
            {
                "event": "term_candidate_reused",
                "candidate": candidate.surface,
                "canonical_zh": known.canonical_zh,
                "curated": known.curated,
                "evidence_count": len(candidate.occurrences),
            },
        )
    request_options = {
        "request": request,
        "model": model,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "max_retries": max_retries,
        "audit_path": audit_path,
        "search_web": search_web,
        "context_size": context_size,
        "target_input_tokens": target_input_tokens,
    }
    batches = _candidate_batches(
        pending_candidates,
        context_size=context_size,
        max_tokens=max_tokens,
        target_input_tokens=target_input_tokens,
    )
    errors: list[Exception] = []
    if batches:
        with ThreadPoolExecutor(
            max_workers=min(max(1, max_concurrency), len(batches)),
            thread_name_prefix="term-extraction",
        ) as executor:
            futures = {
                executor.submit(
                    _screen_candidate_batch,
                    batch,
                    batch_index=index,
                    **request_options,
                ): batch
                for index, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                try:
                    accepted.extend(future.result())
                except Exception as exc:
                    if is_llm_quota_exhausted(exc):
                        for pending in futures:
                            pending.cancel()
                        raise
                    errors.append(exc)
                    _write_audit(
                        audit_path,
                        {
                            "event": "term_extraction_batch_failed",
                            "candidates": [value.surface for value in futures[future]],
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
    if errors:
        raise RuntimeError(f"term extraction failed for {len(errors)} batch(es)") from errors[0]

    model_terms = [
        term
        for term in accepted
        if _normalize_candidate(term.surface) not in known_surfaces
    ]
    model_terms.sort(
        key=lambda term: candidate_order[_normalize_candidate(term.surface)]
    )
    deferred_forms: set[str] = set()
    if maximum_new_terms is not None and len(model_terms) > maximum_new_terms:
        deferred_forms = {
            _normalize_candidate(term.surface)
            for term in model_terms[maximum_new_terms:]
        }
        model_terms = model_terms[:maximum_new_terms]
    accepted = [
        term
        for term in accepted
        if _normalize_candidate(term.surface) in known_surfaces
    ] + model_terms
    stored, published = _store_terms_by_document(retriever, candidates, accepted)
    retriever.complete_term_reviews(
        _normalize_candidate(candidate.surface)
        for candidate in candidates
        if _normalize_candidate(candidate.surface) not in deferred_forms
    )
    return TermExtractionSummary(
        documents=0,
        batches=len(batches),
        candidates=len(pending_candidates),
        stored=stored,
        published=published,
    )


def prepare_pending_terms(
    retriever: FanKnowledgeRetriever,
    *,
    maximum_documents: int | None = None,
    include_backfill: bool = False,
) -> TermPreparationSummary:
    retriever.reconcile_extracted_terms_with_curated()
    documents = retriever.pending_term_documents(
        maximum_documents, include_backfill=include_backfill
    )
    if not documents:
        return TermPreparationSummary()
    touched_forms = retriever.term_candidate_forms_for_documents(
        document.document_id for document in documents
    )
    fresh_candidates = _collect_local_term_candidates(documents, screenable_only=False)
    occurrences_by_document: dict[str, list[TermCandidateOccurrence]] = defaultdict(list)
    for candidate in fresh_candidates:
        normalized_form = _normalize_candidate(candidate.surface)
        touched_forms.add(normalized_form)
        for occurrence in candidate.occurrences:
            occurrences_by_document[occurrence.document_id].append(
                TermCandidateOccurrence(
                    normalized_form=normalized_form,
                    surface=candidate.surface,
                    document_id=occurrence.document_id,
                    logical_document_id=occurrence.logical_document_id,
                    chunk_id=occurrence.chunk_id,
                    context=occurrence.context,
                    source_type=occurrence.source_type,
                    reliability=occurrence.reliability,
                    strong_name_evidence=occurrence.strong_name_evidence,
                    occurrence_count=occurrence.occurrence_count,
                )
            )
    retriever.replace_term_candidate_occurrences(
        (document.document_id for document in documents),
        (
            occurrence
            for document in documents
            for occurrence in occurrences_by_document.get(document.document_id, ())
        ),
    )
    candidates = _candidates_from_stored_occurrences(
        retriever.term_candidate_occurrences(touched_forms)
    )
    known_terms: list[ExtractedTerm] = []
    pending_review_forms: list[str] = []
    known_candidates = 0
    for candidate in candidates:
        known = retriever.known_term_mapping(candidate.surface, ())
        if known is None:
            pending_review_forms.append(_normalize_candidate(candidate.surface))
            continue
        known_candidates += 1
        known_terms.append(
            ExtractedTerm(
                surface=candidate.surface,
                canonical_zh=known.canonical_zh,
                aliases=(),
                reading=known.reading,
                relation="name",
                evidence_chunk_ids=tuple(
                    dict.fromkeys(
                        occurrence.chunk_id for occurrence in candidate.occurrences
                    )
                ),
                confidence=known.confidence,
            )
        )
    _store_terms_by_document(retriever, candidates, known_terms)
    retriever.finish_term_preparation(
        documents,
        touched_forms=touched_forms,
        pending_review_forms=pending_review_forms,
        backfill=include_backfill,
    )
    return TermPreparationSummary(
        documents=len(documents),
        local_candidates=len(fresh_candidates),
        screenable_candidates=len(candidates),
        known_candidates=known_candidates,
        pending_candidates=len(pending_review_forms),
    )


def _store_terms_by_document(
    retriever: FanKnowledgeRetriever,
    candidates: list[LocalTermCandidate],
    terms: list[ExtractedTerm],
) -> tuple[int, int]:
    if not terms:
        return 0, 0
    document_ids = {
        occurrence.document_id
        for candidate in candidates
        for occurrence in candidate.occurrences
    }
    stored = 0
    published = 0
    for document in retriever.term_documents(document_ids):
        chunk_ids = {chunk.chunk_id for chunk in document.chunks}
        document_terms = [
            replace(
                term,
                evidence_chunk_ids=tuple(
                    value for value in term.evidence_chunk_ids if value in chunk_ids
                ),
            )
            for term in terms
            if any(value in chunk_ids for value in term.evidence_chunk_ids)
        ]
        document_stored, document_published = retriever.store_extracted_terms(
            document, document_terms
        )
        stored += document_stored
        published += document_published
    return stored, published


def _collect_local_term_candidates(
    documents: list[PendingTermDocument],
    *,
    screenable_only: bool = True,
) -> list[LocalTermCandidate]:
    occurrences: dict[str, list[LocalTermOccurrence]] = defaultdict(list)
    surfaces: dict[str, str] = {}
    entries: list[tuple[PendingTermDocument, TermExtractionChunk, str, str | None]] = []
    for document in documents:
        for chunk_index, chunk in enumerate(document.chunks):
            for text, sender in _candidate_source_entries(document, chunk):
                candidate_text = _strip_candidate_source_artifacts(text)
                if not candidate_text:
                    continue
                candidate_title = _strip_candidate_source_artifacts(document.title)
                extraction_text = (
                    f"{candidate_title}\n{candidate_text}"
                    if chunk_index == 0
                    and candidate_title
                    and document.source_type
                    not in {
                        "youtube_auto_subtitle",
                        "youtube_manual_subtitle",
                        "youtube_super_chat",
                        "youtube_live_chat",
                    }
                    else candidate_text
                )
                entries.append((document, chunk, extraction_text, sender))
    ginza_indices = [
        index
        for index, (document, _chunk, _text, _sender) in enumerate(entries)
        if document.source_type
        not in {
            "youtube_auto_subtitle",
            "youtube_super_chat",
            "youtube_live_chat",
        }
    ]
    ginza_selected = _ginza_term_surfaces_batch(
        [entries[index][2] for index in ginza_indices]
    )
    ginza_by_index = dict(zip(ginza_indices, ginza_selected, strict=True))
    local_selected = _local_term_surfaces_batch([entry[2] for entry in entries])
    for index, (document, chunk, extraction_text, sender) in enumerate(entries):
        ginza_surfaces = ginza_by_index.get(index, ())
        for surface in dict.fromkeys(
            (*local_selected[index], *ginza_surfaces)
        ):
            normalized = _normalize_candidate(surface)
            if not normalized:
                continue
            surfaces.setdefault(normalized, surface)
            local_context = _text_near_surface(extraction_text, surface)
            context = _candidate_context(
                document, chunk, local_context, sender=sender
            )
            count = max(1, extraction_text.count(surface))
            strong_name_evidence = (
                surface in ginza_surfaces
                or _obvious_name_form(surface)
                or _explicitly_quoted(surface, extraction_text)
            )
            occurrences[normalized].append(
                LocalTermOccurrence(
                    document_id=document.document_id,
                    logical_document_id=_logical_document_id(document),
                    chunk_id=chunk.chunk_id,
                    context=context,
                    source_type=document.source_type,
                    reliability=document.reliability,
                    strong_name_evidence=strong_name_evidence,
                    occurrence_count=count,
                )
            )
    return [
        LocalTermCandidate(surfaces[key], tuple(values))
        for key, values in sorted(occurrences.items())
        if not screenable_only
        or _candidate_is_worth_screening(surfaces[key], values)
    ]


def _candidates_from_stored_occurrences(
    occurrences: list[TermCandidateOccurrence],
) -> list[LocalTermCandidate]:
    grouped: dict[str, list[LocalTermOccurrence]] = defaultdict(list)
    surfaces: dict[str, str] = {}
    for occurrence in occurrences:
        surfaces.setdefault(occurrence.normalized_form, occurrence.surface)
        grouped[occurrence.normalized_form].append(
            LocalTermOccurrence(
                document_id=occurrence.document_id,
                logical_document_id=occurrence.logical_document_id,
                chunk_id=occurrence.chunk_id,
                context=occurrence.context,
                source_type=occurrence.source_type,
                reliability=occurrence.reliability,
                strong_name_evidence=occurrence.strong_name_evidence,
                occurrence_count=occurrence.occurrence_count,
            )
        )
    return [
        LocalTermCandidate(surfaces[key], tuple(values))
        for key, values in sorted(grouped.items())
        if _candidate_is_worth_screening(surfaces[key], values)
    ]


def _normalize_candidate(surface: str) -> str:
    return unicodedata.normalize("NFKC", surface).strip().casefold()


def _candidate_source_text(
    document: PendingTermDocument, chunk: TermExtractionChunk
) -> str:
    return "\n".join(
        text for text, _sender in _candidate_source_entries(document, chunk)
    )


def _candidate_source_entries(
    document: PendingTermDocument, chunk: TermExtractionChunk
) -> tuple[tuple[str, str | None], ...]:
    if document.source_type == "youtube_super_chat":
        messages: list[tuple[str, str | None]] = []
        for line in chunk.text.splitlines():
            match = _SC_LINE_RE.match(line.strip())
            text = (match.group(2) if match else line.strip()).strip()
            sender = match.group(1).strip() if match else None
            if text:
                messages.append((text, sender))
        return tuple(messages)
    text = chunk.text.strip()
    return ((text, document.author),) if text else ()


def _strip_candidate_source_artifacts(text: str) -> str:
    value = _SOURCE_EMAIL_RE.sub(" ", text)
    value = _SOURCE_URL_RE.sub(" ", value)
    value = _SOURCE_QUERY_RE.sub(" ", value)
    return " ".join(value.split())


def _local_term_surfaces(text: str) -> tuple[str, ...]:
    values: list[str] = []
    values.extend(match.group(1).strip() for match in _QUOTED_TERM_RE.finditer(text))
    values.extend(match.group(0) for match in _JAPANESE_HASHTAG_RE.finditer(text))
    values.extend(match.group(0) for match in _MIXED_NAME_RE.finditer(text))
    values.extend(match.group(0) for match in _KATAKANA_TERM_RE.finditer(text))
    values.extend(match.group(0) for match in _LATIN_TERM_RE.finditer(text))
    values.extend(_sudachi_term_surfaces(text))
    return tuple(
        dict.fromkeys(
            _clean_local_surface(value)
            for value in values
            if 2 <= len(_clean_local_surface(value)) <= 40
        )
    )


def _local_term_surfaces_batch(texts: list[str]) -> list[tuple[str, ...]]:
    if len(texts) <= 1 or _LOCAL_SCAN_WORKERS == 1:
        return [_local_term_surfaces(text) for text in texts]
    worker_count = min(_LOCAL_SCAN_WORKERS, len(texts))
    batch_size = (len(texts) + worker_count - 1) // worker_count
    ranges = [
        (start, min(start + batch_size, len(texts)))
        for start in range(0, len(texts), batch_size)
    ]
    results: list[tuple[str, ...]] = [()] * len(texts)
    with ThreadPoolExecutor(
        max_workers=len(ranges), thread_name_prefix="term-local-scan"
    ) as executor:
        futures = {
            executor.submit(_local_term_surface_slice, texts[start:end]): (start, end)
            for start, end in ranges
        }
        for future in as_completed(futures):
            start, end = futures[future]
            results[start:end] = future.result()
    return results


def _local_term_surface_slice(texts: list[str]) -> list[tuple[str, ...]]:
    return [_local_term_surfaces(text) for text in texts]


def _ginza_term_surfaces_batch(texts: list[str]) -> list[tuple[str, ...]]:
    if not texts:
        return []
    try:
        import spacy

        nlp = getattr(_LOCAL_TERM_TOKENIZER, "ginza", None)
        if nlp is None:
            nlp = spacy.load(
                "ja_ginza", exclude=["compound_splitter", "parser", "textcat"]
            )
            _LOCAL_TERM_TOKENIZER.ginza = nlp
    except (ImportError, OSError):
        return [() for _text in texts]
    return [
        _ginza_document_surfaces(document)
        for document in nlp.pipe(
            texts,
            batch_size=64,
            n_process=min(_LOCAL_SCAN_WORKERS, len(texts)),
        )
    ]


def _ginza_document_surfaces(document: object) -> tuple[str, ...]:
    values = [
        entity.text.strip()
        for entity in document.ents
        if entity.label_ in _GINZA_TERM_LABELS
    ]
    return tuple(
        dict.fromkeys(
            cleaned
            for value in values
            if 2 <= len(cleaned := _clean_local_surface(value)) <= 40
        )
    )


def _clean_local_surface(value: str) -> str:
    cleaned = value.strip(" \t\r\n、。！？!?()（）[]［］")
    if len(cleaned) >= 4 and cleaned[0] in "のはがをにでともへや":
        cleaned = cleaned[1:]
    return cleaned


def _sudachi_term_surfaces(text: str) -> tuple[str, ...]:
    try:
        from sudachipy import dictionary, tokenizer

        instance = getattr(_LOCAL_TERM_TOKENIZER, "sudachi", None)
        if instance is None:
            instance = dictionary.Dictionary().create()
            _LOCAL_TERM_TOKENIZER.sudachi = instance
        morphemes = instance.tokenize(text, tokenizer.Tokenizer.SplitMode.C)
    except (ImportError, OSError):
        return ()
    values: list[str] = []
    run: list[object] = []

    def flush() -> None:
        if not run:
            return
        proper = any(value.part_of_speech()[1] == "固有名詞" for value in run)
        if proper or len(run) >= 2:
            values.append("".join(value.surface() for value in run))
        run.clear()

    for morpheme in morphemes:
        if morpheme.part_of_speech()[0] == "名詞":
            run.append(morpheme)
        else:
            flush()
    flush()
    return tuple(values)


def _candidate_is_worth_screening(
    surface: str, occurrences: list[LocalTermOccurrence]
) -> bool:
    compact = surface.lstrip("@#＃").strip()
    if not _candidate_has_valid_structure(surface):
        return False
    if _english_zipf_frequency(compact) > _MAX_ENGLISH_ZIPF:
        return False
    if _background_zipf_frequency(compact) > _MAX_BACKGROUND_ZIPF:
        return False
    document_count = len({value.logical_document_id for value in occurrences})
    return document_count >= 2 or any(
        value.strong_name_evidence for value in occurrences
    )


def _candidate_has_valid_structure(surface: str) -> bool:
    value = unicodedata.normalize("NFKC", surface).strip()
    compact = value.lstrip("@#＃").strip()
    if any(character in value for character in "\r\n\t"):
        return False
    numeric_symbolic_name = bool(_NUMERIC_SYMBOLIC_NAME_RE.fullmatch(compact))
    numeric_prefix_name = bool(_NUMERIC_PREFIX_NAME_RE.match(compact))
    if not compact or not (
        re.search(r"[A-Za-zぁ-んァ-ヶー一-龯々]", compact)
        or numeric_symbolic_name
    ):
        return False
    if unicodedata.category(compact[0]).startswith("N") and not (
        numeric_symbolic_name or numeric_prefix_name
    ):
        return False
    if (
        _DATE_ONLY_RE.fullmatch(compact)
        and not numeric_symbolic_name
        or _MARKUP_RE.search(compact)
    ):
        return False
    return not (
        compact.endswith(("...", "…", "……")) or _NOISY_EDGE_RE.search(value)
    )


def _english_zipf_frequency(surface: str) -> float:
    if not re.fullmatch(r"[A-Za-z]+", surface):
        return 0.0
    try:
        from wordfreq import zipf_frequency

        return zipf_frequency(surface, "en")
    except (ImportError, LookupError):
        return 0.0


def _background_zipf_frequency(surface: str) -> float:
    try:
        from sudachipy import dictionary, tokenizer
        from wordfreq import freq_to_zipf, get_frequency_dict

        instance = getattr(_LOCAL_TERM_TOKENIZER, "sudachi", None)
        if instance is None:
            instance = dictionary.Dictionary().create()
            _LOCAL_TERM_TOKENIZER.sudachi = instance
        tokens = [
            value.surface().casefold()
            for value in instance.tokenize(surface, tokenizer.Tokenizer.SplitMode.A)
            if value.surface().strip()
        ]
        frequencies = get_frequency_dict("ja")
    except (ImportError, LookupError, OSError):
        return 0.0
    token_frequencies = [frequencies.get(value, 0.0) for value in tokens]
    if not token_frequencies or any(value <= 0.0 for value in token_frequencies):
        return 0.0
    combined_frequency = 1.0 / sum(1.0 / value for value in token_frequencies)
    return freq_to_zipf(combined_frequency)


def _logical_document_id(document: PendingTermDocument) -> str:
    if document.source_type.startswith("youtube_"):
        video_id = document.metadata.get("video_id") or document.metadata.get("id")
        if isinstance(video_id, str) and video_id.strip():
            return f"youtube:{video_id.strip()}"
        if document.external_id:
            return f"youtube:{document.external_id.split(':', 1)[0]}"
    return document.document_id


def _obvious_name_form(surface: str) -> bool:
    if _JAPANESE_HASHTAG_RE.fullmatch(surface):
        return True
    if not _LATIN_TERM_RE.fullmatch(surface):
        return False
    compact = surface.lstrip("@#＃")
    return bool(
        surface.startswith(("@", "#", "＃"))
        or any(character.isdigit() or character in "_+-" for character in compact)
        or (any(character.isupper() for character in compact) and not compact.isupper())
    )


def _explicitly_quoted(surface: str, text: str) -> bool:
    return any(
        f"{opening}{surface}{closing}" in text
        for opening, closing in (("「", "」"), ("『", "』"), ("【", "】"))
    )


def _candidate_context(
    document: PendingTermDocument,
    chunk: TermExtractionChunk,
    text: str,
    *,
    sender: str | None = None,
) -> str:
    label = {
        "youtube_auto_subtitle": "ASR",
        "youtube_manual_subtitle": "SUBTITLE",
        "youtube_super_chat": "SC",
        "youtube_metadata": "VIDEO",
        "x_post": "X",
        "instagram_post": "Instagram",
    }.get(
        document.source_type,
        "官网" if document.source_type.startswith("official_") else document.source_type,
    )
    readable_sender = sender.strip() if sender else ""
    header = f"[{label}]" + (f"[{readable_sender}]" if readable_sender else "")
    title = f"《{document.title}》" if document.title else ""
    return f"{header}{title}{text}"[:1200]


def _text_near_surface(text: str, surface: str, radius: int = 180) -> str:
    position = text.find(surface)
    if position < 0:
        return text[: radius * 2]
    start = max(0, position - radius)
    end = min(len(text), position + len(surface) + radius)
    return " ".join(text[start:end].split())


def _candidate_prompt_value(candidate: LocalTermCandidate) -> dict[str, object]:
    contexts = list(
        dict.fromkeys(
            value.context
            for value in sorted(
                candidate.occurrences,
                key=lambda occurrence: (
                    _source_context_priority(occurrence.source_type),
                    -occurrence.reliability,
                    occurrence.document_id,
                ),
            )
        )
    )[:8]
    return {
        "candidate": candidate.surface,
        "occurrence_count": sum(
            value.occurrence_count for value in candidate.occurrences
        ),
        "document_count": candidate.document_count,
        "source_counts": dict(
            sorted(
                _readable_source_counts(candidate.source_counts).items()
            )
        ),
        "contexts": contexts,
    }


def _readable_source_counts(source_counts: Counter[str]) -> Counter[str]:
    readable: Counter[str] = Counter()
    for source_type, count in source_counts.items():
        label = {
            "youtube_auto_subtitle": "ASR",
            "youtube_manual_subtitle": "YouTube字幕",
            "youtube_super_chat": "SC",
            "youtube_live_chat": "Chat",
            "youtube_metadata": "YouTube视频信息",
            "x_post": "X",
            "instagram_post": "Instagram",
        }.get(
            source_type,
            "官网" if source_type.startswith("official_") else source_type,
        )
        readable[label] += count
    return readable


def _source_context_priority(source_type: str) -> int:
    if source_type.startswith("official_"):
        return 0
    return {
        "youtube_metadata": 1,
        "x_post": 2,
        "instagram_post": 2,
        "youtube_manual_subtitle": 3,
        "youtube_auto_subtitle": 4,
        "youtube_super_chat": 5,
        "youtube_live_chat": 6,
    }.get(source_type, 3)


def _candidate_batches(
    candidates: list[LocalTermCandidate],
    *,
    context_size: int,
    max_tokens: int,
    target_input_tokens: int,
) -> list[tuple[LocalTermCandidate, ...]]:
    budget = max(1024, min(target_input_tokens, context_size - max_tokens))
    base = estimate_prompt_tokens(
        render_user_prompt("extract-fan-terms.md", CANDIDATES_JSON="[]")
    )
    batches: list[tuple[LocalTermCandidate, ...]] = []
    current: list[LocalTermCandidate] = []
    tokens = base
    for candidate in candidates:
        value_tokens = estimate_prompt_tokens(
            json.dumps(_candidate_prompt_value(candidate), ensure_ascii=False)
        )
        if current and tokens + value_tokens > budget:
            batches.append(tuple(current))
            current = []
            tokens = base
        current.append(candidate)
        tokens += value_tokens
    if current:
        batches.append(tuple(current))
    return batches


def _screen_candidate_batch(
    candidates: tuple[LocalTermCandidate, ...],
    *,
    search_web: object = None,
    **request_options: object,
) -> list[ExtractedTerm]:
    request_options.pop("context_size", None)
    request_options.pop("target_input_tokens", None)
    prompt = render_user_prompt(
        "extract-fan-terms.md",
        CANDIDATES_JSON=json.dumps(
            [_candidate_prompt_value(value) for value in candidates],
            ensure_ascii=False,
        ),
    )
    try:
        parsed = _request_json(
            "extract-fan-terms.md", prompt=prompt, **request_options
        )
    except TermExtractionOutputTooLong:
        if len(candidates) < 2:
            raise
        middle = len(candidates) // 2
        return [
            *_screen_candidate_batch(
                candidates[:middle], search_web=search_web, **request_options
            ),
            *_screen_candidate_batch(
                candidates[middle:], search_web=search_web, **request_options
            ),
        ]
    values = parsed.get("terms")
    if not isinstance(values, list):
        raise TypeError("term extraction response requires a terms array")
    by_surface = {value.surface: value for value in candidates}
    accepted: list[ExtractedTerm] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        surface = value.get("candidate")
        canonical_zh = value.get("canonical_zh")
        action = value.get("action")
        confidence = value.get("confidence")
        search_query = value.get("search_query", "")
        if (
            not isinstance(surface, str)
            or surface not in by_surface
            or not isinstance(canonical_zh, str)
            or not canonical_zh.strip()
            or action not in {"accept", "search"}
            or not isinstance(confidence, (int, float))
            or not 0 <= float(confidence) <= 1
            or not isinstance(search_query, str)
        ):
            continue
        candidate = by_surface.pop(surface)
        if action == "search":
            canonical_zh = _confirm_searched_candidate(
                candidate,
                canonical_zh.strip(),
                search_query.strip(),
                search_web=search_web,
                **request_options,
            )
            if canonical_zh is None:
                continue
        accepted.append(
            ExtractedTerm(
                surface=surface,
                canonical_zh=canonical_zh.strip(),
                aliases=(),
                reading="",
                relation="name",
                evidence_chunk_ids=tuple(
                    dict.fromkeys(value.chunk_id for value in candidate.occurrences)
                ),
                confidence=float(confidence),
            )
        )
    return accepted


def _confirm_searched_candidate(
    candidate: LocalTermCandidate,
    proposed_zh: str,
    query: str,
    *,
    search_web: object,
    **request_options: object,
) -> str | None:
    if not callable(search_web) or not query:
        return None
    results = _compact_web_results(search_web(query))
    audit_path = request_options.get("audit_path")
    _write_audit(
        audit_path if isinstance(audit_path, Path) else None,
        {
            "event": "term_web_search",
            "candidate": candidate.surface,
            "query": query,
            "results": results,
        },
    )
    prompt = render_user_prompt(
        "review-fan-terms.md",
        CANDIDATE_JSON=json.dumps(
            {
                **_candidate_prompt_value(candidate),
                "proposed_canonical_zh": proposed_zh,
                "web_results": results,
            },
            ensure_ascii=False,
        ),
    )
    parsed = _request_json("review-fan-terms.md", prompt=prompt, **request_options)
    if parsed.get("decision") != "accept":
        return None
    canonical_zh = parsed.get("canonical_zh")
    return canonical_zh.strip() if isinstance(canonical_zh, str) and canonical_zh.strip() else None


def _compact_web_results(response: dict[str, object]) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    raw_results = response.get("results", [])
    if not isinstance(raw_results, list):
        return results
    for index, value in enumerate(raw_results[:_WEB_RESULT_LIMIT]):
        if not isinstance(value, dict):
            continue
        url = str(value.get("href") or value.get("url") or "").strip()
        title = str(value.get("title") or "").strip()
        snippet = str(value.get("body") or value.get("snippet") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        results.append(
            {
                "result_id": f"web-{index}",
                "title": title[:200],
                "url": url[:1000],
                "snippet": snippet[:600],
            }
        )
    return results



def _request_json(
    prompt_name: str,
    *,
    prompt: str,
    request: Callable[[dict[str, object]], dict[str, object]],
    model: str,
    max_tokens: int,
    thinking: str | None,
    max_retries: int,
    audit_path: Path | None,
    batch_index: int,
) -> dict[str, object]:
    body = structured_request_body(
        model=model,
        prompt_name=prompt_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.1,
        json_mode=True,
        thinking=thinking,
    )
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        response: dict[str, object] | None = None
        content: object = None
        try:
            response = request(body)
            if finish_reason(response) == "length":
                raise TermExtractionOutputTooLong(
                    "term extraction output was truncated"
                )
            content = structured_response_content(response, finish_reason=finish_reason)
            return parse_json_object(content)
        except Exception as exc:
            last_error = exc
            _write_audit(
                audit_path,
                {
                    "event": "term_extraction_invalid_response",
                    "prompt": prompt_name,
                    "batch_index": batch_index,
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "content": content,
                },
            )
            if isinstance(exc, TermExtractionOutputTooLong) or (
                isinstance(exc, LLMHTTPError) and exc.status == 402
            ):
                raise
    raise RuntimeError(f"{prompt_name} exhausted LLM retries") from last_error


def _write_audit(path: Path | None, value: dict[str, object]) -> None:
    if path is None:
        return
    with _AUDIT_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
