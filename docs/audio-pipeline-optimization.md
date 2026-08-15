# Audio Pipeline Memory and Concurrency Plan

## Status

Implemented on 2026-08-14. The pipeline now removes avoidable temporary audio files,
retains reusable 16 kHz audio in shared CPU memory, and overlaps only explicitly
configured independent work. Same-model batching and wider GPU concurrency remain
benchmark-driven follow-ups rather than assumptions.

## Current Process Boundaries

The main Python process currently owns or orchestrates:

- the 16 kHz mono analysis waveform;
- the MOSS/pyannote diarization result and cache lifecycle;
- AST singing detection;
- Demucs vocal separation;
- Qwen ASR and the forced aligner;
- speaker-region aggregation and identity assignment.

The following components use separate address spaces:

- MOSS-Transcribe-Diarize: one isolated uv worker processes all long windows while
  keeping the model loaded;
- MossFormer2: used only by the explicit pyannote fallback path;
- ERes2NetV2: one batched uv worker invocation handles all enrollment snippets;
- PaddleOCR: one persistent uv worker process per song-identification run;
- song search: a short-lived uv worker process per tool call;
- ffmpeg, yt-dlp, the CUDA subtitle renderer, and biliup: native subprocesses.

## Two Audio Representations

Maintain two intentional representations instead of sending one degraded waveform to
every model:

```text
original 44.1/48 kHz stereo audio
|- Demucs and singing/music processing
`- resample once to 16 kHz mono
   |- MOSS diarization, or VAD and pyannote in fallback mode
   |- AST speech/singing evidence
   |- speaker embeddings
   `- Qwen ASR and forced alignment
```

The complete 16 kHz mono waveform remains in shared ordinary CPU memory for the job.
One hour of float32 mono audio is about 230 MB. Use pinned memory only as a bounded,
reusable staging area for batches that are about to move to a GPU.

The complete high-quality stereo track is not retained. Only detected song candidate
ranges are decoded at the Demucs sample rate as stereo float32 audio.

## Shared-Memory Transport

Decode 16 kHz mono PCM once and expose it through CPU shared memory. The owner passes
descriptors instead of WAV paths to workers:

```json
{
  "shared_memory": "subtitle-audio-job-id",
  "dtype": "float32",
  "shape": [1, 57600000],
  "sample_rate": 16000,
  "items": [
    {"id": 7, "start_sample": 120000, "end_sample": 240000}
  ]
}
```

Workers attach with `multiprocessing.shared_memory`, construct NumPy views, and only
copy the current inference batch to their own CUDA context. Do not use CUDA IPC across
PyTorch, Paddle, and ClearVoice environments.

The main process owns the segment, calls `close()` and `unlink()` after all consumers
finish, and records active shared-memory names in the job directory so abandoned
segments can be cleaned after a crash.

## Worker Changes

1. MOSS, MossFormer2, and ERes2NetV2 workers batch all same-model job items into one
   invocation, so each model loads once per job.
2. Pass MossFormer2 and ERes2NetV2 input ranges through shared-memory descriptors.
3. Return ERes2NetV2 embeddings as JSON because they are small.
4. Write MossFormer2 outputs into preallocated shared memory and return descriptors.
5. Pass Qwen `(numpy_array, 16000)` inputs directly and remove per-region ffmpeg and
   temporary WAV creation.
6. Keep PaddleOCR persistent as it is. Image shared memory is lower priority.
7. Add true model batching only after measuring supported input lengths, padding cost,
   memory use, and output quality. A persistent worker alone is not proof of batching.

## Debug Artifacts

MossFormer2 output remains in shared memory and is not a durable cache. Write separated
audio to the job directory only when debug mode is enabled. Normal runs must not create
MossFormer2 WAV files. A resumed normal job recomputes any required separation.

## Concurrency Policy

Prefer pipeline parallelism and batching over unrestricted GPU-model concurrency:

- run independent CPU decoding, slicing, resampling, validation, and cache writes while
  a GPU model is working;
- MOSS/pyannote and raw-audio AST are logically independent, but enable concurrent GPU
  execution only if a benchmark shows lower wall time within the memory budget;
- Demucs depends on song candidates; fallback MossFormer2 depends on overlap detection;
- speaker identity depends on diarization and exclusion of confirmed singing regions;
- Qwen ASR starts after routing regions to speech, singing, and ambiguous paths;
- Demucs, MossFormer2, and Qwen are GPU-heavy and should be mutually exclusive by
  default on the current 22 GiB GPU;
- batch multiple ranges for the same loaded model before adding cross-model concurrency.

Use one GPU scheduler with an explicit memory budget rather than independent threads
launching models directly. Record per-stage load time, inference time, transfer time,
peak VRAM, and fallback count before changing concurrency defaults.

## Implemented

1. `AudioBufferPool` decodes 16 kHz float32 PCM once from ffmpeg stdout into shared
   CPU memory and records owned segment names for crash cleanup.
2. Change Qwen to accept NumPy slices and remove `asr-analysis-chunks` WAV traffic.
3. Add shared-memory ownership and crash cleanup.
4. Convert ERes2NetV2 to a persistent shared-memory worker.
5. Convert MossFormer2 input and output to shared memory, with disk output enabled only
   in debug mode.
6. Decode source-quality stereo only for song candidate ranges before Demucs.
7. Stage timing and peak allocated VRAM are logged. `initial_analysis_concurrency=2`
   can overlap MOSS/pyannote and raw AST after a machine-specific memory check; the example
   remains at `1`. Demucs, MossFormer2, ERes2NetV2 and Qwen remain stage-serialized.
8. MOSS runs in target 80-minute windows selected near low-energy boundaries, never
   exceeding 90 minutes. Its raw speaker-aware transcript is cached for audit.
9. Identity matching aggregates all clean evidence for one window-scoped MOSS label,
   trims the farthest 15% around its embedding medoid, then applies capped duration
   weights against the existing ERes2NetV2 multi-center profiles. Overlap inherits the
   result but does not vote; different MOSS labels may map to the same member.

## Acceptance Checks

- A normal job creates no per-cue or per-region ASR WAV files.
- Unit coverage verifies Qwen receives in-memory `(numpy_array, 16000)` inputs while
  preserving the existing timestamp restoration path. A real-video equivalence and
  wall-time benchmark remains required before changing production defaults broadly.
- Worker restart and a killed main process do not leave unbounded shared memory.
- MossFormer2 and ERes2NetV2 load once per job rather than once per request.
- Parallel initial analysis is opt-in and should remain enabled only where measured
  wall time improves without OOM or quality regression.
- Demucs receives source-quality stereo song ranges rather than the 16 kHz mono
  analysis waveform.
