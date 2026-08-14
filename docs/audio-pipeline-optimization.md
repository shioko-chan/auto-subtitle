# Audio Pipeline Memory and Concurrency Plan

## Status

This document records an accepted follow-up optimization. It is not implemented yet.
The objective is to remove avoidable temporary audio files, retain reusable audio in
memory, and overlap independent work without running competing GPU models blindly.

## Current Process Boundaries

The main Python process currently owns:

- the 16 kHz mono analysis waveform;
- pyannote diarization;
- AST singing detection;
- Demucs vocal separation;
- Qwen ASR and the forced aligner;
- speaker-region aggregation and identity assignment.

The following components use separate address spaces:

- MossFormer2: a new uv worker process is started for each request;
- ERes2NetV2: a new uv worker process is started for each request;
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
   |- VAD and pyannote
   |- AST speech/singing evidence
   |- speaker embeddings
   `- Qwen ASR and forced alignment
```

The complete 16 kHz mono waveform should remain in ordinary CPU memory for the job.
One hour of float32 mono audio is about 230 MB. Use pinned memory only as a bounded,
reusable staging area for batches that are about to move to a GPU.

Do not keep the complete high-quality stereo track resident by default. Decode only
the detected song candidate ranges at source quality for Demucs. A bounded cache may
retain nearby or repeatedly used ranges during the current job.

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

1. Keep MossFormer2 and ERes2NetV2 workers alive across requests so model weights are
   loaded once per job.
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
- pyannote and raw-audio AST are logically independent, but enable concurrent GPU
  execution only if a benchmark shows lower wall time within the memory budget;
- Demucs depends on song candidates; MossFormer2 depends on overlap detection;
- speaker identity depends on diarization and exclusion of confirmed singing regions;
- Qwen ASR starts after routing regions to speech, singing, and ambiguous paths;
- Demucs, MossFormer2, and Qwen are GPU-heavy and should be mutually exclusive by
  default on the current 22 GiB GPU;
- batch multiple ranges for the same loaded model before adding cross-model concurrency.

Use one GPU scheduler with an explicit memory budget rather than independent threads
launching models directly. Record per-stage load time, inference time, transfer time,
peak VRAM, and fallback count before changing concurrency defaults.

## Implementation Order

1. Introduce an `AudioBuffer` abstraction and decode 16 kHz PCM once from ffmpeg
   stdout into ordinary CPU memory.
2. Change Qwen to accept NumPy slices and remove `asr-analysis-chunks` WAV traffic.
3. Add shared-memory ownership and crash cleanup.
4. Convert ERes2NetV2 to a persistent shared-memory worker.
5. Convert MossFormer2 input and output to shared memory, with disk output enabled only
   in debug mode.
6. Decode source-quality stereo only for song candidate ranges before Demucs.
7. Add stage timing and VRAM telemetry, then benchmark safe concurrency and true
   same-model batching.

## Acceptance Checks

- A normal job creates no per-cue or per-region ASR WAV files.
- Qwen output and timestamps remain equivalent on a fixed benchmark.
- Worker restart and a killed main process do not leave unbounded shared memory.
- MossFormer2 and ERes2NetV2 load once per job rather than once per request.
- Parallel execution is enabled only where measured wall time improves without OOM or
  quality regression.
- Demucs receives source-quality stereo song ranges rather than the 16 kHz mono
  analysis waveform.
