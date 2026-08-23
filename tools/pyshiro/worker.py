from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pyshiro
import soundfile as sf
from pyshiro.labels import segments_to_phoneme_intervals

_FRAME_SECONDS = 0.005


def _resources() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent / "models"
    return (
        root / "pyshiro-jp-v2.hsmm",
        root / "pyshiro-jp-v2_phonemap.json",
    )


def _convert_readings(
    directory: Path, name: str, readings: list[str], table: object
) -> list[str]:
    path = directory / f"{name}.txt"
    path.write_text("\n".join(readings) + "\n", encoding="utf-8")
    return list(pyshiro.convert_lyric_file(path, table))


def _owned_units(
    directory: Path,
    table: object,
    line_intervals: list[list[tuple[int, int, str]]],
    display_units: object,
) -> list[list[dict[str, object]]]:
    if not isinstance(display_units, list) or len(display_units) != len(line_intervals):
        raise ValueError("display_units must match readings")
    output: list[list[dict[str, object]]] = []
    for line_index, (intervals, raw_units) in enumerate(
        zip(line_intervals, display_units)
    ):
        if not isinstance(raw_units, list) or not raw_units:
            raise ValueError(f"display_units[{line_index}] must be non-empty")
        expected: list[tuple[str, list[str]]] = []
        for unit_index, raw_unit in enumerate(raw_units):
            if not isinstance(raw_unit, dict):
                raise TypeError("display unit must be an object")
            text = str(raw_unit.get("text", ""))
            reading = str(raw_unit.get("reading", ""))
            if not text or not reading:
                raise ValueError("display unit text and reading must be non-empty")
            phonemes = [
                value
                for value in _convert_readings(
                    directory,
                    f"unit-{line_index:04d}-{unit_index:04d}",
                    [reading],
                    table,
                )
                if value != "pau"
            ]
            if not phonemes:
                raise RuntimeError(f"display unit {text!r} produced no phonemes")
            expected.append((text, phonemes))
        actual = [phoneme for _start, _end, phoneme in intervals]
        flattened = [phoneme for _text, values in expected for phoneme in values]
        if actual != flattened:
            raise RuntimeError(
                f"display-unit phonemes differ from aligned line {line_index}: "
                f"expected={flattened!r} actual={actual!r}"
            )
        cursor = 0
        line_output: list[dict[str, object]] = []
        for text, phonemes in expected:
            owned = intervals[cursor : cursor + len(phonemes)]
            cursor += len(phonemes)
            line_output.append(
                {
                    "text": text,
                    "phonemes": phonemes,
                    "start": owned[0][0] * _FRAME_SECONDS,
                    "end": owned[-1][1] * _FRAME_SECONDS,
                }
            )
        output.append(line_output)
    return output


def align(request: dict[str, object]) -> dict[str, object]:
    wav = Path(str(request["wav"]))
    lines = request.get("readings")
    if (
        not isinstance(lines, list)
        or not lines
        or not all(isinstance(v, str) and v for v in lines)
    ):
        raise ValueError("readings must be non-empty strings")
    duration = sf.info(wav).duration
    if duration > 20.001:
        raise ValueError(f"pySHIRO input exceeds 20 seconds: {duration:.3f}s")
    model_path, phonemap_path = _resources()
    if not model_path.is_file() or not phonemap_path.is_file():
        raise RuntimeError("pySHIRO Japanese v2 model resources are unavailable")
    model = pyshiro.load_hsmm(model_path)
    phonemap = pyshiro.load_phonemap(phonemap_path)
    table = pyshiro.load_table()
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        streams = pyshiro.extract_mfcc_from_file(wav)
        phonemes = _convert_readings(temporary, "lyrics", lines, table)
    states = pyshiro.build_state_sequence(phonemes, phonemap, streams[0].shape[0])
    segments, likelihood = pyshiro.forced_align_2pass(
        model, streams, states, nodur_phonemes={"pau", "br"}
    )
    intervals = segments_to_phoneme_intervals(phonemes, segments)
    line_ranges: list[list[float]] = []
    line_intervals: list[list[tuple[int, int, str]]] = []
    active: list[tuple[int, int]] = []
    active_intervals: list[tuple[int, int, str]] = []
    items: list[dict[str, object]] = []
    for start, end, phoneme in intervals:
        items.append(
            {
                "text": phoneme,
                "start": start * _FRAME_SECONDS,
                "end": end * _FRAME_SECONDS,
            }
        )
        if phoneme == "pau":
            if active:
                line_ranges.append(
                    [active[0][0] * _FRAME_SECONDS, active[-1][1] * _FRAME_SECONDS]
                )
                line_intervals.append(active_intervals)
                active = []
                active_intervals = []
        else:
            active.append((start, end))
            active_intervals.append((start, end, phoneme))
    if active:
        line_ranges.append(
            [active[0][0] * _FRAME_SECONDS, active[-1][1] * _FRAME_SECONDS]
        )
        line_intervals.append(active_intervals)
    if len(line_ranges) != len(lines):
        raise RuntimeError(
            "pySHIRO returned "
            f"{len(line_ranges)} line ranges for {len(lines)} lyric lines"
        )
    response = {
        "duration": duration,
        "likelihood_per_frame": likelihood / streams[0].shape[0],
        "lines": line_ranges,
        "phonemes": items,
    }
    if request.get("display_units") is not None:
        with tempfile.TemporaryDirectory() as directory:
            response["units"] = _owned_units(
                Path(directory), table, line_intervals, request["display_units"]
            )
    return response


def main() -> None:
    try:
        result = align(json.load(__import__("sys").stdin))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        raise SystemExit(1)
    print(json.dumps({"ok": True, **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
