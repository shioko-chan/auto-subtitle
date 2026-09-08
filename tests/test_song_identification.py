import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from subtitle_pipeline.config import SongIdentificationConfig
from subtitle_pipeline.lyrics_library import LibrarySong, LyricLine, LyricsLibrary
from subtitle_pipeline.lyrics_matching import JapaneseNormalizer, LyricAnchor, SongMatch
from subtitle_pipeline.song_identification import (
    OCRCandidate,
    SongIdentificationResult,
    SongSearchGroup,
    VerifiedLyricSpan,
    _align_match_with_pyshiro,
    _anchor_has_continuous_support,
    _apply_local_match,
    _build_lyric_search_queries,
    _competing_line_sets,
    _load_cache,
    _lyric_unit_timeline_errors,
    _materialize_verification_stems,
    _merged_cue_duration,
    _parse_worker_json_output,
    _public_http_url,
    _pyshiro_likelihood_wins,
    _rank_lyric_search_results,
    _recover_lyric_ranges,
    _refine_match_with_speech_support,
    _routed_speech_cue_ids,
    _run_pyshiro_lines,
    _search_canonical_lyrics,
    _signature,
    _supported_lyrics_url,
    _title_uses_music_mode,
    _web_search_policy,
    aggregate_ocr_observations,
    apply_lyric_corrections,
    arbitrate_verified_lyrics,
    collect_ocr_candidates,
    group_song_search_groups,
    identify_and_align_songs,
    select_ocr_song_titles,
    translate_aligned_song_lyrics,
)
from subtitle_pipeline.subtitles import Cue, TimedTextUnit


class SongIdentificationTests(unittest.TestCase):
    def test_song_alignment_support_uses_wider_route_without_widening_speech(self):
        group = group_song_search_groups([Cue(100, 110, "song", kind="singing")], 35)[0]
        cues = [
            Cue(
                15,
                20,
                "support",
                kind="speech",
                speaker_assignment="song_alignment_support:0",
            ),
            Cue(15, 20, "ordinary", kind="speech"),
            Cue(100, 110, "song", kind="singing"),
        ]

        result = _routed_speech_cue_ids(
            group,
            cues,
            SongIdentificationConfig(song_search_group_gap_seconds=35),
        )

        self.assertEqual(result, [0])

    @patch("subtitle_pipeline.audio_analysis.separate_vocal_ranges")
    def test_verification_materializes_missing_vocal_stem(self, separate):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "vocal-candidates" / "manifest.json"
            probes: list[dict[str, object]] = [{"start": 10, "end": 15}]
            output_dir = root / "alignment"
            output_dir.mkdir()

            _materialize_verification_stems(
                root / "video.mp4",
                probes,
                output_dir,
                "cuda",
                manifest_path,
            )

            self.assertEqual(separate.call_args.args[1][0][:2], (10, 15))
            self.assertEqual(probes[0]["wav"], output_dir / "range-0000.vocals.wav")

    @patch("subtitle_pipeline.song_identification._match_candidates")
    def test_song_match_refinement_excludes_unmatched_support_group(self, matcher):
        song = LibrarySong(
            "song",
            "title",
            "artist",
            (),
            "https://example.com/song",
            "hash",
            tuple(LyricLine(index, f"line {index}") for index in range(4)),
        )
        original = SongMatch(song, (LyricAnchor(0, 2, 3, 0.8),), 0.8)
        support_match = SongMatch(song, (LyricAnchor(0, 0, 1, 0.7),), 0.7)
        refined = SongMatch(
            song,
            (
                LyricAnchor(0, 0, 1, 0.7),
                LyricAnchor(1, 2, 3, 0.8),
            ),
            0.75,
        )
        matcher.side_effect = [support_match, None, refined]
        cues = [
            Cue(
                0,
                10,
                "support",
                kind="speech",
                speaker_assignment="song_alignment_support:0",
            ),
            Cue(
                60,
                70,
                "unrelated credits",
                kind="speech",
                speaker_assignment="song_alignment_support:1",
            ),
            Cue(20, 30, "alt anchor", kind="singing"),
        ]

        alignment_ids, result = _refine_match_with_speech_support(
            cues,
            [2],
            [0, 1],
            original,
            SongIdentificationConfig(song_search_group_gap_seconds=35),
        )

        self.assertEqual(alignment_ids, [0, 2, 1])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual([anchor.cue_index for anchor in result.anchors], [0, 1])
        self.assertEqual(matcher.call_args.args[0], ["support", "alt anchor"])

    @patch("subtitle_pipeline.song_identification._public_http_url", return_value=True)
    def test_supported_lyrics_url_requires_song_detail_page(self, _public_url):
        self.assertTrue(_supported_lyrics_url("https://utaten.com/lyric/abc123/"))
        self.assertTrue(_supported_lyrics_url("https://www.uta-net.com/movie/7019/"))
        self.assertTrue(
            _supported_lyrics_url("https://www.oricon.co.jp/prof/123/lyrics/456/")
        )
        self.assertTrue(_supported_lyrics_url("https://s.awa.fm/track/abcdef123456"))
        self.assertFalse(_supported_lyrics_url("https://www.uta-net.com/"))
        self.assertFalse(_supported_lyrics_url("https://www.uta-net.com/artist/1399/"))
        self.assertFalse(_supported_lyrics_url("https://utaten.com/search"))

    def test_lyric_unit_timeline_errors_allow_long_positive_units(self):
        errors = _lyric_unit_timeline_errors(
            (
                TimedTextUnit("長", 1.0, 4.25),
                TimedTextUnit("逆", 5.0, 4.5),
            ),
            line_id=7,
            lyric_text="長い歌声",
        )

        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["issue"], "non_positive_unit_duration")
        self.assertEqual(errors[0]["unit_text"], "逆")
        self.assertEqual(errors[0]["lyric_text"], "長い歌声")

    def test_pyshiro_json_parser_ignores_worker_stdout_noise(self):
        self.assertEqual(
            _parse_worker_json_output('loading model...\n{"ok":true,"lines":[]}\n'),
            {"ok": True, "lines": []},
        )

    @patch("subtitle_pipeline.song_identification.subprocess.run")
    def test_pyshiro_worker_failure_audit_preserves_structured_error(self, run):
        run.return_value.returncode = 1
        run.return_value.stdout = json.dumps(
            {
                "ok": False,
                "error_type": "RuntimeError",
                "error": "display-unit phonemes differ from aligned line 0",
                "traceback": "Traceback... worker.py line 82",
            }
        )
        run.return_value.stderr = "environment warning"
        audit: dict[str, object] = {}

        response = _run_pyshiro_lines(
            "uv",
            Path("worker.py"),
            Path("audio.wav"),
            [LyricLine(0, "歌詞")],
            JapaneseNormalizer(),
            failure_audit=audit,
        )

        self.assertIsNone(response)
        self.assertEqual(audit["reason"], "pyshiro_worker_failed")
        self.assertEqual(audit["failure_stage"], "worker_execution")
        self.assertEqual(audit["worker_returncode"], 1)
        self.assertEqual(audit["worker_error_type"], "RuntimeError")
        self.assertIn("display-unit phonemes", str(audit["worker_error"]))
        self.assertEqual(audit["stderr_tail"], "environment warning")
        request = audit["request"]
        self.assertIsInstance(request, dict)
        self.assertEqual(request["lyrics"][0]["text"], "歌詞")

    @patch("subtitle_pipeline.song_identification.subprocess.run")
    def test_pyshiro_worker_failure_audit_records_invalid_output(self, run):
        run.return_value.returncode = 1
        run.return_value.stdout = "model startup noise"
        run.return_value.stderr = "worker crashed"
        audit: dict[str, object] = {}

        response = _run_pyshiro_lines(
            "uv",
            Path("worker.py"),
            Path("audio.wav"),
            [LyricLine(0, "歌詞")],
            JapaneseNormalizer(),
            failure_audit=audit,
        )

        self.assertIsNone(response)
        self.assertEqual(audit["reason"], "pyshiro_worker_output_invalid")
        self.assertEqual(audit["worker_returncode"], 1)
        self.assertEqual(audit["stdout_tail"], "model startup noise")
        self.assertEqual(audit["stderr_tail"], "worker crashed")

    def test_song_translation_stage_backfills_without_invalidating_signature(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"video")
            library_path = root / "lyrics.sqlite3"
            library = LyricsLibrary(library_path)
            try:
                song = library.store_canonical_song(
                    title="曲名",
                    artist="歌手",
                    aliases=[],
                    source_url="https://example.com/lyrics",
                    lines=[("一行目", None), ("二行目", None), ("三行目", None)],
                )
            finally:
                library.close()
            config = SongIdentificationConfig(
                enabled=True, lyrics_library_path=str(library_path)
            )
            signature_before = _signature(video, [], {}, config)
            result = SongIdentificationResult(
                [
                    Cue(1, 2, "一行目", "singer", "singing"),
                    Cue(2, 3, "二行目", "singer", "singing"),
                ],
                [
                    {
                        "song_id": song.song_id,
                        "search_group": {"start": 1, "end": 3},
                        "alignments": [
                            {"corrected_text": "一行目", "lyric_line_ids": [0]},
                            {"corrected_text": "二行目", "lyric_line_ids": [1]},
                        ],
                    }
                ],
            )

            translated = translate_aligned_song_lyrics(
                result,
                config,
                lambda *_args, **_kwargs: (
                    {0: "第一行", 1: "第二行", 2: "第三行"},
                    "llm",
                ),
                lyrics_translation_model="test-model",
            )
            signature_after = _signature(video, [], {}, config)

        self.assertEqual(
            [cue.preferred_translation for cue in translated.corrected_cues],
            ["第一行", "第二行"],
        )
        self.assertEqual(signature_before, signature_after)

    def test_existing_complete_llm_lyrics_are_used_without_another_llm_call(self):
        with TemporaryDirectory() as directory:
            library_path = Path(directory) / "lyrics.sqlite3"
            library = LyricsLibrary(library_path)
            try:
                song = library.store_canonical_song(
                    title="曲名",
                    artist="歌手",
                    aliases=[],
                    source_url="https://example.com/lyrics",
                    lines=[
                        ("前の行", None),
                        ("メタモルフォーゼ", None),
                        ("次の行", None),
                    ],
                )
                library.store_translations(
                    song.song_id,
                    {0: "前一行", 1: "变形虫", 2: "下一行"},
                    source="llm",
                )
            finally:
                library.close()
            result = SongIdentificationResult(
                [Cue(1, 2, "メタモルフォーゼ", "singer", "singing")],
                [
                    {
                        "song_id": song.song_id,
                        "search_group": {"start": 1, "end": 2},
                        "alignments": [
                            {
                                "corrected_text": "メタモルフォーゼ",
                                "lyric_line_ids": [1],
                            }
                        ],
                    }
                ],
            )

            translated = translate_aligned_song_lyrics(
                result,
                SongIdentificationConfig(
                    enabled=True, lyrics_library_path=str(library_path)
                ),
                lambda *_args, **_kwargs: self.fail(
                    "complete existing lyrics must not be translated again"
                ),
                lyrics_translation_model="test-model",
            )
            library = LyricsLibrary(library_path)
            try:
                stored_song = library.get(song.song_id)
            finally:
                library.close()

        self.assertEqual(translated.corrected_cues[0].preferred_translation, "变形虫")
        assert stored_song is not None
        self.assertEqual(stored_song.lines[1].translation_source, "llm")

    def test_complete_external_and_official_lyrics_are_used_as_stored(self):
        with TemporaryDirectory() as directory:
            library_path = Path(directory) / "lyrics.sqlite3"
            library = LyricsLibrary(library_path)
            try:
                song = library.store_canonical_song(
                    title="曲名",
                    artist="歌手",
                    aliases=[],
                    source_url="https://example.com/lyrics",
                    lines=[("一行目", None), ("二行目", None), ("三行目", None)],
                )
                library.store_translations(
                    song.song_id,
                    {0: "外部第一行", 1: "外部第二行", 2: "外部第三行"},
                    source="external",
                )
                library.store_translations(
                    song.song_id,
                    {1: "官方第二行"},
                    source="official",
                )
            finally:
                library.close()
            result = SongIdentificationResult(
                [Cue(1, 2, "一行目", "singer", "singing")],
                [
                    {
                        "song_id": song.song_id,
                        "search_group": {"start": 1, "end": 2},
                        "alignments": [
                            {
                                "corrected_text": "一行目",
                                "lyric_line_ids": [0],
                            }
                        ],
                    }
                ],
            )

            translated = translate_aligned_song_lyrics(
                result,
                SongIdentificationConfig(
                    enabled=True, lyrics_library_path=str(library_path)
                ),
                lambda *_args, **_kwargs: self.fail(
                    "complete external lyrics must not be translated again"
                ),
                lyrics_translation_model="test-model",
            )
            library = LyricsLibrary(library_path)
            try:
                stored_song = library.get(song.song_id)
            finally:
                library.close()

        self.assertEqual(
            translated.corrected_cues[0].preferred_translation,
            "外部第一行",
        )
        assert stored_song is not None
        self.assertEqual(stored_song.lines[0].translation_source, "external")
        self.assertEqual(stored_song.lines[1].translation, "官方第二行")
        self.assertEqual(stored_song.lines[1].translation_source, "official")

    def test_verified_lyrics_skip_translation(self):
        with TemporaryDirectory() as directory:
            library_path = Path(directory) / "lyrics.sqlite3"
            library = LyricsLibrary(library_path)
            try:
                song = library.store_canonical_song(
                    title="曲名",
                    artist="歌手",
                    aliases=[],
                    source_url="https://example.com/lyrics",
                    lines=[("一行目", None), ("二行目", None), ("三行目", None)],
                )
                library.store_translations(
                    song.song_id,
                    {0: "人工第一行", 1: "人工第二行", 2: "人工第三行"},
                    source="verified",
                )
            finally:
                library.close()
            result = SongIdentificationResult(
                [Cue(1, 2, "一行目", "singer", "singing")],
                [
                    {
                        "song_id": song.song_id,
                        "search_group": {"start": 1, "end": 2},
                        "alignments": [
                            {
                                "corrected_text": "一行目",
                                "lyric_line_ids": [0],
                            }
                        ],
                    }
                ],
            )

            translated = translate_aligned_song_lyrics(
                result,
                SongIdentificationConfig(
                    enabled=True, lyrics_library_path=str(library_path)
                ),
                lambda *_args, **_kwargs: self.fail(
                    "verified lyrics must not be translated"
                ),
                lyrics_translation_model="test-model",
            )

        self.assertEqual(
            translated.corrected_cues[0].preferred_translation,
            "人工第一行",
        )

    def test_song_cache_accepts_empty_corrected_cues(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "song-cache.json"
            path.write_text(
                '{"version":24,"signature":"test","reports":[],"corrected_cues":[]}',
                encoding="utf-8",
            )

            result = _load_cache(path, "test", [])

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.corrected_cues, [])

    @patch(
        "subtitle_pipeline.song_identification._search_canonical_lyrics",
        return_value=([], {"queries": [], "fetches": []}),
    )
    @patch(
        "subtitle_pipeline.song_identification._load_ocr_cache",
        return_value=[[]],
    )
    def test_unmatched_song_search_group_preserves_internal_speech(
        self, _ocr_cache, _search
    ):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"video")
            cues = [
                Cue(0, 2, "unmatched singing", "singer", "singing"),
                Cue(2, 3, "in-song speech", "singer", "speech"),
                Cue(3, 5, "more unmatched singing", "singer", "singing"),
                Cue(50, 51, "outside speech", "singer", "speech"),
            ]

            result = identify_and_align_songs(
                video,
                cues,
                {},
                root,
                SongIdentificationConfig(
                    enabled=True, lyrics_library_path=str(root / "lyrics.sqlite3")
                ),
            )

        self.assertEqual(
            [cue.text for cue in result.corrected_cues],
            ["in-song speech", "outside speech"],
        )
        self.assertEqual(
            result.reports[0]["evidence"],
            ["no_acoustically_verified_lyric_match"],
        )

    @patch("subtitle_pipeline.song_identification._validate_song_match")
    @patch("subtitle_pipeline.song_identification._rank_candidate_matches")
    @patch(
        "subtitle_pipeline.song_identification._load_ocr_cache",
        return_value=[[]],
    )
    def test_one_search_group_can_verify_two_songs(
        self, _ocr_cache, rank_matches, validate_match
    ):
        first_song = LibrarySong(
            "first",
            "First Song",
            "Singer",
            (),
            "https://example.com/first",
            "first-hash",
            (LyricLine(0, "第一曲"),),
        )
        second_song = LibrarySong(
            "second",
            "Second Song",
            "Singer",
            (),
            "https://example.com/second",
            "second-hash",
            (LyricLine(0, "第二曲"),),
        )
        first_match = SongMatch(
            first_song, (LyricAnchor(0, 0, 1, 0.9),), 0.9
        )
        second_match = SongMatch(
            second_song, (LyricAnchor(0, 0, 1, 0.9),), 0.9
        )
        rank_matches.side_effect = [[first_match], [second_match]]

        def validate(
            _job_dir,
            _video,
            cues,
            singing_ids,
            _speech_ids,
            match,
            _config,
            _source_maximum_units,
        ):
            cue_id = singing_ids[0]
            lyric = Cue(
                cues[cue_id].start,
                cues[cue_id].end,
                match.song.lines[0].text,
                kind="singing",
            )
            return (
                match,
                singing_ids,
                {cue_id: [lyric]},
                [
                    {
                        "asr_cue_ids": [cue_id],
                        "lyric_line_ids": [0],
                        "start": lyric.start,
                        "end": lyric.end,
                    }
                ],
                [
                    {
                        "status": "aligned",
                        "cue_id": cue_id,
                        "likelihood_per_frame": -10.0,
                    }
                ],
            )

        validate_match.side_effect = validate
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"video")
            result = identify_and_align_songs(
                video,
                [
                    Cue(0, 12, "第一曲", kind="singing"),
                    Cue(20, 32, "第二曲", kind="singing"),
                ],
                {"title": "歌枠"},
                root,
                SongIdentificationConfig(
                    enabled=True,
                    lyrics_library_path=str(root / "lyrics.sqlite3"),
                ),
            )

        self.assertEqual([report["song"] for report in result.reports], [
            "First Song",
            "Second Song",
        ])
        self.assertEqual(
            [cue.text for cue in result.corrected_cues], ["第一曲", "第二曲"]
        )
        self.assertEqual(validate_match.call_count, 2)

    def test_song_cache_signature_ignores_runtime_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"video")
            config = SongIdentificationConfig(
                lyrics_library_path=str(root / "lyrics.sqlite3")
            )
            cues = [Cue(0, 1, "歌詞", kind="singing")]
            first = _signature(
                video,
                cues,
                {
                    "id": "video-id",
                    "title": "歌枠",
                    "channel_id": "channel-id",
                    "epoch": 100,
                    "requested_downloads": [{"temporary": "value"}],
                },
                config,
            )
            second = _signature(
                video,
                cues,
                {
                    "id": "video-id",
                    "title": "歌枠",
                    "channel_id": "channel-id",
                    "epoch": 200,
                    "requested_downloads": [{"temporary": "changed"}],
                },
                config,
            )
            changed_title = _signature(
                video,
                cues,
                {"id": "video-id", "title": "別の歌枠"},
                config,
            )

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed_title)

    @patch("subtitle_pipeline.song_identification._run_pyshiro_lines")
    @patch(
        "subtitle_pipeline.song_identification._vocal_active_ratio", return_value=0.5
    )
    @patch("subtitle_pipeline.song_identification._materialize_verification_stems")
    @patch("subtitle_pipeline.song_identification.shutil.which", return_value="uv")
    def test_recovers_internal_lyric_gap_from_actual_range(
        self, _which, materialize, _activity, run_pyshiro
    ):
        song = LibrarySong(
            "song",
            "title",
            "artist",
            (),
            "https://example.com",
            "hash",
            (
                LyricLine(0, "最初の歌詞", translation="第一行"),
                LyricLine(1, "漏れた歌詞", translation="漏掉的一行"),
                LyricLine(2, "最後の歌詞", translation="最后一行"),
            ),
        )
        match = SongMatch(
            song,
            (LyricAnchor(0, 0, 1, 0.9), LyricAnchor(1, 2, 3, 0.9)),
            0.9,
        )
        run_pyshiro.side_effect = [
            {
                "ok": True,
                "likelihood_per_frame": -17.0,
                "lines": [[0.2, 3.8]],
                "units": [[{"text": "漏れた歌詞", "start": 0.2, "end": 3.8}]],
            },
            {"ok": True, "likelihood_per_frame": -22.0},
            {"ok": True, "likelihood_per_frame": -21.0},
        ]
        with TemporaryDirectory() as directory:
            job_dir = Path(directory)
            def create_stems(_video, probes, output_dir, _device, _manifest):
                for index, probe in enumerate(probes):
                    wav = output_dir / f"{index}.wav"
                    wav.write_bytes(b"wav")
                    probe["wav"] = wav

            materialize.side_effect = create_stems
            replacements, alignments, audit = _recover_lyric_ranges(
                job_dir,
                [
                    Cue(0, 8, "最初の歌詞", "singer", "singing"),
                    Cue(12, 16, "最後の歌詞", "singer", "singing"),
                ],
                [0, 1],
                match,
                SongIdentificationConfig(pyshiro_max_window_seconds=15),
                video=job_dir / "source.mp4",
            )

        self.assertEqual([cue.text for cue in replacements[0]], ["漏れた歌詞"])
        self.assertAlmostEqual(replacements[0][0].start, 8.2)
        self.assertEqual(alignments[0]["lyric_line_ids"], [1])
        self.assertEqual(audit[0]["status"], "range_verified")
        self.assertEqual(audit[0]["route"], "between_anchors")

    @patch("subtitle_pipeline.song_identification._materialize_verification_stems")
    @patch("subtitle_pipeline.song_identification.shutil.which", return_value="uv")
    def test_long_lyric_gap_is_split_at_pyshiro_limit(
        self, _which, materialize
    ):
        song = LibrarySong(
            "song",
            "title",
            "artist",
            (),
            "https://example.com",
            "hash",
            tuple(LyricLine(index, f"歌詞{index}") for index in range(8)),
        )
        match = SongMatch(
            song,
            (LyricAnchor(0, 0, 1, 0.9), LyricAnchor(1, 7, 8, 0.9)),
            0.9,
        )
        with TemporaryDirectory() as directory:
            job_dir = Path(directory)
            _recover_lyric_ranges(
                job_dir,
                [
                    Cue(0, 5, "歌詞0", "singer", "singing"),
                    Cue(65, 70, "歌詞7", "singer", "singing"),
                ],
                [0, 1],
                match,
                SongIdentificationConfig(pyshiro_max_window_seconds=15),
                video=job_dir / "source.mp4",
            )
        probes = materialize.call_args.args[1]
        self.assertEqual(len(probes), 4)
        self.assertTrue(all(probe["end"] - probe["start"] <= 15 for probe in probes))

    def test_gap_recovery_never_bridges_separate_song_takes(self):
        song = LibrarySong(
            "song",
            "title",
            "artist",
            (),
            "https://example.com",
            "hash",
            tuple(LyricLine(index, f"歌詞{index}") for index in range(6)),
        )
        match = SongMatch(
            song,
            (
                LyricAnchor(0, 0, 1, 0.9, 0),
                LyricAnchor(1, 1, 2, 0.9, 0),
                LyricAnchor(2, 4, 5, 0.9, 1),
                LyricAnchor(3, 5, 6, 0.9, 1),
            ),
            0.9,
        )
        with TemporaryDirectory() as directory:
            job_dir = Path(directory)
            result = _recover_lyric_ranges(
                job_dir,
                [
                    Cue(0, 2, "歌詞0", kind="singing"),
                    Cue(2, 4, "歌詞1", kind="singing"),
                    Cue(4, 6, "歌詞4", kind="singing"),
                    Cue(6, 8, "歌詞5", kind="singing"),
                ],
                [0, 1, 2, 3],
                match,
                SongIdentificationConfig(),
                video=job_dir / "source.mp4",
            )

        self.assertEqual(result, ({}, [], []))

    @patch("subtitle_pipeline.song_identification.socket.getaddrinfo")
    def test_public_url_accepts_proxy_fake_ip_for_domain_only(self, getaddrinfo):
        getaddrinfo.return_value = [
            (2, 1, 6, "", ("198.18.2.126", 443)),
        ]

        self.assertTrue(_public_http_url("https://utaten.com/lyric/example/"))
        self.assertFalse(_public_http_url("https://198.18.2.126/private"))
        self.assertFalse(_public_http_url("https://127.0.0.1/private"))

    def test_groups_singing_phrases_without_absorbing_speech(self):
        cues = [
            Cue(0, 2, "intro", "a", "speech"),
            Cue(10, 14, "line one", "a", "singing"),
            Cue(15, 19, "line two", "a", "singing"),
            Cue(20, 25, "talk", "a", "speech"),
            Cue(45, 50, "next song", "a", "singing"),
        ]

        groups = group_song_search_groups(cues, 20)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].cue_ids, (1, 2))
        self.assertEqual(groups[1].cue_ids, (4,))

    def test_ocr_candidate_requires_distinct_persistent_frames(self):
        observations = [
            ("おじゃま虫 / DECO*27", 0.9, 0, 10.0, (1.0, 2.0, 3.0, 4.0)),
            ("おじゃま虫/DECO*27", 0.8, 1, 11.0, (1.0, 2.0, 3.0, 4.0)),
            ("chat message", 0.99, 2, 12.0, (5.0, 6.0, 7.0, 8.0)),
        ]

        candidates = aggregate_ocr_observations(observations, 2)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].frames, 2)
        self.assertIn("おじゃま虫", candidates[0].text)

    def test_song_ocr_runs_only_for_explicit_music_titles(self):
        for keyword in ("歌枠", "弾き語り", "カラオケ", "歌ってみた", "セトリ"):
            with self.subTest(keyword=keyword):
                self.assertTrue(_title_uses_music_mode({"title": f"今夜の{keyword}"}))

        self.assertFalse(
            _title_uses_music_mode(
                {"title": "【ゲーム実況】魔法少女ノ魔女裁判をあそぶ"}
            )
        )

    @patch("subtitle_pipeline.song_identification._EasyOCR")
    def test_non_music_video_skips_song_ocr_worker(self, easy_ocr):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"video")

            identify_and_align_songs(
                video,
                [Cue(0, 5, "not a lyric", kind="singing")],
                {"title": "【ゲーム実況】魔法少女ノ魔女裁判をあそぶ"},
                root,
                SongIdentificationConfig(
                    enabled=True,
                    lyrics_library_path=str(root / "lyrics.sqlite3"),
                ),
            )

        easy_ocr.assert_not_called()

    @patch(
        "subtitle_pipeline.song_identification.collect_ocr_candidates",
        return_value=[],
    )
    @patch("subtitle_pipeline.song_identification._EasyOCR")
    def test_music_video_runs_one_ocr_request_per_group(self, easy_ocr, collect):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"video")

            identify_and_align_songs(
                video,
                [Cue(0, 5, "not a lyric", kind="singing")],
                {"title": "【歌枠】一曲歌います"},
                root,
                SongIdentificationConfig(
                    enabled=True,
                    lyrics_library_path=str(root / "lyrics.sqlite3"),
                ),
            )

        easy_ocr.assert_called_once()
        collect.assert_called_once()
        self.assertFalse(
            _title_uses_music_mode(
                {
                    "title": "朝活雑談",
                    "description": "歌枠と弾き語りの再生リストはこちら",
                }
            )
        )

    @patch("subtitle_pipeline.song_identification._extract_frame")
    def test_song_ocr_reads_one_frame_at_group_start(self, extract_frame):
        extract_frame.return_value = Path("frame.jpg")
        ocr = MagicMock()
        ocr.read.return_value = [("♪ Ready Steady ♪", 0.95, (1.0, 2.0, 3.0, 4.0))]

        candidates = collect_ocr_candidates(
            Path("video.mp4"),
            SongSearchGroup(12.5, 30.0, (0, 1)),
            Path("frames"),
            SongIdentificationConfig(),
            ocr,
        )

        extract_frame.assert_called_once_with(Path("video.mp4"), Path("frames"), 12.5)
        ocr.read.assert_called_once_with(Path("frame.jpg"))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].frames, 1)
        self.assertEqual(candidates[0].first_time, 12.5)
        self.assertEqual(candidates[0].bounds, (1.0, 2.0, 3.0, 4.0))

    def test_search_uses_combined_alt_then_trusted_ocr_query(self):
        hypotheses = [f"これは十分に長い歌詞の候補です{index}" for index in range(6)]
        queries = _build_lyric_search_queries(
            hypotheses, "テストソングタイトル0", set()
        )

        self.assertEqual([item["source"] for item in queries], ["asr", "ocr"])
        self.assertIn(" ", queries[0]["phrase"])
        self.assertEqual(queries[1]["phrase"], "テストソングタイトル0")
        self.assertTrue(all('"' not in item["query"] for item in queries))
        self.assertTrue(all("site:" not in item["query"] for item in queries))

    def test_search_without_ocr_combines_two_alt_fragments(self):
        hypotheses = [f"これは十分に長い歌詞の候補です{index}" for index in range(6)]

        queries = _build_lyric_search_queries(hypotheses, None, set())

        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0]["source"], "asr")
        self.assertEqual(len(queries[0]["phrase"].split()), 2)

    def test_search_prefers_lyric_fragments_over_conversational_asr_text(self):
        queries = _build_lyric_search_queries(
            [
                "今日はどのようなテンポで演奏するかちょっとまだわからないですけどね",
                "愛とか恋とか全部くだらない がっかりするだけ ダメを知るだけ",
                "誰が何と言おうと 正しさなんてわかんないぜ",
            ],
            None,
            set(),
        )

        phrase = queries[0]["phrase"]
        self.assertIn("愛とか恋とか全部くだらない", phrase)
        self.assertIn("正しさなんてわかんないぜ", phrase)

    def test_search_preserves_complete_alt_fragment(self):
        fragment = "誰が何と言おうと正しさなんてわかんないぜそれでも僕は歌い続ける"

        queries = _build_lyric_search_queries([fragment], None, set())

        self.assertEqual(queries[0]["phrase"], fragment)
        self.assertEqual(queries[0]["query"], f"{fragment} 歌詞")

    def test_search_uses_only_llm_selected_ocr_title(self):
        queries = _build_lyric_search_queries([], "Ready Steady", set())

        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0]["source"], "ocr")
        self.assertEqual(queries[0]["phrase"], "Ready Steady")

    def test_llm_selects_only_current_song_title_from_positioned_ocr(self):
        request = MagicMock(
            return_value={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "groups": [
                                        {
                                            "group_id": 0,
                                            "song_title": "Bling-Bang-Bang-Born",
                                        },
                                        {"group_id": 1, "song_title": None},
                                    ]
                                }
                            )
                        },
                    }
                ]
            }
        )
        candidates = [
            [
                OCRCandidate(
                    "Now Singing", 0.99, 1, 0.0, 0.0, (10.0, 10.0, 100.0, 30.0)
                ),
                OCRCandidate(
                    "Bling-Bang-Bang-Born",
                    0.90,
                    1,
                    0.0,
                    0.0,
                    (10.0, 40.0, 250.0, 70.0),
                ),
                OCRCandidate(
                    "鈴木このみ", 0.99, 1, 0.0, 0.0, (10.0, 300.0, 120.0, 330.0)
                ),
            ],
            [],
        ]

        titles = select_ocr_song_titles(
            candidates,
            video_title="【歌枠】test",
            request=request,
            model="test-model",
            json_mode=True,
            thinking=None,
        )

        self.assertEqual(titles, ["Bling-Bang-Bang-Born", None])
        prompt = request.call_args.args[0]["messages"][1]["content"]
        self.assertIn('"bounds":[10.0,40.0,250.0,70.0]', prompt)

    def test_search_ocr_excludes_previously_confirmed_song_name(self):
        queries = _build_lyric_search_queries([], "OLD SONG", {"oldsong"})

        self.assertEqual(queries, [])

    def test_search_group_excludes_speech_between_singing_anchors(self):
        cues = [
            Cue(0, 2, "song start", "a", "singing"),
            Cue(2, 8, "misclassified lyric", "a", "speech"),
            Cue(8, 10, "song end", "a", "singing"),
        ]

        groups = group_song_search_groups(cues, 10)

        self.assertEqual(groups[0].cue_ids, (0, 2))

    def test_web_search_policy_uses_title_mode_and_merged_alt_coverage(self):
        overlapping = [
            Cue(0, 8, "first", kind="singing"),
            Cue(6, 10, "second", kind="singing"),
        ]
        self.assertEqual(_merged_cue_duration(overlapping), 10.0)

        relaxed = _web_search_policy(
            overlapping,
            {"title": "【歌枠】歌います"},
            has_trusted_ocr=False,
        )
        standard = _web_search_policy(
            overlapping,
            {"title": "ゲーム実況"},
            has_trusted_ocr=False,
        )

        self.assertEqual(relaxed["mode"], "relaxed")
        self.assertTrue(relaxed["eligible"])
        self.assertEqual(standard["mode"], "standard")
        self.assertFalse(standard["eligible"])

    def test_standard_search_accepts_fifteen_seconds_or_trusted_ocr(self):
        long_alt = [Cue(0, 15, "long lyric", kind="singing")]
        short_alt = [Cue(0, 2, "short lyric", kind="singing")]

        self.assertTrue(
            _web_search_policy(long_alt, {"title": "雑談"}, has_trusted_ocr=False)[
                "eligible"
            ]
        )
        ocr_policy = _web_search_policy(
            short_alt, {"title": "雑談"}, has_trusted_ocr=True
        )
        self.assertTrue(ocr_policy["eligible"])
        self.assertEqual(ocr_policy["decision_reason"], "trusted_ocr")

    @patch(
        "subtitle_pipeline.song_identification._supported_lyrics_url",
        return_value=True,
    )
    def test_search_summary_ranking_prioritizes_ocr_title_match(self, _supported):
        results = [
            {
                "title": "unrelated result",
                "url": "https://lyrics.test/first",
                "snippet": "Ready Steady appears only in this summary",
            },
            {
                "title": "Ready Steady 歌詞",
                "url": "https://lyrics.test/second",
                "snippet": "song information",
            },
        ]

        ranked = _rank_lyric_search_results(
            results,
            phrase="Ready Steady / Giga",
            source="ocr",
            normalizer=JapaneseNormalizer(),
        )

        self.assertEqual(ranked[0]["url"], "https://lyrics.test/second")
        self.assertTrue(ranked[0]["ranking"]["ocr_title_match"])
        self.assertEqual(ranked[0]["summary_rank"], 0)

    @patch("subtitle_pipeline.song_identification._match_candidates")
    @patch("subtitle_pipeline.song_identification._rank_lyric_search_results")
    @patch("subtitle_pipeline.song_identification._WebTools")
    def test_search_stops_after_first_confirmed_query_and_fetches_three_pages(
        self, web_tools_type, rank_results, match_candidates
    ):
        tools = MagicMock()
        web_tools_type.return_value = tools
        tools.search.return_value = json.dumps([])
        tools.fetch_lyrics.side_effect = lambda url: (
            f"Song {url[-1]}",
            "Artist",
            [url, "line two", "line three"],
        )
        tools.errors = []
        rank_results.return_value = [
            {"url": f"https://lyrics.test/{index}"} for index in range(4)
        ]
        match_candidates.return_value = object()
        queries = [
            {"source": "asr", "phrase": "first phrase", "query": "first"},
            {"source": "asr", "phrase": "second phrase", "query": "second"},
        ]

        songs, audit = _search_canonical_lyrics(
            ["first phrase"], queries, SongIdentificationConfig()
        )

        self.assertEqual(len(songs), 3)
        self.assertEqual(tools.search.call_count, 1)
        self.assertEqual(tools.fetch_lyrics.call_count, 3)
        self.assertEqual(audit["confirmed_after_query"], 0)

    @patch("subtitle_pipeline.song_identification._match_candidates")
    @patch("subtitle_pipeline.song_identification._rank_lyric_search_results")
    @patch("subtitle_pipeline.song_identification._WebTools")
    def test_search_runs_at_most_two_queries_and_six_unique_fetches(
        self, web_tools_type, rank_results, match_candidates
    ):
        tools = MagicMock()
        web_tools_type.return_value = tools
        tools.search.return_value = json.dumps([])
        tools.fetch_lyrics.side_effect = lambda url: (
            f"Song {url.rsplit('/', 1)[-1]}",
            "Artist",
            [url, "line two", "line three"],
        )
        tools.errors = []
        rank_results.side_effect = [
            [{"url": f"https://lyrics.test/a{index}"} for index in range(4)],
            [{"url": f"https://lyrics.test/b{index}"} for index in range(4)],
        ]
        match_candidates.return_value = None
        queries = [
            {"source": "asr", "phrase": f"phrase {index}", "query": f"query {index}"}
            for index in range(3)
        ]

        songs, audit = _search_canonical_lyrics(
            ["first phrase"], queries, SongIdentificationConfig()
        )

        self.assertEqual(len(songs), 6)
        self.assertEqual(tools.search.call_count, 2)
        self.assertEqual(tools.fetch_lyrics.call_count, 6)
        self.assertEqual(len(audit["queries"]), 2)
        self.assertIsNone(audit["confirmed_after_query"])

    @patch("subtitle_pipeline.song_identification._match_candidates")
    @patch("subtitle_pipeline.song_identification._rank_lyric_search_results")
    @patch("subtitle_pipeline.song_identification._WebTools")
    def test_search_deduplicates_identical_lyrics_from_multiple_pages(
        self, web_tools_type, rank_results, match_candidates
    ):
        tools = MagicMock()
        web_tools_type.return_value = tools
        tools.search.return_value = json.dumps([])
        tools.fetch_lyrics.side_effect = lambda url: (
            "Same Song",
            "Singer",
            [
                "同じ歌詞！" if url.endswith("0") else " 同じ歌詞! ",
                "同じ二行目",
                "同じ三行目",
            ],
        )
        tools.errors = []
        rank_results.return_value = [
            {"url": f"https://lyrics.test/{index}"} for index in range(3)
        ]
        match_candidates.return_value = None

        songs, _audit = _search_canonical_lyrics(
            ["同じ歌詞"],
            [{"source": "asr", "phrase": "同じ歌詞", "query": "同じ歌詞"}],
            SongIdentificationConfig(),
        )

        self.assertEqual(len(songs), 1)
        self.assertEqual(tools.fetch_lyrics.call_count, 3)

    @patch("subtitle_pipeline.song_identification._verify_lyric_range")
    @patch("subtitle_pipeline.song_identification._materialize_verification_stems")
    @patch("subtitle_pipeline.song_identification.shutil.which", return_value="uv")
    def test_initial_alt_anchor_uses_unified_range_verifier(
        self, _which, materialize, verify
    ):
        song = LibrarySong(
            "song",
            "Title",
            "Singer",
            (),
            "https://example.com/song",
            "hash",
            (LyricLine(0, "歌詞"),),
        )
        match = SongMatch(song, (LyricAnchor(0, 0, 1, 0.9),), 0.9)
        verified_cue = Cue(10.2, 14.8, "歌詞", kind="singing")
        verify.return_value = ([verified_cue], [], {"status": "aligned"})
        with TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker.py"
            worker.write_text("", encoding="utf-8")
            config = SongIdentificationConfig(pyshiro_worker_project=str(root))
            timings, audits = _align_match_with_pyshiro(
                root,
                root / "video.mp4",
                [Cue(10, 15, "歌詞", kind="singing")],
                [0],
                match,
                config,
            )

        probe = verify.call_args.args[0]
        self.assertEqual(probe["route"], "initial_alt_anchor")
        self.assertEqual(probe["success_status"], "aligned")
        self.assertEqual(timings[0][0][1:3], (10.2, 14.8))
        self.assertEqual(audits, [{"status": "aligned"}])
        materialize.assert_called_once()

    def test_verified_lyrics_trim_overlapping_speech_at_aligner_units(self):
        speech = Cue(
            0,
            4,
            "歌詞です次の話",
            "speaker",
            "speech",
            source_units=(
                TimedTextUnit("歌詞", 0.0, 1.0),
                TimedTextUnit("です", 1.0, 2.0),
                TimedTextUnit("次の", 2.2, 3.0),
                TimedTextUnit("話", 3.0, 4.0),
            ),
        )
        lyric = Cue(0, 2.05, "公式歌詞", kind="singing")
        spans = [VerifiedLyricSpan(0, 2.05, 9, "song", (1,), -15.0)]

        result, audit = arbitrate_verified_lyrics([speech, lyric], spans)

        self.assertEqual(
            [(cue.kind, cue.text) for cue in result],
            [("singing", "公式歌詞"), ("speech", "次の話")],
        )
        self.assertEqual(audit[0]["action"], "trimmed_at_aligner_units")

    def test_verified_lyrics_keep_unitless_speech_below_eighty_percent(self):
        speech = Cue(0, 10, "途中の話", kind="speech")
        spans = [VerifiedLyricSpan(0, 7.9, 1, "song", (0,), -15.0)]

        result, audit = arbitrate_verified_lyrics([speech], spans)

        self.assertEqual(result, [speech])
        self.assertEqual(audit[0]["action"], "kept_conflict")

    def test_verified_lyrics_discard_unitless_speech_at_eighty_percent(self):
        speech = Cue(0, 10, "误识别为讲话", kind="speech")
        spans = [VerifiedLyricSpan(0, 8, 1, "song", (0,), -15.0)]

        result, audit = arbitrate_verified_lyrics([speech], spans)

        self.assertEqual(result, [])
        self.assertEqual(audit[0]["action"], "discarded_without_units")

    def test_pyshiro_candidate_requires_absolute_and_relative_likelihood(self):
        config = SongIdentificationConfig(
            pyshiro_likelihood_floor=-30,
            pyshiro_likelihood_margin=0.2,
        )

        accepted = _pyshiro_likelihood_wins(
            {"likelihood_per_frame": -20},
            [{"likelihood_per_frame": -20.2}],
            config,
        )
        too_close = _pyshiro_likelihood_wins(
            {"likelihood_per_frame": -20},
            [{"likelihood_per_frame": -20.1}],
            config,
        )
        below_floor = _pyshiro_likelihood_wins(
            {"likelihood_per_frame": -31}, [], config
        )

        self.assertTrue(accepted[0])
        self.assertFalse(too_close[0])
        self.assertFalse(below_floor[0])

    def test_pyshiro_gap_uses_smaller_margin(self):
        config = SongIdentificationConfig(
            pyshiro_likelihood_floor=-30,
            pyshiro_gap_likelihood_margin=0.1,
            pyshiro_likelihood_margin=0.2,
        )

        accepted = _pyshiro_likelihood_wins(
            {"likelihood_per_frame": -20.0},
            [{"likelihood_per_frame": -20.15}],
            config,
            margin=config.pyshiro_gap_likelihood_margin,
        )
        strict = _pyshiro_likelihood_wins(
            {"likelihood_per_frame": -20.0},
            [{"likelihood_per_frame": -20.15}],
            config,
        )

        self.assertTrue(accepted[0])
        self.assertFalse(strict[0])

    def test_strong_anchor_uses_floor_without_competitor_preference(self):
        config = SongIdentificationConfig(pyshiro_likelihood_floor=-30)

        accepted = _pyshiro_likelihood_wins(
            {"likelihood_per_frame": -20.0},
            [{"likelihood_per_frame": -19.0}],
            config,
            margin=0,
            require_preference=False,
        )
        below_floor = _pyshiro_likelihood_wins(
            {"likelihood_per_frame": -31.0},
            [{"likelihood_per_frame": -40.0}],
            config,
            margin=0,
            require_preference=False,
        )

        self.assertTrue(accepted[0])
        self.assertFalse(below_floor[0])

    def test_anchor_strength_requires_multiple_lines_or_contiguous_evidence(self):
        isolated = LyricAnchor(0, 0, 1, 0.9)
        left = LyricAnchor(1, 2, 3, 0.9)
        right = LyricAnchor(2, 3, 4, 0.9)
        multi_line = LyricAnchor(4, 5, 7, 0.9)

        self.assertFalse(_anchor_has_continuous_support(isolated, (isolated, left)))
        self.assertTrue(_anchor_has_continuous_support(left, (left, right)))
        self.assertTrue(_anchor_has_continuous_support(multi_line, (multi_line,)))

    def test_competing_lyrics_exclude_equivalent_normalized_readings(self):
        song = LibrarySong(
            "song",
            "title",
            "artist",
            (),
            "https://example.com",
            "hash",
            (
                LyricLine(0, "今日は", "きょうは"),
                LyricLine(1, "別の歌詞", "べつのかし"),
                LyricLine(2, "今日 は", "きょうは"),
                LyricLine(3, "最後の歌詞", "さいごのかし"),
            ),
        )

        competitors = _competing_line_sets(
            song, [0], normalizer=JapaneseNormalizer(), minimum=4
        )

        self.assertNotIn("今日 は", [lines[0].text for lines in competitors])
        self.assertIn("別の歌詞", [lines[0].text for lines in competitors])

    def test_pyshiro_line_ranges_expand_one_asr_cue_into_lyric_line_cues(self):
        cues = [Cue(10, 20, "misheard lyrics", "singer", "singing")]
        song = LibrarySong(
            "song",
            "title",
            "artist",
            (),
            "https://example.com/lyrics",
            "hash",
            (
                LyricLine(0, "一行目", translation="第一行"),
                LyricLine(1, "二行目", translation="第二行"),
            ),
        )
        match = SongMatch(song, (LyricAnchor(0, 0, 2, 0.9),), 0.9)

        replacements, alignments = _apply_local_match(
            cues,
            [0],
            match,
            {
                0: (
                    (0, 10.5, 14.0, (TimedTextUnit("一行目", 10.5, 14.0),)),
                    (1, 14.2, 19.5, (TimedTextUnit("二行目", 14.2, 19.5),)),
                )
            },
        )

        self.assertEqual(
            replacements[0],
            [
                Cue(
                    10.5,
                    14.0,
                    "一行目",
                    "singer",
                    "singing",
                    preferred_translation="第一行",
                    source_units=(TimedTextUnit("一行目", 10.5, 14.0),),
                    language="Japanese",
                ),
                Cue(
                    14.2,
                    19.5,
                    "二行目",
                    "singer",
                    "singing",
                    preferred_translation="第二行",
                    source_units=(TimedTextUnit("二行目", 14.2, 19.5),),
                    language="Japanese",
                ),
            ],
        )
        self.assertEqual([item["lyric_line_ids"] for item in alignments], [[0], [1]])

    def test_verified_english_lyric_keeps_english_language_and_timing(self):
        cues = [Cue(10, 14, "misheard", "singer", "singing")]
        song = LibrarySong(
            "song",
            "title",
            "artist",
            (),
            "https://example.com/lyrics",
            "hash",
            (
                LyricLine(
                    0, "Ready set and find out!", translation="准备好就去找到答案"
                ),
            ),
        )
        match = SongMatch(song, (LyricAnchor(0, 0, 1, 0.9),), 0.9)

        replacements, _alignments = _apply_local_match(
            cues,
            [0],
            match,
            {
                0: (
                    (
                        0,
                        10.2,
                        13.8,
                        (TimedTextUnit("Ready set and find out!", 10.2, 13.8),),
                    ),
                )
            },
        )

        self.assertEqual(replacements[0][0].language, "English")
        self.assertEqual(replacements[0][0].text, "Ready set and find out!")

    def test_applies_only_verified_ordered_lyric_alignment(self):
        cues = [
            Cue(0, 1, "talk", "a", "speech"),
            Cue(1, 4, "wrong one", "a", "singing"),
            Cue(4, 7, "wrong two", "a", "singing"),
            Cue(7, 8, "talk", "a", "speech"),
        ]
        reports = [
            {
                "confidence": "medium",
                # SongSearchGroup.cue_ids is a tuple until the report is serialized.
                "search_group": {"cue_ids": (1, 2)},
                "alignments": [
                    {
                        "asr_cue_ids": [1, 2],
                        "match": "lyrics",
                        "corrected_text": "corrected lyric",
                    }
                ],
            }
        ]

        corrected = apply_lyric_corrections(cues, reports)

        self.assertEqual(len(corrected), 3)
        self.assertEqual(
            corrected[1],
            Cue(1, 7, "corrected lyric", "a", "singing", language="English"),
        )

    def test_does_not_apply_low_confidence_or_out_of_search_group_alignment(self):
        cues = [Cue(0, 2, "raw", "a", "singing")]
        reports = [
            {
                "confidence": "low",
                "search_group": {"cue_ids": [0]},
                "alignments": [
                    {"asr_cue_ids": [0], "match": "lyrics", "corrected_text": "bad"}
                ],
            },
            {
                "confidence": "high",
                "search_group": {"cue_ids": []},
                "alignments": [
                    {"asr_cue_ids": [0], "match": "lyrics", "corrected_text": "bad"}
                ],
            },
        ]

        self.assertEqual(apply_lyric_corrections(cues, reports), cues)


if __name__ == "__main__":
    unittest.main()
