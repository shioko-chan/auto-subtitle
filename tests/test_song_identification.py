import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from subtitle_pipeline.config import SongIdentificationConfig
from subtitle_pipeline.lyrics_library import LibrarySong, LyricLine, LyricsLibrary
from subtitle_pipeline.lyrics_matching import LyricAnchor, SongMatch
from subtitle_pipeline.song_identification import (
    OCRCandidate,
    SongIdentificationResult,
    _apply_local_match,
    _build_lyric_search_queries,
    _load_cache,
    _public_http_url,
    _recover_lyric_gaps,
    _signature,
    aggregate_ocr_observations,
    apply_lyric_corrections,
    group_singing_episodes,
    identify_and_align_songs,
    translate_aligned_song_lyrics,
)
from subtitle_pipeline.subtitles import Cue, TimedTextUnit


class SongIdentificationTests(unittest.TestCase):
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
                        "episode": {"start": 1, "end": 3},
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

    def test_song_cache_accepts_empty_corrected_cues(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "song-cache.json"
            path.write_text(
                '{"version":10,"signature":"test","reports":[],'
                '"corrected_cues":[]}',
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
    def test_unmatched_song_episode_discards_all_internal_text(
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
                    enabled=True,
                    lyrics_library_path=str(root / "lyrics.sqlite3")
                ),
            )

        self.assertEqual(
            [cue.text for cue in result.corrected_cues], ["outside speech"]
        )
        self.assertEqual(
            result.reports[0]["evidence"],
            ["no_continuous_canonical_lyric_match"],
        )

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
    @patch(
        "subtitle_pipeline.song_identification._extract_vocal_window",
        return_value=True,
    )
    @patch("subtitle_pipeline.song_identification.shutil.which", return_value="uv")
    def test_recovers_internal_lyric_gap_only_after_all_checks(
        self, _which, _extract, _activity, run_pyshiro
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
            {"ok": True, "likelihood_per_frame": -20.0},
            {
                "ok": True,
                "likelihood_per_frame": -21.0,
                "lines": [[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]],
                "units": [
                    [{"text": "最初", "start": 0.0, "end": 2.0}],
                    [{"text": "漏れた歌詞", "start": 2.0, "end": 4.0}],
                    [{"text": "最後", "start": 4.0, "end": 6.0}],
                ],
            },
        ]
        with TemporaryDirectory() as directory:
            job_dir = Path(directory)
            manifest_dir = job_dir / "vocal-candidates"
            manifest_dir.mkdir()
            (manifest_dir / "manifest.json").write_text(
                '[{"start":0,"end":20,"path":"vocals.wav"}]',
                encoding="utf-8",
            )

            replacements, alignments, audit = _recover_lyric_gaps(
                job_dir,
                [
                    Cue(0, 8, "最初の歌詞", "singer", "singing"),
                    Cue(8, 16, "最後の歌詞", "singer", "singing"),
                ],
                [0, 1],
                match,
                SongIdentificationConfig(),
            )

        self.assertEqual([cue.text for cue in replacements[0]], ["漏れた歌詞"])
        self.assertEqual(alignments[0]["lyric_line_ids"], [1])
        self.assertEqual(audit[0]["status"], "gap_recovered")

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
            manifest_dir = job_dir / "vocal-candidates"
            manifest_dir.mkdir()
            (manifest_dir / "manifest.json").write_text(
                '[{"start":0,"end":20,"path":"vocals.wav"}]',
                encoding="utf-8",
            )
            result = _recover_lyric_gaps(
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
            )

        self.assertEqual(result, ({}, [], []))

    @patch("subtitle_pipeline.song_identification._run_pyshiro_lines")
    @patch(
        "subtitle_pipeline.song_identification._vocal_active_ratio", return_value=0.5
    )
    @patch(
        "subtitle_pipeline.song_identification._extract_vocal_window",
        return_value=True,
    )
    @patch("subtitle_pipeline.song_identification.shutil.which", return_value="uv")
    def test_acoustic_neighbors_fill_take_suffix_and_next_take_prefix(
        self, _which, _extract, _activity, run_pyshiro
    ):
        song = LibrarySong(
            "song",
            "title",
            "artist",
            (),
            "https://example.com",
            "hash",
            (
                LyricLine(0, "次の冒頭", translation="下一轮开头"),
                LyricLine(1, "次の錨", translation="下一轮锚点"),
                LyricLine(2, "前の錨", translation="上一轮锚点"),
                LyricLine(3, "前の末尾", translation="上一轮结尾"),
                LyricLine(4, "歌われない行", translation="未演唱"),
            ),
        )
        match = SongMatch(
            song,
            (
                LyricAnchor(0, 2, 3, 0.9, 0),
                LyricAnchor(3, 1, 2, 0.9, 1),
            ),
            0.9,
        )
        run_pyshiro.side_effect = [
            {
                "ok": True,
                "likelihood_per_frame": -20.0,
                "lines": [[0.25, 1.75]],
                "units": [[{"text": "前の末尾", "start": 0.25, "end": 1.75}]],
            },
            {
                "ok": True,
                "likelihood_per_frame": -19.0,
                "lines": [[0.2, 1.8]],
                "units": [[{"text": "次の冒頭", "start": 0.2, "end": 1.8}]],
            },
        ]
        with TemporaryDirectory() as directory:
            job_dir = Path(directory)
            manifest_dir = job_dir / "vocal-candidates"
            manifest_dir.mkdir()
            (manifest_dir / "manifest.json").write_text(
                '[{"start":0,"end":20,"path":"vocals.wav"}]',
                encoding="utf-8",
            )

            replacements, alignments, audit = _recover_lyric_gaps(
                job_dir,
                [
                    Cue(0, 2, "前の錨", "singer", "singing"),
                    Cue(2, 4, "前の末尾", "singer", "singing"),
                    Cue(4, 6, "聞き取れない", "singer", "singing"),
                    Cue(6, 8, "次の錨", "singer", "singing"),
                ],
                [0, 1, 2, 3],
                match,
                SongIdentificationConfig(),
            )

        self.assertEqual([cue.text for cue in replacements[1]], ["前の末尾"])
        self.assertEqual([cue.text for cue in replacements[2]], ["次の冒頭"])
        self.assertEqual(
            [item["match"] for item in alignments],
            ["lyrics_acoustic_neighbor", "lyrics_acoustic_neighbor"],
        )
        self.assertEqual(
            [item["status"] for item in audit],
            ["neighbor_recovered", "neighbor_recovered"],
        )

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

        episodes = group_singing_episodes(cues, 20)

        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0].cue_ids, (1, 2))
        self.assertEqual(episodes[1].cue_ids, (4,))

    def test_ocr_candidate_requires_distinct_persistent_frames(self):
        observations = [
            ("おじゃま虫 / DECO*27", 0.9, 0, 10.0),
            ("おじゃま虫/DECO*27", 0.8, 1, 11.0),
            ("chat message", 0.99, 2, 12.0),
        ]

        candidates = aggregate_ocr_observations(observations, 2)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].frames, 2)
        self.assertIn("おじゃま虫", candidates[0].text)

    def test_search_reserves_four_queries_each_for_asr_and_qualified_ocr(self):
        hypotheses = [f"これは十分に長い歌詞の候補です{index}" for index in range(6)]
        ocr = [
            OCRCandidate(f"曲名: テストソングタイトル{index}", 0.95, 3, 0.0, 2.0)
            for index in range(6)
        ]

        queries = _build_lyric_search_queries(hypotheses, ocr, [])

        self.assertEqual(
            [item["source"] for item in queries], ["asr"] * 4 + ["ocr"] * 4
        )
        self.assertTrue(all('"' not in item["query"] for item in queries))
        self.assertTrue(all("site:" not in item["query"] for item in queries))

    def test_search_uses_short_lyric_fragments_instead_of_longest_asr_text(self):
        queries = _build_lyric_search_queries(
            [
                "今日はどのようなテンポで演奏するかちょっとまだわからないですけどね",
                "愛とか恋とか全部くだらない がっかりするだけ ダメを知るだけ",
                "誰が何と言おうと 正しさなんてわかんないぜ",
            ],
            [],
            [],
        )

        phrases = [item["phrase"] for item in queries]
        self.assertIn("愛とか恋とか全部くだらない", phrases)
        self.assertIn("正しさなんてわかんないぜ", phrases)
        self.assertTrue(all(8 <= len(item) <= 20 for item in phrases))

    def test_search_ignores_ocr_ui_noise_and_unstructured_comments(self):
        ocr = [
            OCRCandidate("08/11 TUE", 0.99, 5, 0.0, 4.0),
            OCRCandidate("Date 08/10 MON", 0.99, 5, 0.0, 4.0),
            OCRCandidate("08/11TUE00:4349", 0.99, 5, 0.0, 4.0),
            OCRCandidate("08/1@青UE", 0.99, 5, 0.0, 4.0),
            OCRCandidate("589/10000", 0.99, 5, 0.0, 4.0),
            OCRCandidate("なるほどね~ん", 0.99, 5, 0.0, 4.0),
            OCRCandidate("♪ Ready Steady ♪", 0.99, 5, 0.0, 4.0),
        ]

        queries = _build_lyric_search_queries([], ocr, [])

        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0]["source"], "ocr")
        self.assertEqual(queries[0]["phrase"], "Ready Steady")

    def test_episode_includes_cues_between_singing_anchors(self):
        cues = [
            Cue(0, 2, "song start", "a", "singing"),
            Cue(2, 8, "misclassified lyric", "a", "speech"),
            Cue(8, 10, "song end", "a", "singing"),
        ]

        episodes = group_singing_episodes(cues, 10)

        self.assertEqual(episodes[0].cue_ids, (0, 1, 2))

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
                # SongEpisode.cue_ids is a tuple until the report is serialized.
                "episode": {"cue_ids": (1, 2)},
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

    def test_does_not_apply_low_confidence_or_out_of_episode_alignment(self):
        cues = [Cue(0, 2, "raw", "a", "singing")]
        reports = [
            {
                "confidence": "low",
                "episode": {"cue_ids": [0]},
                "alignments": [
                    {"asr_cue_ids": [0], "match": "lyrics", "corrected_text": "bad"}
                ],
            },
            {
                "confidence": "high",
                "episode": {"cue_ids": []},
                "alignments": [
                    {"asr_cue_ids": [0], "match": "lyrics", "corrected_text": "bad"}
                ],
            },
        ]

        self.assertEqual(apply_lyric_corrections(cues, reports), cues)


if __name__ == "__main__":
    unittest.main()
