import tempfile
import unittest
from pathlib import Path

from subtitle_pipeline.lyrics_library import LibrarySong, LyricLine, LyricsLibrary
from subtitle_pipeline.lyrics_matching import match_song


class LyricsLibraryTests(unittest.TestCase):
    def test_canonical_source_is_required_and_asr_has_no_write_path(self):
        with tempfile.TemporaryDirectory() as directory:
            library = LyricsLibrary(Path(directory) / "lyrics.sqlite3")
            with self.assertRaises(ValueError):
                library.store_canonical_song(
                    title="Song",
                    artist="Artist",
                    aliases=[],
                    source_url="",
                    lines=[("one", None), ("two", None), ("three", None)],
                )
            song = library.store_canonical_song(
                title="夢現妄想世界",
                artist="夢限大みゅーたいぷ",
                aliases=[],
                source_url="https://utaten.com/lyric/example/",
                lines=[
                    ("伝えたくて", None),
                    ("丁寧に書き連ねても", None),
                    ("邪魔ばっか入る", None),
                ],
            )
            library.store_translations(
                song.song_id,
                {0: "想要告诉你", 1: "即使认真写下", 2: "也总被打扰"},
                source="llm",
            )
            loaded = library.get(song.song_id)
            self.assertEqual(loaded.lines[0].translation, "想要告诉你")
            library.store_translations(song.song_id, {0: "官方译词"}, source="official")
            library.store_translations(
                song.song_id, {0: "人工确认译词"}, source="verified"
            )
            library.store_translations(
                song.song_id, {0: "后来写入的官方译词"}, source="official"
            )
            library.store_translations(
                song.song_id, {0: "后来写入的人工译词"}, source="verified"
            )
            library.store_translations(
                song.song_id, {0: "较低优先级译词"}, source="llm"
            )
            verified_line = library.get(song.song_id).lines[0]
            self.assertEqual(verified_line.translation, "人工确认译词")
            self.assertEqual(verified_line.translation_source, "verified")
            library.close()

    def test_matches_multiple_sequential_noisy_anchors(self):
        song = LibrarySong(
            "song",
            "title",
            "artist",
            (),
            "https://example.com",
            "hash",
            (
                LyricLine(0, "伝えたくて"),
                LyricLine(1, "ガイドラインは読めるか"),
                LyricLine(2, "丁寧に書き連ねても"),
                LyricLine(3, "邪魔ばっか入るか"),
            ),
        )
        result = match_song(
            [
                "伝えたくて",
                "ガイドライン読めるか",
                "丁寧に書き連ねても",
                "邪魔ばっか入るか",
            ],
            [song],
            minimum_anchors=3,
        )
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(result.anchors), 3)


if __name__ == "__main__":
    unittest.main()
