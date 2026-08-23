import unittest

from subtitle_pipeline.lyrics_library import LibrarySong, LyricLine
from subtitle_pipeline.lyrics_matching import JapaneseNormalizer, match_song


class JapaneseNormalizerTests(unittest.TestCase):
    def test_display_units_split_kana_and_kanji_to_character_granularity(self):
        normalizer = JapaneseNormalizer()

        units = normalizer.display_units("ガイドラインは読めるか?")

        self.assertEqual(
            [text for text, _reading in units],
            ["ガ", "イ", "ド", "ラ", "イ", "ン", "は", "読", "め", "る", "か?"],
        )
        self.assertEqual(
            "".join(reading for _text, reading in units),
            normalizer("ガイドラインは読めるか?"),
        )

    def test_display_units_use_supplied_reading_for_compound_kanji(self):
        normalizer = JapaneseNormalizer()

        units = normalizer.display_units("今日だ", "きょうだ")

        self.assertEqual([text for text, _reading in units], ["今", "日", "だ"])
        self.assertEqual([reading for _text, reading in units], ["きょ", "う", "だ"])

    def test_spaces_and_punctuation_do_not_generate_spoken_readings(self):
        normalizer = JapaneseNormalizer()

        units = normalizer.display_units("刹那 オーバードライブっぽくさ。")

        self.assertEqual(
            normalizer("刹那 オーバードライブっぽくさ。"),
            "せつなおーばーどらいぶっぽくさ",
        )
        self.assertNotIn("きごう", "".join(reading for _text, reading in units))
        self.assertEqual(
            "".join(text for text, _reading in units), "刹那 オーバードライブっぽくさ。"
        )

    def test_long_vowel_marks_are_preserved_and_attached(self):
        normalizer = JapaneseNormalizer()

        units = normalizer.display_units("ストーリー オーバー ニュータイプ")

        self.assertEqual(
            normalizer("ストーリー オーバー ニュータイプ"),
            "すとーりーおーばーにゅーたいぷ",
        )
        self.assertEqual(
            [(text, reading) for text, reading in units if "ー" in text],
            [
                ("トー", "とー"),
                ("リー ", "りー"),
                ("オー", "おー"),
                ("バー ", "ばー"),
                ("ニュー", "にゅー"),
            ],
        )

    def test_semiglobal_match_finds_a_middle_song_fragment(self):
        song = LibrarySong(
            "song",
            "title",
            "artist",
            (),
            "https://example.com",
            "hash",
            tuple(
                LyricLine(index, text)
                for index, text in enumerate(
                    [
                        "遠いイントロ",
                        "まだ知らない朝",
                        "君と走り出す",
                        "光の方へ行こう",
                        "ここから始まる",
                        "長いアウトロ",
                    ]
                )
            ),
        )

        match = match_song(
            ["君と走り出す", "光の方へ行こう", "ここから始まる"],
            [song],
            minimum_anchors=3,
            minimum_score=0.4,
        )

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(
            [(item.line_start, item.line_end) for item in match.anchors],
            [(2, 3), (3, 4), (4, 5)],
        )


if __name__ == "__main__":
    unittest.main()
