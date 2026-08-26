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

    def test_english_lyrics_keep_display_text_and_use_katakana_readings(self):
        normalizer = JapaneseNormalizer()

        units = normalizer.display_units("Ready set and find out!")

        self.assertEqual(
            [text for text, _reading in units],
            ["Ready ", "set ", "and ", "find ", "out!"],
        )
        self.assertEqual(
            [reading for _text, reading in units],
            ["れでぃ", "せっと", "あんど", "ふぁいんど", "あうと"],
        )

    def test_mixed_lyrics_only_transliterate_english_words(self):
        normalizer = JapaneseNormalizer()

        units = normalizer.display_units("君と find out!")

        self.assertEqual("".join(text for text, _reading in units), "君と find out!")
        self.assertEqual(
            units,
            [
                ("君", "きみ"),
                ("と ", "と"),
                ("find ", "ふぁいんど"),
                ("out!", "あうと"),
            ],
        )

    def test_english_song_term_override_supplies_alignment_reading(self):
        normalizer = JapaneseNormalizer()

        units = normalizer.display_units("it's only one newtype)")

        self.assertEqual(
            "".join(text for text, _reading in units), "it's only one newtype)"
        )
        self.assertEqual(
            "".join(reading for _text, reading in units),
            "いっつおうんりうぉんにゅーたいぷ",
        )

    def test_single_letter_and_compound_english_words_have_g2p_readings(self):
        normalizer = JapaneseNormalizer()

        units = normalizer.display_units("I wanna be free, NewWorld")

        self.assertEqual(
            "".join(text for text, _reading in units),
            "I wanna be free, NewWorld",
        )
        readings = {text.strip(" ,"): reading for text, reading in units}
        self.assertEqual(readings["I"], "あい")
        self.assertEqual(readings["NewWorld"], "にゅーわーるど")

    def test_out_of_dictionary_english_word_does_not_drop_lyric_line(self):
        normalizer = JapaneseNormalizer()

        units = normalizer.display_units("HyperNewWorldX")

        self.assertTrue(units)
        self.assertEqual("".join(text for text, _reading in units), "HyperNewWorldX")
        self.assertTrue(all(reading for _text, reading in units))

    def test_english_lyrics_participate_in_song_matching(self):
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
                        "Ready set and find out",
                        "Let me save the planet",
                        "We're standing by your side",
                    ]
                )
            ),
        )

        match = match_song(
            [
                "Ready set and find out",
                "Let me save the planet",
                "We're standing by your side",
            ],
            [song],
            anchor_threshold=0.9,
            minimum_anchors=3,
            minimum_score=0.4,
        )

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(
            [(anchor.line_start, anchor.line_end) for anchor in match.anchors],
            [(0, 1), (1, 2), (2, 3)],
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

    def test_semiglobal_match_extracts_repeated_short_version_as_separate_takes(self):
        song = LibrarySong(
            "song",
            "title",
            "artist",
            (),
            "https://example.com",
            "hash",
            tuple(
                LyricLine(index, text)
                for index, text in enumerate(["始まり", "駆け出す", "見つけた"])
            ),
        )

        match = match_song(
            [
                "始まり",
                "駆け出す",
                "見つけた",
                "始まり",
                "駆け出す",
                "見つけた",
                "始まり",
                "駆け出す",
                "見つけた",
            ],
            [song],
            anchor_threshold=0.9,
            minimum_anchors=3,
            minimum_score=0.4,
        )

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(
            [
                (anchor.cue_index, anchor.line_start, anchor.take_index)
                for anchor in match.anchors
            ],
            [
                (0, 0, 0),
                (1, 1, 0),
                (2, 2, 0),
                (3, 0, 1),
                (4, 1, 1),
                (5, 2, 1),
                (6, 0, 2),
                (7, 1, 2),
                (8, 2, 2),
            ],
        )

    def test_repeated_chorus_written_twice_remains_one_monotonic_take(self):
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
                    ["声を上げて", "走り出そう", "光の先へ"] * 2
                )
            ),
        )

        match = match_song(
            ["声を上げて", "走り出そう", "光の先へ"] * 2,
            [song],
            anchor_threshold=0.9,
            minimum_anchors=3,
            minimum_score=0.4,
        )

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(len(match.anchors), 6)
        self.assertEqual({anchor.take_index for anchor in match.anchors}, {0})

    def test_long_asr_window_can_anchor_more_than_four_lyric_lines(self):
        lines = [f"第{index}行の歌詞です" for index in range(12)]
        song = LibrarySong(
            "song",
            "title",
            "artist",
            (),
            "https://example.com",
            "hash",
            tuple(LyricLine(index, text) for index, text in enumerate(lines)),
        )

        match = match_song(
            ["".join(lines[:6]), "".join(lines[6:])],
            [song],
            anchor_threshold=0.9,
            minimum_anchors=2,
            minimum_score=0.4,
        )

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(
            [(anchor.line_start, anchor.line_end) for anchor in match.anchors],
            [(0, 6), (6, 12)],
        )

    def test_two_anchor_take_does_not_bypass_total_song_minimum(self):
        song = LibrarySong(
            "song",
            "title",
            "artist",
            (),
            "https://example.com",
            "hash",
            (
                LyricLine(0, "最初の歌詞"),
                LyricLine(1, "次の歌詞"),
                LyricLine(2, "最後の歌詞"),
            ),
        )

        match = match_song(
            ["最初の歌詞", "次の歌詞"],
            [song],
            anchor_threshold=0.9,
            minimum_anchors=3,
            minimum_score=0.4,
        )

        self.assertIsNone(match)


if __name__ == "__main__":
    unittest.main()
