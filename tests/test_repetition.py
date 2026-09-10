import unittest

from subtitle_pipeline.repetition import find_repetition_loop


class RepetitionTests(unittest.TestCase):
    def test_detects_single_character_generation_loop(self):
        match = find_repetition_loop("哒" * 500)

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.pattern, "哒")
        self.assertEqual(match.repeats, 160)

    def test_detects_repeated_sentence(self):
        phrase = "I'm so tired of being a nobody."
        match = find_repetition_loop(phrase * 20)

        self.assertIsNotNone(match)
        assert match is not None
        self.assertGreaterEqual(match.repeats, 4)

    def test_ignores_short_natural_repetition(self):
        self.assertIsNone(find_repetition_loop("はいはいはいはい、大丈夫です"))


if __name__ == "__main__":
    unittest.main()
