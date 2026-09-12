import unittest

from subtitle_pipeline.video_tags import video_tags


class VideoTagsTests(unittest.TestCase):
    def test_identity_and_game_tags_ignore_description(self):
        context = {
            "characters": [
                {"source_name": "千石ユノ", "canonical": "千石由乃"},
                {"source_name": "仲町あられ", "canonical": "仲町阿拉蕾"},
            ],
            "franchises": [{"name": "梦限大MewType"}, {"name": "BanG Dream!"}],
        }
        self.assertEqual(video_tags("【バイオ7】3回目", {
            "channel": "千石ユノ", "description": "仲町あられ"}, context),
            ["千石由乃", "梦限大MewType", "BanG Dream!", "生化危机7", "中文字幕"])

    def test_unknown_game_is_not_guessed_and_content_tags_are_deduplicated(self):
        self.assertEqual(video_tags("歌枠・歌回・雑談・未知のゲーム", {}, {}),
                         ["歌回", "杂谈", "中文字幕"])
