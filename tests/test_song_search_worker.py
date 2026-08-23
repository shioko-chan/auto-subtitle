import importlib.util
import unittest
from pathlib import Path


def _worker_module():
    path = Path("tools/song_search/worker.py").resolve()
    spec = importlib.util.spec_from_file_location("song_search_worker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SongSearchWorkerTests(unittest.TestCase):
    def test_parses_structured_utaten_lyrics_without_rt_readings(self):
        worker = _worker_module()
        document = """
        <script type="application/ld+json">
        {"@type":"MusicComposition","name":"夢現妄想世界",
         "byArtist":{"name":"夢限大みゅーたいぷ"}}
        </script>
        <div class="hiragana">
          <span class="ruby"><span class="rb">伝</span><span class="rt">つた</span></span>えたくて<br>
          ガイドラインは読めるか<br>
          丁寧に書き連ねても<br>
          邪魔ばっか入るか
        </div>
        """
        value = worker._parse_utaten(document)
        self.assertEqual(value["title"], "夢現妄想世界")
        self.assertEqual(value["artist"], "夢限大みゅーたいぷ")
        self.assertEqual(value["lines"][0], "伝えたくて")
        self.assertEqual(len(value["lines"]), 4)


if __name__ == "__main__":
    unittest.main()
