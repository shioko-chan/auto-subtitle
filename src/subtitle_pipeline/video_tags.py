"""Deterministic upload tags from title and curated identity data."""
from __future__ import annotations

import re

_CONTENT_TAGS = (
    (r"歌枠|歌回|歌唱枠|karaoke", "歌回"),
    (r"雑談|杂谈|ざつだん", "杂谈"),
    (r"朝活|おはよう", "晨间直播"),
    (r"弾き語り|弹唱", "弹唱"),
    (r"作業|作业", "作业直播"),
)
_GAME_TAGS = (
    (r"バイオ(?:ハザード)?\s*7|生化危机\s*7|resident evil\s*7", "生化危机7"),
    (r"マインクラフト|マイクラ|minecraft|我的世界", "我的世界"),
    (r"魔法少女ノ魔女裁判|魔法少女的魔女审判", "魔法少女的魔女审判"),
    (r"夜間警備|夜间警备", "夜间警备"),
)


def video_tags(title: str, metadata: dict, context: dict) -> list[str]:
    # Descriptions often list every band member; they are not identity evidence.
    identity = " ".join(str(metadata.get(key) or "") for key in
                        ("channel", "uploader", "uploader_id")) + " " + title
    identity = "".join(identity.casefold().split())
    tags = []
    for character in context.get("characters", []):
        names = [character.get("source_name"), character.get("canonical"),
                 *character.get("aliases", [])]
        if any(isinstance(name, str) and name and
               "".join(name.casefold().split()) in identity for name in names):
            tags.append(character["canonical"])
    tags.extend(item["name"] for item in context.get("franchises", []))
    for pattern, tag in (*_CONTENT_TAGS, *_GAME_TAGS):
        if re.search(pattern, title, re.IGNORECASE):
            tags.append(tag)
    tags.append("中文字幕")
    return list(dict.fromkeys(tags))
