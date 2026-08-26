from __future__ import annotations

from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

_STRESS = str.maketrans("", "", "\u02c8\u02cc'\u0361")
_VOWELS: dict[str, tuple[str, str, bool]] = {
    "i": ("i", "", False),
    "i\u02d0": ("i", "\u30fc", True),
    "\u026a": ("i", "", False),
    "e": ("e", "", False),
    "\u025b": ("e", "", False),
    "\u00e6": ("a", "", False),
    "a": ("a", "", False),
    "\u0251": ("a", "", False),
    "\u0251\u02d0": ("a", "\u30fc", True),
    "\u0250": ("a", "", False),
    "\u028c": ("a", "", False),
    "\u0259": ("a", "", False),
    "\u025c": ("a", "", False),
    "\u025c\u02d0": ("a", "\u30fc", True),
    "\u025a": ("a", "\u30fc", True),
    "\u025d": ("a", "\u30fc", True),
    "\u0252": ("o", "", False),
    "\u0254": ("o", "", False),
    "\u0254\u02d0": ("o", "\u30fc", True),
    "o": ("o", "", False),
    "u": ("u", "", False),
    "u\u02d0": ("u", "\u30fc", True),
    "\u028a": ("u", "", False),
    "e\u026a": ("e", "\u30a4", True),
    "a\u026a": ("a", "\u30a4", True),
    "a\u028a": ("a", "\u30a6", True),
    "\u0254\u026a": ("o", "\u30a4", True),
    "o\u028a": ("o", "\u30a6", True),
    "\u0259\u028a": ("o", "\u30a6", True),
}

_ONSETS: dict[str, tuple[str, str, str, str, str]] = {
    "": ("\u30a2", "\u30a4", "\u30a6", "\u30a8", "\u30aa"),
    "p": ("\u30d1", "\u30d4", "\u30d7", "\u30da", "\u30dd"),
    "b": ("\u30d0", "\u30d3", "\u30d6", "\u30d9", "\u30dc"),
    "t": ("\u30bf", "\u30c6\u30a3", "\u30c8\u30a5", "\u30c6", "\u30c8"),
    "d": ("\u30c0", "\u30c7\u30a3", "\u30c9\u30a5", "\u30c7", "\u30c9"),
    "k": ("\u30ab", "\u30ad", "\u30af", "\u30b1", "\u30b3"),
    "\u0261": ("\u30ac", "\u30ae", "\u30b0", "\u30b2", "\u30b4"),
    "g": ("\u30ac", "\u30ae", "\u30b0", "\u30b2", "\u30b4"),
    "f": ("\u30d5\u30a1", "\u30d5\u30a3", "\u30d5", "\u30d5\u30a7", "\u30d5\u30a9"),
    "v": ("\u30f4\u30a1", "\u30f4\u30a3", "\u30f4", "\u30f4\u30a7", "\u30f4\u30a9"),
    "\u03b8": ("\u30b5", "\u30b7", "\u30b9", "\u30bb", "\u30bd"),
    "\u00f0": ("\u30b6", "\u30b8", "\u30ba", "\u30bc", "\u30be"),
    "s": ("\u30b5", "\u30b7", "\u30b9", "\u30bb", "\u30bd"),
    "z": ("\u30b6", "\u30b8", "\u30ba", "\u30bc", "\u30be"),
    "\u0283": ("\u30b7\u30e3", "\u30b7", "\u30b7\u30e5", "\u30b7\u30a7", "\u30b7\u30e7"),
    "\u0292": ("\u30b8\u30e3", "\u30b8", "\u30b8\u30e5", "\u30b8\u30a7", "\u30b8\u30e7"),
    "t\u0283": ("\u30c1\u30e3", "\u30c1", "\u30c1\u30e5", "\u30c1\u30a7", "\u30c1\u30e7"),
    "d\u0292": ("\u30b8\u30e3", "\u30b8", "\u30b8\u30e5", "\u30b8\u30a7", "\u30b8\u30e7"),
    "m": ("\u30de", "\u30df", "\u30e0", "\u30e1", "\u30e2"),
    "n": ("\u30ca", "\u30cb", "\u30cc", "\u30cd", "\u30ce"),
    "\u014b": ("\u30ca", "\u30cb", "\u30cc", "\u30cd", "\u30ce"),
    "h": ("\u30cf", "\u30d2", "\u30d5", "\u30d8", "\u30db"),
    "l": ("\u30e9", "\u30ea", "\u30eb", "\u30ec", "\u30ed"),
    "\u0279": ("\u30e9", "\u30ea", "\u30eb", "\u30ec", "\u30ed"),
    "r": ("\u30e9", "\u30ea", "\u30eb", "\u30ec", "\u30ed"),
    "w": ("\u30ef", "\u30a6\u30a3", "\u30a6", "\u30a6\u30a7", "\u30a6\u30a9"),
    "j": ("\u30e4", "\u30a4", "\u30e6", "\u30a4\u30a7", "\u30e8"),
}

_PALATAL: dict[str, tuple[str, str, str, str, str]] = {
    "p": ("\u30d4\u30e3", "\u30d4", "\u30d4\u30e5", "\u30d4\u30a7", "\u30d4\u30e7"),
    "b": ("\u30d3\u30e3", "\u30d3", "\u30d3\u30e5", "\u30d3\u30a7", "\u30d3\u30e7"),
    "k": ("\u30ad\u30e3", "\u30ad", "\u30ad\u30e5", "\u30ad\u30a7", "\u30ad\u30e7"),
    "g": ("\u30ae\u30e3", "\u30ae", "\u30ae\u30e5", "\u30ae\u30a7", "\u30ae\u30e7"),
    "\u0261": ("\u30ae\u30e3", "\u30ae", "\u30ae\u30e5", "\u30ae\u30a7", "\u30ae\u30e7"),
    "m": ("\u30df\u30e3", "\u30df", "\u30df\u30e5", "\u30df\u30a7", "\u30df\u30e7"),
    "n": ("\u30cb\u30e3", "\u30cb", "\u30cb\u30e5", "\u30cb\u30a7", "\u30cb\u30e7"),
    "h": ("\u30d2\u30e3", "\u30d2", "\u30d2\u30e5", "\u30d2\u30a7", "\u30d2\u30e7"),
    "l": ("\u30ea\u30e3", "\u30ea", "\u30ea\u30e5", "\u30ea\u30a7", "\u30ea\u30e7"),
    "r": ("\u30ea\u30e3", "\u30ea", "\u30ea\u30e5", "\u30ea\u30a7", "\u30ea\u30e7"),
    "\u0279": ("\u30ea\u30e3", "\u30ea", "\u30ea\u30e5", "\u30ea\u30a7", "\u30ea\u30e7"),
}

_CODAS = {
    "m": "\u30f3",
    "n": "\u30f3",
    "\u014b": "\u30f3",
    "b": "\u30d6",
    "d": "\u30c9",
    "g": "\u30b0",
    "\u0261": "\u30b0",
    "f": "\u30d5",
    "v": "\u30f4",
    "\u03b8": "\u30b9",
    "\u00f0": "\u30ba",
    "s": "\u30b9",
    "z": "\u30ba",
    "\u0283": "\u30b7\u30e5",
    "\u0292": "\u30b8",
    "t\u0283": "\u30c1",
    "d\u0292": "\u30b8",
    "ts": "\u30c3\u30c4",
    "h": "\u30d5",
    "l": "\u30eb",
    "r": "\u30eb",
    "\u0279": "\u30eb",
    "w": "\u30a6",
    "j": "\u30a4",
}

_LETTER_NAMES = {
    "a": "\u30a8\u30fc",
    "b": "\u30d3\u30fc",
    "c": "\u30b7\u30fc",
    "d": "\u30c7\u30a3\u30fc",
    "e": "\u30a4\u30fc",
    "f": "\u30a8\u30d5",
    "g": "\u30b8\u30fc",
    "h": "\u30a8\u30a4\u30c1",
    "i": "\u30a2\u30a4",
    "j": "\u30b8\u30a7\u30a4",
    "k": "\u30b1\u30fc",
    "l": "\u30a8\u30eb",
    "m": "\u30a8\u30e0",
    "n": "\u30a8\u30cc",
    "o": "\u30aa\u30fc",
    "p": "\u30d4\u30fc",
    "q": "\u30ad\u30e5\u30fc",
    "r": "\u30a2\u30fc\u30eb",
    "s": "\u30a8\u30b9",
    "t": "\u30c6\u30a3\u30fc",
    "u": "\u30e6\u30fc",
    "v": "\u30f4\u30a3\u30fc",
    "w": "\u30c0\u30d6\u30ea\u30e5\u30fc",
    "x": "\u30a8\u30c3\u30af\u30b9",
    "y": "\u30ef\u30a4",
    "z": "\u30ba\u30a3\u30fc",
}


class EnglishJapaneseG2P:
    """Convert English pronunciations into the Japanese pySHIRO inventory."""

    def __init__(self) -> None:
        self._backend = EspeakBackend(
            "en-gb",
            preserve_punctuation=False,
            with_stress=False,
        )
        self._separator = Separator(phone=" ", word=None)
        self._cache: dict[str, str] = {}

    def katakana(self, word: str) -> str:
        normalized = word.casefold().replace("\u2019", "'")
        if normalized in self._cache:
            return self._cache[normalized]
        values = self._backend.phonemize(
            [normalized],
            separator=self._separator,
            strip=True,
            njobs=1,
        )
        phones = _clean_phones(values[0] if values else "")
        reading = _phones_to_katakana(phones)
        if not reading:
            reading = "".join(_LETTER_NAMES.get(char, "") for char in normalized)
        if len(self._cache) >= 4096:
            self._cache.pop(next(iter(self._cache)))
        self._cache[normalized] = reading
        return reading


def _clean_phones(value: str) -> list[str]:
    phones = [
        phone
        for item in value.split()
        for phone in _split_phone_chunk(item.translate(_STRESS))
    ]
    output: list[str] = []
    index = 0
    while index < len(phones):
        if index + 1 < len(phones) and (phones[index], phones[index + 1]) in {
            ("t", "\u0283"),
            ("d", "\u0292"),
            ("t", "s"),
        }:
            output.append(phones[index] + phones[index + 1])
            index += 2
            continue
        output.append(phones[index])
        index += 1
    return output


def _split_phone_chunk(value: str) -> list[str]:
    symbols = sorted(
        {
            *_VOWELS,
            *_ONSETS,
            *_CODAS,
            "t\u0283",
            "d\u0292",
            "ts",
        }
        - {""},
        key=len,
        reverse=True,
    )
    output: list[str] = []
    cursor = 0
    while cursor < len(value):
        symbol = next(
            (item for item in symbols if value.startswith(item, cursor)), None
        )
        if symbol is None:
            cursor += 1
            continue
        output.append(symbol)
        cursor += len(symbol)
    return output


def _phones_to_katakana(phones: list[str]) -> str:
    output: list[str] = []
    previous_extended_vowel = False
    index = 0
    while index < len(phones):
        phone = phones[index]
        vowel = _VOWELS.get(phone)
        if vowel is not None:
            output.append(_render_vowel("", vowel))
            previous_extended_vowel = vowel[2]
            index += 1
            continue
        if (
            index + 2 < len(phones)
            and phones[index + 1] == "j"
            and phone in _PALATAL
            and (vowel := _VOWELS.get(phones[index + 2])) is not None
        ):
            output.append(_render_vowel(phone, vowel, palatal=True))
            previous_extended_vowel = vowel[2]
            index += 3
            continue
        if (
            index + 1 < len(phones)
            and phone in _ONSETS
            and (vowel := _VOWELS.get(phones[index + 1])) is not None
        ):
            output.append(_render_vowel(phone, vowel))
            previous_extended_vowel = vowel[2]
            index += 2
            continue
        if phone == "p":
            output.append("\u30d7" if previous_extended_vowel else "\u30c3\u30d7")
        elif phone == "t":
            output.append("\u30c8" if previous_extended_vowel else "\u30c3\u30c8")
        elif phone == "k":
            output.append("\u30af" if previous_extended_vowel else "\u30c3\u30af")
        else:
            output.append(_CODAS.get(phone, ""))
        previous_extended_vowel = False
        index += 1
    return "".join(output)


def _render_vowel(
    onset: str,
    vowel: tuple[str, str, bool],
    *,
    palatal: bool = False,
) -> str:
    category, suffix, _extended = vowel
    position = "aiueo".index(category)
    table = _PALATAL[onset] if palatal else _ONSETS[onset]
    return table[position] + suffix
