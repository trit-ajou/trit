from enum import Enum
from typing import Literal


class Lang(Enum):
    EN = 1  # english
    KR = 2  # korean
    JP = 3  # japanese
    SP = 4  # special chars


UNICODE_RANGES = {
    Lang.EN: [(0x0020, 0x007E)],  # Basic Latin (English + Punctuation)
    Lang.KR: [(0xAC00, 0xD7A3)],  # Hangul Syllables
    Lang.JP: [
        (0x3040, 0x309F),  # Hiragana
        (0x30A0, 0x30FF),  # Katakana
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs (Kanji) - Common
        (0xFF00, 0xFFEF),  # Halfwidth and Fullwidth Forms (some punctuation)
    ],
    Lang.SP: [
        (0x0021, 0x002F),
        (0x003A, 0x0040),
        (0x005B, 0x0060),
        (0x007B, 0x007E),  # Punctuation
        (0x2000, 0x206F),  # General Punctuation
        (0x3000, 0x303F),  # CJK Symbols and Punctuation
    ],
}


class BBox(tuple):
    def __new__(cls, x1: int, y1: int, x2: int, y2: int):
        return super().__new__(cls, (x1, y1, x2, y2))

    @property
    def x1(self) -> int:
        return self[0]

    @property
    def y1(self) -> int:
        return self[1]

    @property
    def x2(self) -> int:
        return self[2]

    @property
    def y2(self) -> int:
        return self[3]

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def slice(
        self,
    ) -> tuple[
        slice[None],
        slice[int, int, Literal[1]],
        slice[int, int, Literal[1]],
    ]:
        return (slice(None), slice(self.x1, self.x2), slice(self.y1, self.y2))

    def expand(self, margin: int, img_size: tuple[int, int]) -> "BBox":
        w, h = img_size
        x1 = max(0, self.x1 - margin)
        y1 = max(0, self.y1 - margin)
        x2 = min(w, self.x2 + margin)
        y2 = min(h, self.y2 + margin)
        return BBox(x1, y1, x2, y2)

    def intersects(self, other: "BBox") -> bool:
        return not (
            self.x2 < other.x1
            or self.x1 > other.x2
            or self.y2 < other.y1
            or self.y1 > other.y2
        )

    def union(self, other: "BBox") -> "BBox":
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        x2 = max(self.x2, other.x2)
        y2 = max(self.y2, other.y2)
        return BBox(x1, y1, x2, y2)
