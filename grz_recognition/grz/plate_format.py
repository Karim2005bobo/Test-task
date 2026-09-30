import re

RU_LETTERS = "ABEKMHOPCTYX"
DIGITS = "0123456789"

OCR_ALPHABET = DIGITS + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK = 0
CHAR2IDX = {c: i + 1 for i, c in enumerate(OCR_ALPHABET)}
IDX2CHAR = {i + 1: c for i, c in enumerate(OCR_ALPHABET)}
NUM_CLASSES = len(OCR_ALPHABET) + 1

PLATE_TYPES = ["type1", "type1a", "type1b", "other"]
TYPE2IDX = {t: i for i, t in enumerate(PLATE_TYPES)}

_L = "[ABEKMHOPCTYX]"
_REG = r"\d{2,3}"
PLATE_RE = {
    "type1": re.compile(rf"^{_L}\d{{3}}{_L}{{2}}{_REG}$"),
    "type1a": re.compile(rf"^{_L}\d{{3}}{_L}{{2}}{_REG}$"),
    "type1b": re.compile(rf"^{_L}{{2}}\d{{3}}{_REG}$"),
}

MASKS = {
    "type1": ("LDDDLLDD", "LDDDLLDDD"),
    "type1a": ("LDDDLLDD", "LDDDLLDDD"),
    "type1b": ("LLDDDDD", "LLDDDDDD"),
}

TO_LETTER = {"0": "O", "8": "B", "4": "A", "7": "T", "3": "E", "6": "B", "1": "T", "2": "Z", "5": "S"}
TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "B": "8", "A": "4", "T": "7", "E": "3",
            "Z": "2", "S": "5", "I": "1", "L": "1", "G": "6", "U": "0"}


def is_valid(num: str, plate_type: str) -> bool:
    return plate_type in PLATE_RE and bool(PLATE_RE[plate_type].match(num))


def allowed_chars(mask_char: str) -> str:
    if mask_char == "L":
        return RU_LETTERS
    return DIGITS


def encode(text: str):
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]
