"""Правила формата ГРЗ РФ (типы 1, 1А, 1Б) и алфавит распознавателя.

Модуль общий для генератора, обучения и инференса, чтобы алфавит и маска
номера были описаны ровно в одном месте.
"""
import re

# Буквы, совпадающие по начертанию в кириллице и латинице (ГОСТ Р 50577-2018).
RU_LETTERS = "ABEKMHOPCTYX"
DIGITS = "0123456789"

# Алфавит OCR: цифры + полная латиница (нужна, чтобы читать «прочие» знаки:
# дипломатические D/CD/T, иностранные номера). Индекс 0 зарезервирован под CTC blank.
OCR_ALPHABET = DIGITS + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK = 0
CHAR2IDX = {c: i + 1 for i, c in enumerate(OCR_ALPHABET)}
IDX2CHAR = {i + 1: c for i, c in enumerate(OCR_ALPHABET)}
NUM_CLASSES = len(OCR_ALPHABET) + 1

PLATE_TYPES = ["type1", "type1a", "type1b", "other"]
TYPE2IDX = {t: i for i, t in enumerate(PLATE_TYPES)}

# Маски (ГОСТ Р 50577-2018, формат организаторов):
#   type1, type1a: Л ЦЦЦ ЛЛ + регион (A123BC77, A123BC777)
#   type1b:        ЛЛ ЦЦЦ + регион   (AB12377, AB123777) – такси/пассажирские
# Регион: 2 цифры или 3 цифры, начинающиеся с 1/2/7.
_L = "[ABEKMHOPCTYX]"
_REG = r"(\d{2}|[127]\d{2})"
PLATE_RE = {
    "type1": re.compile(rf"^{_L}\d{{3}}{_L}{{2}}{_REG}$"),
    "type1a": re.compile(rf"^{_L}\d{{3}}{_L}{{2}}{_REG}$"),
    "type1b": re.compile(rf"^{_L}{{2}}\d{{3}}{_REG}$"),
}

# Маски по позициям: 'L' – буква серии, 'D' – цифра, 'R' – первая цифра
# трёхзначного кода региона (только 1, 2, 7).
MASKS = {
    "type1": ("LDDDLLDD", "LDDDLLRDD"),
    "type1a": ("LDDDLLDD", "LDDDLLRDD"),
    "type1b": ("LLDDDDD", "LLDDDRDD"),
}

# Визуально похожие символы: используются при восстановлении номера по маске.
TO_LETTER = {"0": "O", "8": "B", "4": "A", "7": "T", "3": "E", "6": "B", "1": "T", "2": "Z", "5": "S"}
TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "B": "8", "A": "4", "T": "7", "E": "3",
            "Z": "2", "S": "5", "I": "1", "L": "1", "G": "6", "U": "0"}


def is_valid(num: str, plate_type: str) -> bool:
    """Номер полностью соответствует маске своего типа."""
    return plate_type in PLATE_RE and bool(PLATE_RE[plate_type].match(num))


def allowed_chars(mask_char: str) -> str:
    if mask_char == "L":
        return RU_LETTERS
    if mask_char == "R":
        return "127"
    return DIGITS


def encode(text: str):
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]
