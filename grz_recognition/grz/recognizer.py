"""Инференс распознавателя (ONNX Runtime) и декодирование CTC.

Декодирование в два шага:
  1. жадное (без ограничений) – используется для «прочих» знаков;
  2. Витерби по решётке CTC, ограниченной маской номера РФ
     (L DDD LL DD[D]) – для типов 1/1А/1Б. Ограниченный поиск сам исправляет
     типичные путаницы O/0, B/8 и т.п., так как запрещает буквы на местах цифр
     и наоборот, а также неверную первую цифру трёхзначного региона.
Символы с низкой уверенностью заменяются на '#', как в эталонной разметке.
"""
import numpy as np

from grz.plate_format import BLANK, CHAR2IDX, IDX2CHAR, MASK_8, MASK_9, allowed_chars

NEG = -1e9
_ALLOWED = {m: np.array([CHAR2IDX[c] for c in allowed_chars(m)]) for m in "LDR"}


def greedy_decode(probs):
    """probs: [N, T, C] -> список строк."""
    out = []
    for p in probs:
        best = p.argmax(-1)
        s, prev = [], BLANK
        for b in best:
            if b != prev and b != BLANK:
                s.append(IDX2CHAR[int(b)])
            prev = b
        out.append("".join(s))
    return out


def greedy_with_conf(p):
    """p: [T, C] -> (строка, уверенности символов, средний log-prob пути)."""
    best = p.argmax(-1)
    chars, confs, prev = [], [], BLANK
    for t, b in enumerate(best):
        if b != BLANK and b != prev:
            chars.append(IDX2CHAR[int(b)])
            confs.append(float(p[t, b]))
        elif b != BLANK and b == prev:
            confs[-1] = max(confs[-1], float(p[t, b]))
        prev = b
    score = float(np.log(p[np.arange(len(best)), best] + 1e-12).sum())
    return "".join(chars), confs, score


def constrained_decode(p, mask):
    """Лучший путь CTC, порождающий строку, удовлетворяющую маске.

    p: [T, C] вероятности. Возвращает (строка, уверенности символов, log-prob пути).
    Состояния: blank[k] – выпущено k символов, сейчас пробел;
               char[k][j] – сейчас «звучит» k-й символ, равный sets[k][j].
    """
    T = p.shape[0]
    L = len(mask)
    lp = np.log(p + 1e-12)
    sets = [None] + [_ALLOWED[m] for m in mask]
    blank = np.full(L + 1, NEG)
    char = [None] + [np.full(len(sets[k]), NEG) for k in range(1, L + 1)]
    blank[0] = lp[0, BLANK]
    char[1] = lp[0, sets[1]].copy()
    # bp_blank[t, k]: -1 – из blank[k], j>=0 – из char[k][j]
    bp_blank = np.full((T, L + 1), -1, np.int16)
    # bp_char[k][t, j]: -1 – остаться, -2 – из blank[k-1], j'>=0 – из char[k-1][j']
    bp_char = [None] + [np.full((T, len(sets[k])), -1, np.int16) for k in range(1, L + 1)]
    for t in range(1, T):
        nb = np.empty(L + 1)
        nc = [None] * (L + 1)
        lpb = lp[t, BLANK]
        for k in range(L + 1):
            if k >= 1:
                jc = int(char[k].argmax())
                if char[k][jc] > blank[k]:
                    nb[k] = char[k][jc] + lpb
                    bp_blank[t, k] = jc
                else:
                    nb[k] = blank[k] + lpb
            else:
                nb[k] = blank[k] + lpb
            if k == 0:
                continue
            cur = sets[k]
            n = len(cur)
            opts = [char[k], np.full(n, blank[k - 1])]
            codes = [np.full(n, -1), np.full(n, -2)]
            if k >= 2:
                prev, pset = char[k - 1], sets[k - 1]
                order = np.argsort(-prev)
                j1 = order[0]
                j2 = order[1] if len(order) > 1 else order[0]
                same = cur == pset[j1]
                opts.append(np.where(same, prev[j2], prev[j1]))
                codes.append(np.where(same, j2, j1))
            allv = np.vstack(opts)
            arg = allv.argmax(0)
            cols = np.arange(n)
            nc[k] = allv[arg, cols] + lp[t, cur]
            bp_char[k][t] = np.vstack(codes)[arg, cols]
        blank, char = nb, nc
    jL = int(char[L].argmax())
    if char[L][jL] > blank[L]:
        kind, k, j, score = "c", L, jL, float(char[L][jL])
    else:
        kind, k, j, score = "b", L, 0, float(blank[L])
    out = [0] * (L + 1)
    confs = [0.0] * (L + 1)
    for t in range(T - 1, -1, -1):
        if kind == "c":
            c = int(sets[k][j])
            out[k] = c
            confs[k] = max(confs[k], float(p[t, c]))
            code = int(bp_char[k][t, j])
            if code == -2:
                kind, k = "b", k - 1
            elif code >= 0:
                k, j = k - 1, code
        else:
            code = int(bp_blank[t, k])
            if code >= 0:
                kind, j = "c", code
    text = "".join(IDX2CHAR[c] for c in out[1:])
    return text, confs[1:], score


def _compress(p, thr=0.995):
    """Схлопывает серии «чистых» пробелов в один кадр – ускоряет Витерби в 2-3 раза."""
    keep, prev_blank = [], False
    for t in range(p.shape[0]):
        is_blank = p[t, BLANK] > thr
        if not (is_blank and prev_blank):
            keep.append(t)
        prev_blank = is_blank
    return p[keep]


def decode_ru(p):
    """Лучшая из масок 8 и 9 символов для номера РФ."""
    p = _compress(p)
    best = None
    for mask in (MASK_8, MASK_9):
        r = constrained_decode(p, mask)
        if best is None or r[2] > best[2]:
            best = r
    return best


class Recognizer:
    """Обёртка ONNX-модели распознавателя."""

    def __init__(self, path, providers):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=providers)
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, x):
        """x: float32 [N, 3, 48, 224] -> (ctc probs [N, T, C], type probs [N, 5])."""
        ctc, cls = self.sess.run(None, {self.inp: x})
        return ctc, cls
