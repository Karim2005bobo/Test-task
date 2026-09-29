"""Многозадачный распознаватель знака: CTC-чтение символов + классификация типа.

Вход: 3x48x224 (выпрямленная «лента» знака, см. grz/rectify.py).
Выходы:
  logits_ctc  [T=56, N, C]  – по кадрам ленты, C = len(OCR_ALPHABET) + 1 (blank)
  logits_type [N, 5]        – type1 / type1a / type1b / other / none (не знак)
"""
import torch
import torch.nn as nn

from grz.plate_format import NUM_CLASSES

N_TYPES = 5  # 4 типа из задания + «не знак» для отсева ложных детекций


def cbr(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class PlateNet(nn.Module):
    def __init__(self, width=1.0):
        super().__init__()
        c = [int(v * width) for v in (32, 64, 128, 192, 256)]
        self.stem = nn.Sequential(
            cbr(3, c[0]), nn.MaxPool2d(2),                 # 24x112
            cbr(c[0], c[1]), nn.MaxPool2d(2),              # 12x56
            cbr(c[1], c[2]), cbr(c[2], c[2]), nn.MaxPool2d((2, 1)),   # 6x56
            cbr(c[2], c[3]), cbr(c[3], c[3]), nn.MaxPool2d((2, 1)),   # 3x56
        )
        self.collapse = cbr(c[3], c[4], k=(3, 1), p=0)   # 1x56
        self.rnn = nn.LSTM(c[4], 128, num_layers=1, bidirectional=True, batch_first=False)
        self.ctc = nn.Linear(256, NUM_CLASSES)
        self.cls = nn.Sequential(nn.Linear(c[3] * 2, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),
                                 nn.Linear(128, N_TYPES))

    def forward(self, x):
        f = self.stem(x)                                   # N, C, 3, 56
        g = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        s = self.collapse(f).squeeze(2).permute(2, 0, 1)  # T, N, C
        s, _ = self.rnn(s)
        return self.ctc(s), self.cls(g)
