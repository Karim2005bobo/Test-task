import torch
import torch.nn as nn

from grz.plate_format import NUM_CLASSES

N_TYPES = 5  


def cbr(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class PlateNet(nn.Module):
    def __init__(self, width=1.0):
        super().__init__()
        c = [int(v * width) for v in (32, 64, 128, 192, 256)]
        self.stem = nn.Sequential(
            cbr(3, c[0]), nn.MaxPool2d(2),            
            cbr(c[0], c[1]), nn.MaxPool2d(2),         
            cbr(c[1], c[2]), cbr(c[2], c[2]), nn.MaxPool2d((2, 1)),
            cbr(c[2], c[3]), cbr(c[3], c[3]), nn.MaxPool2d((2, 1)), 
        )
        self.collapse = cbr(c[3], c[4], k=(3, 1), p=0)  
        self.rnn = nn.LSTM(c[4], 128, num_layers=1, bidirectional=True, batch_first=False)
        self.ctc = nn.Linear(256, NUM_CLASSES)
        self.cls = nn.Sequential(nn.Linear(c[3] * 2, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),
                                 nn.Linear(128, N_TYPES))

    def forward(self, x):
        f = self.stem(x)                                
        g = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        s = self.collapse(f).squeeze(2).permute(2, 0, 1)  # T, N, C
        s, _ = self.rnn(s)
        return self.ctc(s), self.cls(g)
