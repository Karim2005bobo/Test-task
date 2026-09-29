"""Обучение распознавателя (CTC + классификация типа) на подготовленных кропах.

python training/train_recognizer.py --data work/ocr --out work/rec --epochs 12
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grz.plate_format import encode  # noqa: E402
from grz.recognizer import greedy_decode  # noqa: E402
from training.model import PlateNet  # noqa: E402


class CropDataset(Dataset):
    def __init__(self, X, labels, idx, train):
        self.X, self.labels, self.idx, self.train = X, labels, idx, train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        img = np.array(self.X[i])
        text, t = self.labels[i]
        if self.train:
            rng = np.random
            if rng.rand() < 0.5:
                h, w = img.shape[:2]
                M = np.float32([[rng.uniform(0.94, 1.06), rng.uniform(-0.04, 0.04), rng.uniform(-4, 4)],
                                [rng.uniform(-0.03, 0.03), rng.uniform(0.92, 1.08), rng.uniform(-3, 3)]])
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            if rng.rand() < 0.5:
                img = np.clip(img.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25), 0, 255).astype(np.uint8)
            if rng.rand() < 0.1:
                img = 255 - img if t == 4 else img  # инверсия только для негативов
        x = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        return x, torch.tensor(encode(text), dtype=torch.long), t, text


def collate(batch):
    xs, ys, ts, texts = zip(*batch)
    lens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    flat = torch.cat(ys) if lens.sum() > 0 else torch.zeros(0, dtype=torch.long)
    return torch.stack(xs), flat, lens, torch.tensor(ts), list(texts)


def evaluate(model, loader):
    model.eval()
    n = ok = tok = 0
    per = {}
    with torch.no_grad():
        for x, _, _, t, texts in loader:
            lc, lt = model(x)
            pred = greedy_decode(lc.softmax(-1).permute(1, 0, 2).numpy())
            pt = lt.argmax(1).numpy()
            for p, gt, tt, ptt in zip(pred, texts, t.numpy(), pt):
                n += 1
                tok += int(tt == ptt)
                if tt < 4:
                    s = per.setdefault(int(tt), [0, 0])
                    s[0] += 1
                    s[1] += int(p == gt)
                    ok += int(p == gt)
    model.train()
    return tok / max(n, 1), {k: v[1] / v[0] for k, v in sorted(per.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=float, default=0.75)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-steps", type=int, default=0, help="для отладки")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(args.out, exist_ok=True)

    X = np.load(os.path.join(args.data, "X.npy"), mmap_mode="r")
    labels = []
    for line in open(os.path.join(args.data, "y.tsv")):
        text, t, _syn = line.rstrip("\n").split("\t")
        labels.append((text, int(t)))
    N = len(labels)
    perm = np.random.RandomState(0).permutation(N)
    nval = min(3000, N // 30)
    val_idx, tr_idx = perm[:nval], perm[nval:]
    tr = DataLoader(CropDataset(X, labels, tr_idx, True), batch_size=args.bs, shuffle=True,
                    num_workers=2, collate_fn=collate, drop_last=True, persistent_workers=True)
    va = DataLoader(CropDataset(X, labels, val_idx, False), batch_size=256, collate_fn=collate, num_workers=1)

    model = PlateNet(args.width)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu")["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    best, step = -1, 0
    for ep in range(args.epochs):
        t0 = time.time()
        for x, y, ylen, t, _ in tr:
            lc, lt = model(x)
            logp = lc.log_softmax(-1)
            T = torch.full((x.shape[0],), logp.shape[0], dtype=torch.long)
            loss = ctc(logp, y, T, ylen) + 0.5 * F.cross_entropy(lt, t, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f} {(time.time() - t0) / (step - ep * len(tr)):.3f}s/it", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        type_acc, seq_acc = evaluate(model, va)
        score = np.mean(list(seq_acc.values())) + type_acc
        print(f"== epoch {ep}: type_acc {type_acc:.4f} seq_acc {seq_acc} ({time.time() - t0:.0f}s)", flush=True)
        ck = {"model": model.state_dict(), "width": args.width, "epoch": ep, "type_acc": type_acc, "seq_acc": seq_acc}
        torch.save(ck, os.path.join(args.out, "last.pt"))
        if score > best:
            best = score
            torch.save(ck, os.path.join(args.out, "best.pt"))
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
