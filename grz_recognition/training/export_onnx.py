"""Экспорт обученных моделей в ONNX (каталог weights/).

python training/export_onnx.py --rec work/rec/best.pt --det work/det_runs/yolo11n_pose_grz/weights/best.pt \
    --vehicle work/dl/yolo11n.pt
"""
import argparse
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import PlateNet  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")


class Exported(torch.nn.Module):
    """Возвращает вероятности: ctc [N, T, C] и тип [N, 5]."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        ctc, cls = self.m(x)
        return ctc.softmax(-1).permute(1, 0, 2), cls.softmax(-1)


def export_rec(ckpt, out):
    ck = torch.load(ckpt, map_location="cpu")
    m = PlateNet(ck.get("width", 1.0))
    m.load_state_dict(ck["model"])
    m.eval()
    x = torch.zeros(1, 3, 48, 224)
    torch.onnx.export(Exported(m), x, out, input_names=["x"], output_names=["ctc", "cls"],
                      dynamic_axes={"x": {0: "n"}, "ctc": {0: "n"}, "cls": {0: "n"}}, opset_version=17,
                      dynamo=False)
    print("recognizer ->", out, "epoch", ck.get("epoch"), "val", ck.get("seq_acc"), ck.get("type_acc"))


def export_yolo(pt, out, imgsz):
    from ultralytics import YOLO
    path = YOLO(pt).export(format="onnx", imgsz=imgsz, dynamic=False, simplify=False, opset=17)
    shutil.copy(path, out)
    print("yolo ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec")
    ap.add_argument("--det")
    ap.add_argument("--vehicle")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.rec:
        export_rec(args.rec, os.path.join(OUT, "recognizer.onnx"))
    if args.det:
        export_yolo(args.det, os.path.join(OUT, "detector.onnx"), args.imgsz)
    if args.vehicle:
        export_yolo(args.vehicle, os.path.join(OUT, "vehicle.onnx"), args.imgsz)


if __name__ == "__main__":
    main()
