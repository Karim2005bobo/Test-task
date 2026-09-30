import cv2
import numpy as np


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, left, top


class Detection:
    __slots__ = ("box", "score", "layout", "quad", "kpt_conf")

    def __init__(self, box, score, layout, quad, kpt_conf):
        self.box, self.score, self.layout, self.quad, self.kpt_conf = box, score, layout, quad, kpt_conf


class PlateDetector:
    def __init__(self, path, providers, imgsz=640, conf=0.25, iou=0.5):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.imgsz = shape[2] if isinstance(shape[2], int) else imgsz
        self.conf, self.iou = conf, iou

    def __call__(self, img):
        lb, r, dx, dy = letterbox(img, self.imgsz)
        x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.inp: np.ascontiguousarray(x)})[0][0].T  # [A, 4+2+12]
        cls_scores = out[:, 4:6]
        scores = cls_scores.max(1)
        keep = scores > self.conf
        out, scores, layout = out[keep], scores[keep], cls_scores[keep].argmax(1)
        if not len(out):
            return []
        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, w, h], 1)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf, self.iou)
        dets = []
        H, W = img.shape[:2]
        for i in np.array(idx).reshape(-1):
            b = boxes[i].copy()
            b[0] = (b[0] - dx) / r
            b[1] = (b[1] - dy) / r
            b[2:] /= r
            k = out[i, 6:].reshape(4, 3)
            quad = np.stack([(k[:, 0] - dx) / r, (k[:, 1] - dy) / r], 1)
            quad[:, 0] = np.clip(quad[:, 0], 0, W - 1)
            quad[:, 1] = np.clip(quad[:, 1], 0, H - 1)
            dets.append(Detection(b, float(scores[i]), int(layout[i]), quad.astype(np.float32),
                                  float(k[:, 2].min())))
        return dets


class VehicleDetector:

    CLASSES = (2, 3, 5, 7)

    def __init__(self, path, providers, conf=0.25):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.imgsz = self.sess.get_inputs()[0].shape[2]
        self.conf = conf

    def __call__(self, img):
        lb, r, dx, dy = letterbox(img, self.imgsz)
        x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.inp: np.ascontiguousarray(x)})[0][0].T  # [A, 84]
        sc = out[:, 4:][:, list(self.CLASSES)].max(1)
        out = out[sc > self.conf]
        boxes = []
        for cx, cy, w, h in out[:, :4]:
            boxes.append(((cx - w / 2 - dx) / r, (cy - h / 2 - dy) / r, (cx + w / 2 - dx) / r, (cy + h / 2 - dy) / r))
        return boxes
