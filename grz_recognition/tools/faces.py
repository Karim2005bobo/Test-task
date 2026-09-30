import os

import cv2

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_detection_yunet_2023mar.onnx")


def detect_faces(img, score=0.6, tile=640):
    H, W = img.shape[:2]
    det = cv2.FaceDetectorYN.create(MODEL, "", (tile, tile), score, 0.3, 5000)
    boxes = []
    step = tile // 2
    for y in range(0, max(H - step, 1), step):
        for x in range(0, max(W - step, 1), step):
            crop = img[y:y + tile, x:x + tile]
            h, w = crop.shape[:2]
            det.setInputSize((w, h))
            _, faces = det.detect(crop)
            if faces is not None:
                for f in faces:
                    boxes.append((int(f[0]) + x, int(f[1]) + y, int(f[2]), int(f[3])))
    return boxes


def blur_faces(img, pad=0.3):
    H, W = img.shape[:2]
    boxes = detect_faces(img)
    for x, y, w, h in boxes:
        x0, y0 = max(0, int(x - w * pad)), max(0, int(y - h * pad))
        x1, y1 = min(W, int(x + w * (1 + pad))), min(H, int(y + h * (1 + pad)))
        roi = img[y0:y1, x0:x1]
        if roi.size:
            k = max(15, (max(x1 - x0, y1 - y0) // 2) | 1)
            img[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
    return len(boxes)
