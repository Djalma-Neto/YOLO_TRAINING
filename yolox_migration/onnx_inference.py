"""CPU/GPU-capable inference for standard decoded YOLOX ONNX outputs."""
from __future__ import annotations

import numpy as np
from PIL import Image


def preprocess(image, size):
    image = image.convert("RGB")
    w, h = image.size
    ratio = min(size[0] / h, size[1] / w)
    resized = image.resize((int(w * ratio), int(h * ratio)), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size[1], size[0]), (114, 114, 114))
    canvas.paste(resized, (0, 0))
    tensor = np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1)[None].astype(np.float32)
    return tensor, ratio


def _nms(boxes, scores, threshold):
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    order = scores.argsort()[::-1]; keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-12)
        order = order[np.where(iou <= threshold)[0] + 1]
    return keep


def postprocess(output, ratio, original_shape, threshold=.25, nms_threshold=.45, num_classes=None):
    """Expected official YOLOX export output [1,N,5+C], decoded xywh,obj,classes."""
    pred = np.asarray(output)
    if pred.ndim == 3: pred = pred[0]
    if pred.ndim != 2 or pred.shape[1] < 6: raise ValueError("Expected YOLOX output shaped [1,N,5+C]")
    if num_classes is not None and pred.shape[1] != 5 + num_classes:
        raise ValueError("ONNX output class count differs from configured classes")
    boxes = pred[:, :4].copy()
    boxes[:, 0] = pred[:, 0] - pred[:, 2] / 2; boxes[:, 1] = pred[:, 1] - pred[:, 3] / 2
    boxes[:, 2] = pred[:, 0] + pred[:, 2] / 2; boxes[:, 3] = pred[:, 1] + pred[:, 3] / 2
    class_scores = pred[:, 5:] * pred[:, 4:5]
    results = []
    h, w = original_shape[:2]
    for cls in range(class_scores.shape[1]):
        scores = class_scores[:, cls]; indices = np.flatnonzero(scores >= threshold)
        if not len(indices): continue
        for idx in _nms(boxes[indices], scores[indices], nms_threshold):
            box = boxes[indices[idx]] / ratio
            box[[0, 2]] = np.clip(box[[0, 2]], 0, w); box[[1, 3]] = np.clip(box[[1, 3]], 0, h)
            results.append((cls, float(scores[indices[idx]]), [float(v) for v in box]))
    return sorted(results, key=lambda x: x[1], reverse=True)


def predict(model_path, image_path, classes, threshold=.25, providers=None):
    import onnxruntime as ort
    try: image = Image.open(image_path).convert("RGB")
    except OSError as exc: raise FileNotFoundError(image_path) from exc
    session = ort.InferenceSession(str(model_path), providers=providers or ["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    height = int(shape[2]) if isinstance(shape[2], int) else 640
    width = int(shape[3]) if isinstance(shape[3], int) else 640
    tensor, ratio = preprocess(image, (height, width))
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    return postprocess(output, ratio, (image.height, image.width, 3), threshold, num_classes=len(classes))
