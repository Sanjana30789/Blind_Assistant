# import cv2
# import numpy as np
# import onnxruntime as ort
# import threading
# import os
# from .currency_logic import process_predictions
# from utils.logger import logger

# # ── config ────────────────────────────────────────────────────────────────────
# MODEL_PATH   = os.path.join(os.path.dirname(__file__), "best.onnx")
# CONFIDENCE   = 0.5
# IOU_THRESHOLD = 0.4
# MAX_FPS      = 20

# # ✅ Your dataset labels
# CLASS_NAMES = [
#     "100_rupees",
#     "10_rupees",
#     "2000_rupees",
#     "200_rupees",
#     "20_rupees",
#     "500_rupees",
#     "50_rupees"
# ]

# # ── state ─────────────────────────────────────────────────────────────────────
# _thread   = None
# _stop_evt = threading.Event()


# # ── helpers ───────────────────────────────────────────────────────────────────
# def _letterbox(img, new_shape=(640, 640)):
#     """Resize + pad to square while keeping aspect ratio."""
#     h, w = img.shape[:2]
#     scale = min(new_shape[0] / h, new_shape[1] / w)
#     nh, nw = int(h * scale), int(w * scale)
#     img = cv2.resize(img, (nw, nh))
#     top    = (new_shape[0] - nh) // 2
#     bottom = new_shape[0] - nh - top
#     left   = (new_shape[1] - nw) // 2
#     right  = new_shape[1] - nw - left
#     img = cv2.copyMakeBorder(img, top, bottom, left, right,
#                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
#     return img, scale, (left, top)


# def _postprocess(outputs, orig_shape, scale, pad, conf_thresh, iou_thresh):
#     """Decode Ultralytics ONNX with built-in NMS"""

#     preds = outputs[0][0]   # (300, 6)

#     boxes = preds[:, :4]
#     confidences = preds[:, 4]
#     class_ids = preds[:, 5].astype(int)

#     # Confidence filter
#     mask = confidences >= conf_thresh
#     boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

#     if len(boxes) == 0:
#         return {"predictions": [], "image": {"width": orig_shape[1], "height": orig_shape[0]}}

#     # Undo letterbox
#     pad_x, pad_y = pad
#     x1 = (boxes[:, 0] - pad_x) / scale
#     y1 = (boxes[:, 1] - pad_y) / scale
#     x2 = (boxes[:, 2] - pad_x) / scale
#     y2 = (boxes[:, 3] - pad_y) / scale

#     predictions = []
#     for i in range(len(boxes)):
#         cid = int(class_ids[i])

#         # Safe mapping
#         label = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"

#         predictions.append({
#             "x": float((x1[i] + x2[i]) / 2),
#             "y": float((y1[i] + y2[i]) / 2),
#             "width": float(x2[i] - x1[i]),
#             "height": float(y2[i] - y1[i]),
#             "confidence": float(confidences[i]),
#             "class_id": cid,
#             "class": label,
#         })

#     return {"predictions": predictions,
#             "image": {"width": orig_shape[1], "height": orig_shape[0]}}

# # ── main loop ─────────────────────────────────────────────────────────────────
# def _run(stop_evt: threading.Event):
#     logger.info(f"Loading ONNX model from {MODEL_PATH}")

#     # GPU if available
#     providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
#     session = ort.InferenceSession(MODEL_PATH, providers=providers)

#     input_name  = session.get_inputs()[0].name
#     input_shape = session.get_inputs()[0].shape
#     h_in = input_shape[2] if isinstance(input_shape[2], int) else 640
#     w_in = input_shape[3] if isinstance(input_shape[3], int) else 640

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         logger.error("Cannot open camera")
#         return

#     delay = 1.0 / MAX_FPS
#     logger.info("Currency pipeline started ✓ (local ONNX)")

#     while not stop_evt.is_set():
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning("Empty frame — skipping")
#             continue

#         orig_shape = frame.shape
#         img, scale, pad = _letterbox(frame, (h_in, w_in))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = img.astype(np.float32) / 255.0
#         img = np.transpose(img, (2, 0, 1))[np.newaxis]

#         outputs = session.run(None, {input_name: img})
#         result  = _postprocess(outputs, orig_shape, scale, pad,
#                                 CONFIDENCE, IOU_THRESHOLD)
#         process_predictions(result)

#         stop_evt.wait(delay)

#     cap.release()
#     logger.info("Camera released ✓")


# # ── public API ────────────────────────────────────────────────────────────────
# def start_currency_detection():
#     global _thread, _stop_evt

#     if _thread is not None and _thread.is_alive():
#         logger.warning("Currency detection already running — ignoring start call")
#         return

#     _stop_evt.clear()
#     _thread = threading.Thread(target=_run, args=(_stop_evt,), daemon=True)
#     _thread.start()


# def stop_currency_detection():
#     global _thread, _stop_evt

#     if _thread is None or not _thread.is_alive():
#         logger.warning("No active currency pipeline to stop")
#         return

#     _stop_evt.set()
#     _thread.join(timeout=5)
#     _thread = None
#     logger.info("Currency pipeline stopped ✓")



import cv2
import numpy as np
import onnxruntime as ort
import threading
import os
from .currency_logic import process_predictions
from utils.logger import logger

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "best.onnx")
CONFIDENCE   = 0.5
IOU_THRESHOLD = 0.4
MAX_FPS      = 20

# ✅ Your dataset labels
CLASS_NAMES = [
    "100_rupees",
    "10_rupees",
    "2000_rupees",
    "200_rupees",
    "20_rupees",
    "500_rupees",
    "50_rupees"
]

# ── state ─────────────────────────────────────────────────────────────────────
_thread   = None
_stop_evt = threading.Event()


# ── helpers ───────────────────────────────────────────────────────────────────
def _letterbox(img, new_shape=(640, 640)):
    """Resize + pad to square while keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(h * scale), int(w * scale)

    img = cv2.resize(img, (nw, nh))

    top    = (new_shape[0] - nh) // 2
    bottom = new_shape[0] - nh - top
    left   = (new_shape[1] - nw) // 2
    right  = new_shape[1] - nw - left

    img = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    return img, scale, (left, top)


def _postprocess(outputs, orig_shape, scale, pad, conf_thresh, iou_thresh):
    """Decode Ultralytics ONNX with built-in NMS"""

    preds = outputs[0][0]   # (300, 6)

    boxes = preds[:, :4]
    confidences = preds[:, 4]
    class_ids = preds[:, 5].astype(int)

    # Confidence filter
    mask = confidences >= conf_thresh
    boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

    if len(boxes) == 0:
        return {
            "predictions": [],
            "image": {"width": orig_shape[1], "height": orig_shape[0]}
        }

    # Undo letterbox
    pad_x, pad_y = pad

    x1 = (boxes[:, 0] - pad_x) / scale
    y1 = (boxes[:, 1] - pad_y) / scale
    x2 = (boxes[:, 2] - pad_x) / scale
    y2 = (boxes[:, 3] - pad_y) / scale

    predictions = []

    for i in range(len(boxes)):

        cid = int(class_ids[i])

        label = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"

        predictions.append({
            "x": float((x1[i] + x2[i]) / 2),
            "y": float((y1[i] + y2[i]) / 2),
            "width": float(x2[i] - x1[i]),
            "height": float(y2[i] - y1[i]),
            "confidence": float(confidences[i]),
            "class_id": cid,
            "class": label,
        })

    return {
        "predictions": predictions,
        "image": {"width": orig_shape[1], "height": orig_shape[0]}
    }


# ── main loop ─────────────────────────────────────────────────────────────────
def _run(stop_evt: threading.Event):

    logger.info(f"Loading ONNX model from {MODEL_PATH}")

    # ✅ Check model exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        return

    # ✅ Use available providers automatically
    providers = ort.get_available_providers()
    session = ort.InferenceSession(MODEL_PATH, providers=providers)

    input_name  = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    h_in = input_shape[2] if isinstance(input_shape[2], int) else 640
    w_in = input_shape[3] if isinstance(input_shape[3], int) else 640

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Cannot open camera")
        return

    delay = 1.0 / MAX_FPS

    logger.info("Currency pipeline started ✓ (local ONNX)")

    while not stop_evt.is_set():

        ret, frame = cap.read()

        if not ret:
            logger.warning("Empty frame — skipping")
            stop_evt.wait(0.05)   # prevent CPU spinning
            continue

        orig_shape = frame.shape

        img, scale, pad = _letterbox(frame, (h_in, w_in))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32)
        img /= 255.0

        img = np.transpose(img, (2, 0, 1))[np.newaxis]

        outputs = session.run(None, {input_name: img})

        result = _postprocess(
            outputs,
            orig_shape,
            scale,
            pad,
            CONFIDENCE,
            IOU_THRESHOLD
        )

        process_predictions(result)

        stop_evt.wait(delay)

    cap.release()

    logger.info("Camera released ✓")


# ── public API ────────────────────────────────────────────────────────────────
def start_currency_detection():

    global _thread, _stop_evt

    if _thread is not None and _thread.is_alive():
        logger.warning("Currency detection already running — ignoring start call")
        return

    _stop_evt.clear()

    _thread = threading.Thread(
        target=_run,
        args=(_stop_evt,),
        daemon=True
    )

    _thread.start()


def stop_currency_detection():

    global _thread, _stop_evt

    if _thread is None or not _thread.is_alive():
        logger.warning("No active currency pipeline to stop")
        return

    _stop_evt.set()

    _thread.join(timeout=5)

    _thread = None

    logger.info("Currency pipeline stopped ✓")