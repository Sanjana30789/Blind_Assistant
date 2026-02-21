# modules/scene/camera.py — Camera frame capture using OpenCV.
# Used by all three modules (scene, reading, currency).

import cv2
import time
from utils.logger import logger
from utils.image_utils import frame_to_base64, resize_frame
from config import CAMERA_INDEX, CAMERA_WARMUP_MS


def capture_frame_as_base64() -> str:
    """
    Open camera → wait for warmup → capture one frame → encode to base64.

    Returns:
        base64 JPEG string ready for GPT-4o Vision API.

    Raises:
        RuntimeError: if camera cannot be opened or frame capture fails.
    """
    logger.debug(f"Opening camera (index {CAMERA_INDEX})...")
    cam = cv2.VideoCapture(CAMERA_INDEX)

    if not cam.isOpened():
        cam.release()
        raise RuntimeError(
            "Camera not found. Please check your camera is connected and not used by another app."
        )

    # Warm up — first frames from webcams are often dark or blurry
    time.sleep(CAMERA_WARMUP_MS / 1000.0)

    # Discard a few frames to let auto-exposure settle
    for _ in range(3):
        cam.read()

    ret, frame = cam.read()
    cam.release()

    if not ret or frame is None:
        raise RuntimeError("Camera opened but could not capture a frame. Please try again.")

    # Resize if too large (keeps API cost low, speeds up upload)
    frame = resize_frame(frame, max_width=1024)
    b64   = frame_to_base64(frame, quality=85)

    logger.debug("Frame captured successfully ✓")
    return b64