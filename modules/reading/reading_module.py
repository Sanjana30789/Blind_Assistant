# modules/reading/reading_module.py

import base64
import cv2
import numpy as np
import time
from modules.scene.vlm_client import VLMClient
from utils.logger import logger
from utils.image_utils import frame_to_base64, resize_frame
from config import GROQ_API_KEY, VLM_MODEL, CAMERA_INDEX


class ReadingModule:

    def __init__(self):
        self.vlm = VLMClient()

    def _sharpness_score(self, b64_image: str) -> float:
        try:
            img_bytes = base64.b64decode(b64_image)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            return cv2.Laplacian(img, cv2.CV_64F).var()
        except Exception as e:
            logger.warning(f"Sharpness calculation failed: {e}")
            return 0.0

    def _capture_frames(self, count: int = 3) -> list:
        """Open camera ONCE and capture all frames — much faster."""
        frames = []
        cam = cv2.VideoCapture(CAMERA_INDEX)

        if not cam.isOpened():
            cam.release()
            raise RuntimeError("Camera not found.")

        # Warmup once
        time.sleep(0.5)
        for _ in range(3):
            cam.read()  # discard first few frames

        for i in range(count):
            ret, frame = cam.read()
            if ret and frame is not None:
                frame = resize_frame(frame, max_width=1920)
                b64 = frame_to_base64(frame, quality=95)
                frames.append(b64)
                logger.debug(f"Frame {i + 1}/{count} captured ✓")
            time.sleep(0.2)  # small gap between frames

        cam.release()
        return frames

    def _pick_sharpest(self, frames: list) -> str:
        scores = [self._sharpness_score(f) for f in frames]
        best_index = scores.index(max(scores))
        logger.debug(f"Sharpness scores: {[round(s, 1) for s in scores]} — using frame {best_index}")
        return frames[best_index]

    def run(self) -> str:
        logger.info("ReadingModule.run() | capturing 3 frames")

        # ── Step 1 — Capture all 3 frames with one camera open ──
        try:
            frames = self._capture_frames(3)
        except RuntimeError as e:
            logger.error(f"Camera error: {e}")
            return "I could not access the camera. Please check it is connected."

        if not frames:
            return "I could not capture any frames from the camera."

        # ── Step 2 — Pick sharpest frame ──
        best_frame = self._pick_sharpest(frames)
        logger.info("Best frame selected for reading ✓")

        # ── Step 3 — Reading prompt ──
        reading_prompt = """
You are a reading assistant for visually impaired users.

Read ALL visible text in this image completely. Do not skip or truncate anything.

Instructions:
- Read every single word exactly as written
- Read from top to bottom, left to right
- For medicine labels: name, dosage, instructions, warnings
- For receipts: every item, price, and total
- For documents: full text top to bottom
- For screens/phones: read all visible text
- Add brief context first e.g. "This is a medicine label" or "This is a receipt"
- If no text visible: say "I could not find any text. Please hold the document closer."
- Do NOT summarize — read the COMPLETE text

Read now:
"""

        # ── Step 4 — Call Groq Vision with higher token limit ──
        logger.info("Sending frame to Groq Vision...")

        try:
            from groq import Groq
            from config import GROQ_API_KEY
            client = Groq(api_key=GROQ_API_KEY)

            response = client.chat.completions.create(
                model=VLM_MODEL,
                max_tokens=2048,  # higher limit for full text reading
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{best_frame}"
                                }
                            },
                            {
                                "type": "text",
                                "text": reading_prompt
                            }
                        ]
                    }
                ]
            )
            result = response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Groq Vision failed: {e}")
            return "I could not read the text. Please try again."

        if not result:
            return "I could not read any text from the image. Please try again."

        logger.info(f"Reading result: {result[:100]}...")
        return result.strip()