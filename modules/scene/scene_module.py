# modules/scene/scene_module.py
# Scene tool → returns structured awareness (NOT narration)

import json
import re
from modules.scene.camera import capture_frame_as_base64
from modules.scene.vlm_client import VLMClient
from utils.logger import logger


class SceneModule:

    def __init__(self):
        self.vlm = VLMClient()

    def _parse_scene_json(self, raw: str) -> dict:
        """Robustly extract JSON from VLM output, handling markdown fences."""
        # Strip markdown fences
        text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        # Find JSON block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())

        raise ValueError("No JSON found in VLM output")

    def run(self) -> str:
        """
        Scene perception tool.
        Returns a spoken string describing the scene.
        """
        logger.info("SceneModule.run() | capturing multiple frames")

        # ── Step 1 — Capture frames ──
        frames = []
        for i in range(3):
            try:
                frames.append(capture_frame_as_base64())
            except RuntimeError as e:
                logger.error(f"Camera failed on frame {i}: {e}")
                return "I could not access the camera."

        # ── Step 2 — Perception prompt ──
        perception_prompt = """
Analyze the scene and return structured awareness.

Focus ONLY on:
- objects close to the camera
- objects in hand
- possible obstacles ahead
- general environment context

If unsure about objects, return empty lists.

Respond strictly in this JSON format with no extra text:
{"near": [], "in_hand": [], "obstacles": [], "context": "", "confidence": 0.0}
"""

        # ── Step 3 — Call VLM ──
        raw_output = self.vlm.describe(frames[0], perception_prompt)
        logger.debug(f"Raw perception output: {raw_output[:200]}")

        # ── Step 4 — Parse JSON ──
        try:
            scene_data = self._parse_scene_json(raw_output)

            scene_data.setdefault("near", [])
            scene_data.setdefault("in_hand", [])
            scene_data.setdefault("obstacles", [])
            scene_data.setdefault("context", "")
            scene_data.setdefault("confidence", 0.5)

        except Exception as e:
            logger.warning(f"Failed to parse scene JSON: {e} — using fallback")
            scene_data = {
                "near": [],
                "in_hand": [],
                "obstacles": [],
                "context": raw_output.strip(),
                "confidence": 0.3
            }

        logger.info(f"Scene awareness: {scene_data}")

        # ── Step 5 — Convert dict → spoken string ──
        return self._to_speech(scene_data)

    def _to_speech(self, data: dict) -> str:
        """Convert structured scene dict into a natural spoken sentence."""
        parts = []

        context = data.get("context", "").strip()
        if context:
            parts.append(context)

        near = [str(o) for o in data.get("near", []) if o]
        if near:
            parts.append(f"Nearby I can see: {', '.join(near)}.")

        in_hand = [str(o) for o in data.get("in_hand", []) if o]
        if in_hand:
            parts.append(f"You appear to be holding: {', '.join(in_hand)}.")

        obstacles = [str(o) for o in data.get("obstacles", []) if o]
        if obstacles:
            parts.append(f"Watch out for: {', '.join(obstacles)}.")

        if not parts:
            return "I can see the scene but could not identify anything clearly."

        return " ".join(parts)