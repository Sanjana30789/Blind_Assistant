import sys
import os
import warnings
import cv2
camera = None
camera_running = False
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["QWEN_2_5_ENABLED"] = "False"
os.environ["QWEN_3_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["SMOLVLM2_ENABLED"] = "False"
os.environ["DEPTH_ESTIMATION_ENABLED"] = "False"

# ── Path setup — must be FIRST before any local imports ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# ── Local imports ──
from utils.logger import logger
from utils.audio_utils import check_microphone_available
from tts.speaker import Speaker
from modules.stt.listener import listen, listen_from_file
from core.agent import agent
from core.state import AssistantState

# ── FastAPI imports ──
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import tempfile, shutil, uvicorn, threading, webbrowser, time, json, asyncio
from collections import deque

# ══════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════
app = FastAPI()
speaker = Speaker()
memory = {"last_scene": None}
camera = cv2.VideoCapture(0)

# SSE — keeps last 200 log entries so late-joining browsers get history
log_queue: deque = deque(maxlen=200)
sse_clients: list = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Serve the UI
@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(BASE_DIR, "UI", "index.html"))


# ══════════════════════════════════════════════
# SSE HELPERS  (replace logger.info / logger.debug calls)
# ══════════════════════════════════════════════
def push_log(level: str, msg: str):
    """Write to Python logger AND stream to all connected browser clients."""
    if level == "INFO":
        logger.info(msg)
    elif level == "DEBUG":
        logger.debug(msg)
    else:
        logger.warning(msg)

    entry = {"type": "log", "level": level, "msg": msg}
    log_queue.append(entry)
    for q in list(sse_clients):
        try:
            q.put_nowait(entry)
        except Exception:
            sse_clients.remove(q)


def push_event(data: dict):
    """Push a non-log SSE event (response / module / status) to all clients."""
    for q in list(sse_clients):
        try:
            q.put_nowait(data)
        except Exception:
            pass


# ══════════════════════════════════════════════
# STATE BUILDER
# ══════════════════════════════════════════════
def build_state(transcript: str) -> AssistantState:
    return {
        "raw_transcript":         transcript.strip(),
        "cleaned_transcript":     "",
        "mode":                   "unknown",
        "confidence":             0.0,
        "extra_context":          "",
        "raw_output":             {},
        "final_output":           "",
        "needs_clarification":    False,
        "clarification_question": "",
        "error":                  None,
        "retry_count":            0,
        "spoken":                 False,
    }


# ══════════════════════════════════════════════
# PIPELINE  (used by both web API and mic loop)
# ══════════════════════════════════════════════
def run_pipeline(transcript: str) -> dict:
    if not transcript.strip():
        push_log("WARN", "Skipping empty transcript")
        return {"response": "", "mode": "unknown", "confidence": 0.0}

    push_log("INFO", f"─── New request: '{transcript}' ───")
    push_event({"type": "status", "status": "processing"})

    state = build_state(transcript)

    try:
        result_state = agent.invoke(state)

        mode       = result_state.get("mode", "unknown")
        confidence = result_state.get("confidence", 0.0)
        output     = result_state.get("final_output", "")

        # Fallback: try other keys if final_output is empty
        if not output:
            output = (result_state.get("response") or
                      result_state.get("output") or
                      result_state.get("answer") or
                      result_state.get("text") or
                      "No response generated.")

        push_log("DEBUG", f"Graph result keys: {list(result_state.keys())}")
        push_log("DEBUG", f"mode={mode} | confidence={confidence} | "
                          f"final_output={output[:80]}")

        # Stream module + response to UI
        push_event({"type": "module",   "module": mode})
        push_event({"type": "response", "text": output,
                    "confidence": confidence, "mode": mode})

        if isinstance(result_state.get("raw_output"), dict):
            memory["last_scene"] = result_state["raw_output"]

        push_log("INFO", "─── Request complete ───")
        push_event({"type": "status", "status": "ready"})

        return {"response": output, "mode": mode, "confidence": confidence}

    except KeyError as e:
        msg = f"Pipeline state key error — missing key: {e}"
        push_log("WARN", msg)
        speaker.speak("Sorry, I had trouble understanding that.")
        push_event({"type": "status", "status": "ready"})
        return {"response": "Sorry, I had trouble understanding that.",
                "mode": "unknown", "confidence": 0.0}

    except ValueError as e:
        msg = f"Pipeline value error: {e}"
        push_log("WARN", msg)
        speaker.speak("Sorry, I had trouble understanding that.")
        push_event({"type": "status", "status": "ready"})
        return {"response": "Sorry, I had trouble understanding that.",
                "mode": "unknown", "confidence": 0.0}

    except Exception as e:
        msg = f"Pipeline error — type={type(e).__name__} | detail={e}"
        push_log("WARN", msg)
        speaker.speak("Sorry, I had trouble understanding that.")
        push_event({"type": "status", "status": "ready"})
        return {"response": "Sorry, I had trouble understanding that.",
                "mode": "unknown", "confidence": 0.0}


# ══════════════════════════════════════════════
# SSE STREAM ENDPOINT  — GET /api/stream
# ══════════════════════════════════════════════
@app.get("/api/stream")
async def stream_events():
    from asyncio import Queue
    client_q: Queue = Queue()
    sse_clients.append(client_q)

    # Replay buffered history for newly connected browser
    for entry in list(log_queue):
        await client_q.put(entry)

    async def generator():
        try:
            while True:
                # Poll every 0.1s so we never block the event loop
                try:
                    data = client_q.get_nowait()
                    yield f"data: {json.dumps(data)}\n\n"
                except Exception:
                    # Nothing in queue — send heartbeat every 5s
                    yield ": heartbeat\n\n"
                    await asyncio.sleep(5)
                    continue
        except asyncio.CancelledError:
            pass
        finally:
            try:
                sse_clients.remove(client_q)
            except ValueError:
                pass

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


# ══════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════

# 1. Text input → pipeline
class TextRequest(BaseModel):
    text: str

@app.post("/api/text")
def process_text(req: TextRequest):
    result = run_pipeline(req.text)
    audio_path = _try_speak_to_file(result["response"])
    result["audio_url"] = f"/api/audio?path={audio_path}" if audio_path else None
    return JSONResponse(result)


# 2. Voice recording → Whisper STT → pipeline
@app.post("/api/voice")
async def process_voice(audio: UploadFile = File(...)):
    # Browser MediaRecorder outputs WebM/Opus — use .webm so Groq gets the right format
    suffix = "." + (audio.content_type.split("/")[-1].split(";")[0] or "webm")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name
    try:
        push_log("INFO", "🎙 Received audio — transcribing…")
        transcript = listen_from_file(tmp_path)

        if not transcript:
            push_log("WARN", "Could not transcribe audio — empty result")
            return JSONResponse({
                "response": "Could not transcribe. Please try again.",
                "mode": "unknown", "confidence": 0.0, "transcript": ""
            })

        push_log("INFO", f"STT → \"{transcript}\"")
        result = run_pipeline(transcript)
        result["transcript"] = transcript

        audio_path = _try_speak_to_file(result["response"])
        result["audio_url"] = f"/api/audio?path={audio_path}" if audio_path else None
        return JSONResponse(result)

    finally:
        os.unlink(tmp_path)

@app.get("/api/camera")
def camera_stream():
    return StreamingResponse(
        generate_camera_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

def generate_camera_stream():
    global camera_running, camera

    while camera_running:

        success, frame = camera.read()

        if not success:
            time.sleep(0.03)
            continue

        # encode frame
        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

        # small delay so CPU doesn't overload
        time.sleep(0.03)


# 3. Serve TTS audio file
@app.get("/api/audio")
def serve_audio(path: str):
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/mpeg")
    return JSONResponse({"error": "Audio not found"}, status_code=404)


# 4. Health check — UI polls this to show connection status
@app.get("/api/health")
def health():
    return {"status": "ok"}


# ══════════════════════════════════════════════
# HELPER — speak_to_file (graceful fallback)
# ══════════════════════════════════════════════
def _try_speak_to_file(text: str):
    """Call speaker.speak_to_file() if it exists, else fall back to speaker.speak()."""
    if hasattr(speaker, "speak_to_file"):
        try:
            return speaker.speak_to_file(text)
        except Exception as e:
            push_log("WARN", f"speak_to_file failed: {e} — falling back to speak()")
    speaker.speak(text)
    return None


# ══════════════════════════════════════════════
# OPTIONAL BACKGROUND MIC LOOP
# ══════════════════════════════════════════════
def mic_loop():
    """
    Always-on microphone listener — identical to the original terminal loop.
    Runs in a background thread alongside the web server.
    """
    if not check_microphone_available():
        push_log("WARN", "Mic loop: no microphone found — skipping")
        return

    push_log("INFO", "🎙 Background microphone loop started")
    while True:
        try:
            transcript = listen()
            if transcript:
                push_log("INFO", f"🎙 Mic heard: \"{transcript}\"")
                # Push transcript to UI so "Last heard" chip updates
                push_event({"type": "transcript", "text": transcript})
                run_pipeline(transcript)
        except Exception as e:
            push_log("WARN", f"Mic loop error: {type(e).__name__}: {e}")
            time.sleep(1)


# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════
def open_browser():
    time.sleep(1.8)
    webbrowser.open("http://localhost:8000")


if __name__ == "__main__":
    logger.info("══════════════════════════════════════")
    logger.info("    Blind Assistant — Starting Up     ")
    logger.info("    Press Ctrl+C to stop              ")
    logger.info("══════════════════════════════════════")

    push_log("INFO", "Blind Assistant — Starting Up")
    push_log("INFO", "UI available at http://localhost:8000")

    threading.Thread(target=open_browser, daemon=True).start()
    threading.Thread(target=mic_loop, daemon=True).start()

    # Run uvicorn in a thread so the main thread stays free to catch Ctrl+C
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning"))
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down")
        speaker.speak("Goodbye!")
        os._exit(0)