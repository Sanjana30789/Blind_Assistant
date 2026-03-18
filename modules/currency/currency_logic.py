# from utils.logger import logger
# import time


# def speak(text: str):
#     try:
#         from tts.speaker import Speaker
#         Speaker().speak(text)
#     except Exception as e:
#         logger.error(f"TTS error in currency_logic: {e}")


# def process_predictions(result):
#     # -------- Persistent memory (thread-safe) --------
#     if not hasattr(process_predictions, "last_spoken"):
#         process_predictions.last_spoken = ""
#         process_predictions.last_time = 0

#     last_spoken = process_predictions.last_spoken
#     last_time = process_predictions.last_time
#     cooldown = 3  # seconds

#     # logger.debug(f"Currency raw result: {result}")

#     try:
#         predictions = None

#         if isinstance(result, dict):
#             predictions = result.get("predictions") or result.get("output") or result.get("detections")

#         elif isinstance(result, list) and len(result) > 0:
#             first = result[0]
#             predictions = first.get("predictions") or first.get("output") or first.get("detections")

#         if predictions is None:
#             return

#         # -------- Extract label --------
#         class_names = []

#         if hasattr(predictions, "data"):
#             class_names = list(predictions.data.get("class_name", []))

#         elif isinstance(predictions, dict):
#             class_names = predictions.get("class_name", []) or predictions.get("classes", [])

#         elif isinstance(predictions, list):
#             class_names = [p.get("class") or p.get("class_name") for p in predictions if p]

#         class_names = [str(c) for c in class_names if c]

#         if not class_names:
#             return

#         label = class_names[0]

#         # -------- Speak logic --------
#         now = time.time()

#         if label != last_spoken and (now - last_time > cooldown):
#             message = f"{label.replace('_', ' ')} detected"
#             logger.info(f"Currency detected: {message}")
#             speak(message)

#             # store state safely
#             process_predictions.last_spoken = label
#             process_predictions.last_time = now

#     except Exception as e:
#         logger.error(f"Error processing currency predictions: {e}", exc_info=True)


from utils.logger import logger
import time

try:
    from tts.speaker import Speaker
    speaker = Speaker()
except Exception:
    speaker = None


def speak(text: str):
    try:
        if speaker:
            speaker.speak(text)
    except Exception as e:
        logger.error(f"TTS error in currency_logic: {e}")


def process_predictions(result):

    # -------- Persistent memory --------
    if not hasattr(process_predictions, "last_spoken"):
        process_predictions.last_spoken = ""
        process_predictions.last_time = 0

    last_spoken = process_predictions.last_spoken
    last_time = process_predictions.last_time

    cooldown = 3  # seconds

    try:
        predictions = None

        if isinstance(result, dict):
            predictions = result.get("predictions") or result.get("output") or result.get("detections")

        elif isinstance(result, list) and len(result) > 0:
            first = result[0]
            predictions = first.get("predictions") or first.get("output") or first.get("detections")

        if predictions is None:
            return

        # -------- Extract label --------
        class_names = []

        if hasattr(predictions, "data"):
            class_names = list(predictions.data.get("class_name", []))

        elif isinstance(predictions, dict):
            class_names = predictions.get("class_name", []) or predictions.get("classes", [])

        elif isinstance(predictions, list):
            class_names = [p.get("class") or p.get("class_name") for p in predictions if p]

        class_names = [str(c) for c in class_names if c]

        if not class_names:
            return

        label = class_names[0]

        # -------- Speak logic --------
        now = time.time()

        if label != last_spoken and (now - last_time > cooldown):

            message = f"{label.replace('_', ' ')} detected"

            logger.info(f"Currency detected: {message}")

            speak(message)

            process_predictions.last_spoken = label
            process_predictions.last_time = now

    except Exception as e:
        logger.error(f"Error processing currency predictions: {e}", exc_info=True)