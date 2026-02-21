# from utils.logger import logger

# last_spoken = ""


# def speak(text: str):
#     """Use the project's Speaker so there's no TTS conflict with pyttsx3."""
#     try:
#         from tts.speaker import Speaker
#         Speaker().speak(text)
#     except Exception as e:
#         logger.error(f"TTS error in currency_logic: {e}")


# def process_predictions(result):
#     global last_spoken

#     logger.debug(f"Currency raw result: {result}")

#     try:
#         # Workflow results come as a dict — log it so we can see exact structure
#         # Handle both workflow output formats
#         predictions = None

#         # Format 1: direct predictions key
#         if isinstance(result, dict):
#             predictions = result.get("predictions") or result.get("output") or result.get("detections")

#         # Format 2: result is a list
#         elif isinstance(result, list) and len(result) > 0:
#             first = result[0]
#             predictions = first.get("predictions") or first.get("output") or first.get("detections")

#         if predictions is None:
#             logger.debug("No predictions key found in result")
#             last_spoken = ""
#             return

#         # Extract class names — sv.Detections or dict
#         class_names = []

#         if hasattr(predictions, "data"):
#             # supervision Detections object
#             class_names = list(predictions.data.get("class_name", []))
#         elif isinstance(predictions, dict):
#             class_names = predictions.get("class_name", []) or predictions.get("classes", [])
#         elif isinstance(predictions, list):
#             # list of prediction dicts e.g. [{"class": "50", ...}, ...]
#             class_names = [p.get("class") or p.get("class_name") for p in predictions if p]

#         class_names = [str(c) for c in class_names if c]

#         if class_names:
#             label = class_names[0]
#             if label != last_spoken:
#                 message = f"{label} rupees note detected"
#                 logger.info(f"Currency detected: {message}")
#                 speak(message)
#                 last_spoken = label
#         else:
#             last_spoken = ""

#     except Exception as e:
#         logger.error(f"Error processing currency predictions: {e}", exc_info=True)


from utils.logger import logger
import time


def speak(text: str):
    try:
        from tts.speaker import Speaker
        Speaker().speak(text)
    except Exception as e:
        logger.error(f"TTS error in currency_logic: {e}")


def process_predictions(result):
    # -------- Persistent memory (thread-safe) --------
    if not hasattr(process_predictions, "last_spoken"):
        process_predictions.last_spoken = ""
        process_predictions.last_time = 0

    last_spoken = process_predictions.last_spoken
    last_time = process_predictions.last_time
    cooldown = 3  # seconds

    # logger.debug(f"Currency raw result: {result}")

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

            # store state safely
            process_predictions.last_spoken = label
            process_predictions.last_time = now

    except Exception as e:
        logger.error(f"Error processing currency predictions: {e}", exc_info=True)