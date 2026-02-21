from inference import InferencePipeline
from .currency_logic import process_predictions
from utils.logger import logger

pipeline = None


def start_currency_detection():
    global pipeline

    # If already running, don't start again
    if pipeline is not None:
        logger.warning("Currency detection already running — ignoring start call")
        return

    def sink(result, video_frame):
        process_predictions(result)

    logger.info("Initialising InferencePipeline...")

    pipeline = InferencePipeline.init_with_workflow(
        api_key="bWopVNDhPK2NeW9R2UCT",
        workspace_name="ws1-bxj0b",
        workflow_id="detect-count-and-visualize-3",
        video_reference=0,
        max_fps=20,
        on_prediction=sink
    )

    pipeline.start()
    # ✅ Do NOT call pipeline.join() here — it blocks forever
    # The pipeline runs in its own thread; we return immediately
    logger.info("Currency pipeline started ✓")


def stop_currency_detection():
    global pipeline

    if pipeline is None:
        logger.warning("No active currency pipeline to stop")
        return

    try:
        # ✅ Correct method is terminate(), not stop()
        pipeline.terminate()
        logger.info("Currency pipeline terminated ✓")
    except Exception as e:
        logger.error(f"Error terminating pipeline: {e}", exc_info=True)
    finally:
        pipeline = None