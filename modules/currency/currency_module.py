import threading
from .currency_detector import start_currency_detection, stop_currency_detection
from utils.logger import logger

currency_thread = None
currency_active = False


def start_currency_mode():
    global currency_thread, currency_active

    if currency_active:
        logger.warning("Currency already active — ignoring duplicate start")
        return

    currency_active = True
    currency_thread = threading.Thread(target=start_currency_detection, daemon=True)
    currency_thread.start()
    logger.info("Currency thread started ✓")


def stop_currency_mode():
    global currency_thread, currency_active

    if not currency_active:
        logger.warning("Currency not active — nothing to stop")
        return

    stop_currency_detection()
    currency_active = False
    currency_thread = None
    logger.info("Currency thread stopped ✓")