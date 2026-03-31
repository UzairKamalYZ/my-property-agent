import logging
import time

from agent import LocalAgent
from config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def run_cron_job():
    """Initializes LocalAgent, asks a predefined prompt, and logs the response."""
    logger.info("Cron job starting")
    try:
        with LocalAgent() as agent:
            prompt = "Give me a list of 2 bed apartment in poland with price less than 1000."
            logger.info("Cron job asking agent: prompt_len=%d", len(prompt))
            logger.debug("Cron job prompt: %r", prompt)
            response = agent.ask(prompt)
            logger.info("Cron job received response: response_len=%d", len(str(response)))
            logger.debug("Cron job response: %r", response)
    except Exception as e:
        logger.exception("Cron job failed: %s", e)
    logger.info("Cron job finished")


def main():
    interval_minutes = 30
    logger.info("Cron agent started — running every %d minute(s)", interval_minutes)
    try:
        while True:
            run_cron_job()
            logger.info("Cron agent sleeping for %d minute(s)", interval_minutes)
            time.sleep(interval_minutes * 60)
    except KeyboardInterrupt:
        logger.info("Cron agent terminated by user")
    except Exception as e:
        logger.exception("Unexpected error in cron main loop: %s", e)


if __name__ == "__main__":
    main()
