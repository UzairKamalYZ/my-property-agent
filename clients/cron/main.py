from logging_config import setup_logging
setup_logging()

import time

from agentP.src.agent import LocalAgent
from agentP.src.config.config import Config
from clients.base import BaseClient


def _run_job() -> None:
    """Run a single scheduled search and print the result."""
    print(f"[{time.ctime()}] Starting cron job...")
    try:
        with LocalAgent() as agent:
            prompt = Config.CRON_SEARCH_PROMPT
            print(f"[{time.ctime()}] Asking agent: '{prompt}'")
            response = agent.ask(prompt)
            print(f"[{time.ctime()}] Agent response: {response}")
    except Exception as e:
        print(f"[{time.ctime()}] Error during cron job: {e}")
    print(f"[{time.ctime()}] Cron job finished.")


class CronClient(BaseClient):
    """Runs a property search on a fixed interval."""

    def __init__(self, interval_minutes: int = 30):
        self.interval_minutes = interval_minutes
        self._running = False

    def start(self) -> None:
        self._running = True
        print(f"[{time.ctime()}] Cron client started. Running every {self.interval_minutes} minutes.")
        try:
            while self._running:
                _run_job()
                print(f"[{time.ctime()}] Sleeping for {self.interval_minutes} minutes...")
                # Sleep in 1-second ticks so stop() can interrupt promptly.
                for _ in range(self.interval_minutes * 60):
                    if not self._running:
                        break
                    time.sleep(1)
        except KeyboardInterrupt:
            print(f"[{time.ctime()}] Cron client terminated by user.")
        except Exception as e:
            print(f"[{time.ctime()}] Unexpected error in main loop: {e}")

    def stop(self) -> None:
        self._running = False
        print(f"[{time.ctime()}] Cron client stopping.")


if __name__ == "__main__":
    CronClient().start()
