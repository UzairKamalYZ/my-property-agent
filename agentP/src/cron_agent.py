import time

from agentP.src.agent import LocalAgent
from agentP.src.config.config import Config


def run_cron_job():
    """
    Initializes LocalAgent, asks a predefined prompt, and prints the response.
    """
    print(f"[{time.ctime()}] Starting cron job...")
    try:
        with LocalAgent() as agent:
            prompt = Config.CRON_SEARCH_PROMPT
            print(f"[{time.ctime()}] Asking agent: '{prompt}'")
            response = agent.ask(prompt)
            print(f"[{time.ctime()}] Agent response:{response}")
    except Exception as e:
        print(f"[{time.ctime()}] Error during cron job: {e}")
    print(f"[{time.ctime()}] Cron job finished.")


def main():
    interval_minutes = 30
    print(f"[{time.ctime()}] Cron agent started. Running every {interval_minutes} minutes.")
    try:
        while True:
            run_cron_job()
            print(f"[{time.ctime()}] Sleeping for {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)
    except KeyboardInterrupt:
        print(f"[{time.ctime()}] Cron agent terminated by user.")
    except Exception as e:
        print(f"[{time.ctime()}] An unexpected error occurred in main loop: {e}")


if __name__ == "__main__":
    main()
