import time


from agent import LocalAgent

def run_cron_job():
    """
    Initializes LocalAgent, asks a predefined prompt, and prints the response.
    """
    print(f"[{time.ctime()}] Starting cron job...")
    try:
        # LocalAgent manages its own session, so a new instance per job is okay
        # or we can pass a session_id to maintain conversation state if needed.
        # For a single prompt, a new instance is simplest.
        with LocalAgent() as agent:
            prompt = "Give me a list of 2 bed apartment in poland with price less than 1000."
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
