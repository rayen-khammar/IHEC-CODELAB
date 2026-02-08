import time
from app.config import BVMT_URL, POLL_INTERVAL
from app.database import init_db
from app.ingestion import run_one_cycle

def main():
    init_db()

    print("========================================")
    print("🚀 BVMT Realtime Ingestion Started")
    print(f"URL: {BVMT_URL}")
    print(f"Polling interval: {POLL_INTERVAL} sec")
    print("========================================")

    while True:
        try:
            df = run_one_cycle(BVMT_URL)
            print("cycle OK | rows:", len(df))
            print(df.head(5))
        except Exception as e:
            print("erreur ingestion:", e)

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
