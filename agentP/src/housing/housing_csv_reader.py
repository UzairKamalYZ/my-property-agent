from pathlib import Path
import pandas as pd


DATASET_DIR = Path("dataset")
CHUNK_SIZE = 50_000


d

def main():
    count = 0

    for row in stream_csv_files(DATASET_DIR):
        count += 1

        # Example: print first 3 rows only
        if count <= 3:
            print(row)

    print(f"\nTotal rows processed: {count}")


if __name__ == "__main__":
    main()
