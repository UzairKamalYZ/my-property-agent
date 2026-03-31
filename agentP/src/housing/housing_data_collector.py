import logging
from pathlib import Path

import pandas as pd

from agentP.src.model.embedder import Embedder

logger = logging.getLogger(__name__)


class housing_data_collector:
    def __init__(self):
        self.embedder = Embedder()
        logger.debug("housing_data_collector initialized")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        pass

    def generateEmbededDocument(self, row: dict) -> str:
        parts = []

        if row.get("city"):
            parts.append(f"Apartment in {str(row['city']).capitalize()}.")

        if row.get("rooms") and row.get("squareMeters"):
            parts.append(f"{row['rooms']} rooms, {row['squareMeters']} square meters.")

        if row.get("floor") is not None and row.get("floorCount") is not None:
            parts.append(
                f"Located on floor {row['floor']} of a {row['floorCount']}-floor building."
            )

        if row.get("ownership"):
            parts.append(f"{str(row['ownership']).capitalize()} ownership.")

        if row.get("buildingMaterial") and str(row["buildingMaterial"]) != "nan":
            parts.append(f"{str(row['buildingMaterial']).capitalize()} building.")

        # Proximity logic
        nearby = []
        if row.get("schoolDistance") is not None and row["schoolDistance"] <= 0.8:
            nearby.append("schools")
        if row.get("clinicDistance") is not None and row["clinicDistance"] <= 0.8:
            nearby.append("clinics")
        if row.get("restaurantDistance") is not None and row["restaurantDistance"] <= 0.8:
            nearby.append("restaurants")

        if nearby:
            parts.append(f"Close to {', '.join(nearby)}.")

        if row.get("price"):
            parts.append(f"Price is {row['price']} PLN.")

        # Dynamic fallback: include ALL remaining fields
        for key, value in row.items():
            if value is None or str(value).lower() in {"nan", "", "none"}:
                continue

            if key in {
                "city", "rooms", "squareMeters", "floor", "floorCount",
                "ownership", "buildingMaterial", "schoolDistance",
                "clinicDistance", "restaurantDistance", "price"
            }:
                continue

            parts.append(f"{key.replace('_', ' ')}: {value}.")

        return " ".join(parts)

    def stream_csv_files(self, dataset_dir: Path, CHUNK_SIZE=5000):
        """
        Generator that yields rows (dict) from all CSV files in a folder,
        reading them in streaming mode to avoid memory issues.
        """
        for csv_file in dataset_dir.glob("*.csv"):
            logger.info("Reading CSV file: %s", csv_file.name)
            for chunk in pd.read_csv(
                csv_file,
                chunksize=CHUNK_SIZE,
                encoding="utf-8",
                on_bad_lines="skip"
            ):
                for row in chunk.to_dict(orient="records"):
                    yield row

    def persist(self, data: dict):
        logger.info("Persisting %d document(s) to vector store", len(data))
        vectors = self.embedder.embed_documents_to_vectors(data)
        self.embedder.save_vectors_in_store(vectors)
        logger.debug("Persist complete")
