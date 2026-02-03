from pathlib import Path

from agentP.src.housing.housing_data_collector import housing_data_collector

if __name__ == "__main__":
    with (housing_data_collector() as collector):
        BATCH_SIZE = 100
        documents = {}

        for row in collector.stream_csv_files(Path('../datasets/pl-housing')):
            doc = dict(row)  # copy all CSV fields
            doc["text"] = collector.generateEmbededDocument(row)

            documents[row["id"]] = doc

            # When batch is full → persist
            if len(documents) == BATCH_SIZE:
                collector.persist(documents)
                documents.clear()  # VERY important

        # Persist remaining records (last partial batch)
        if documents:
            collector.persist(documents)