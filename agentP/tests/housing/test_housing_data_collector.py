import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestHousingDataCollector(unittest.TestCase):

    def setUp(self):
        self.mock_embedder = MagicMock()
        self.mock_embedder_cls = MagicMock(return_value=self.mock_embedder)

        with patch("agentP.src.model.embedder.Embedder", self.mock_embedder_cls):
            from housing.housing_data_collector import housing_data_collector
            self.collector = housing_data_collector.__new__(housing_data_collector)
            self.collector.embedder = self.mock_embedder

    # ------------------------------------------------------------------
    # generateEmbededDocument()
    # ------------------------------------------------------------------

    def test_should_include_city_in_document_when_city_is_present(self):
        row = {"city": "warsaw"}
        result = self.collector.generateEmbededDocument(row)
        self.assertIn("Warsaw", result)

    def test_should_include_rooms_and_sqm_when_both_present(self):
        row = {"rooms": 3, "squareMeters": 75}
        result = self.collector.generateEmbededDocument(row)
        self.assertIn("3 rooms", result)
        self.assertIn("75 square meters", result)

    def test_should_not_include_rooms_when_sqm_is_missing(self):
        row = {"rooms": 3}
        result = self.collector.generateEmbededDocument(row)
        self.assertNotIn("rooms", result)

    def test_should_include_floor_info_when_floor_and_floor_count_present(self):
        row = {"floor": 2, "floorCount": 5}
        result = self.collector.generateEmbededDocument(row)
        self.assertIn("floor 2", result)
        self.assertIn("5-floor building", result)

    def test_should_include_ownership_capitalised_when_present(self):
        row = {"ownership": "full"}
        result = self.collector.generateEmbededDocument(row)
        self.assertIn("Full ownership", result)

    def test_should_include_building_material_when_not_nan(self):
        row = {"buildingMaterial": "brick"}
        result = self.collector.generateEmbededDocument(row)
        self.assertIn("Brick building", result)

    def test_should_exclude_building_material_when_value_is_nan(self):
        row = {"buildingMaterial": "nan"}
        result = self.collector.generateEmbededDocument(row)
        self.assertNotIn("building", result)

    def test_should_include_price_when_price_present(self):
        row = {"price": 450000}
        result = self.collector.generateEmbededDocument(row)
        self.assertIn("450000 PLN", result)

    def test_should_include_nearby_schools_when_distance_within_threshold(self):
        row = {"schoolDistance": 0.5}
        result = self.collector.generateEmbededDocument(row)
        self.assertIn("schools", result)

    def test_should_exclude_schools_when_distance_exceeds_threshold(self):
        row = {"schoolDistance": 1.2}
        result = self.collector.generateEmbededDocument(row)
        self.assertNotIn("schools", result)

    def test_should_include_clinics_when_clinic_distance_within_threshold(self):
        row = {"clinicDistance": 0.8}
        result = self.collector.generateEmbededDocument(row)
        self.assertIn("clinics", result)

    def test_should_include_restaurants_when_restaurant_distance_within_threshold(self):
        row = {"restaurantDistance": 0.3}
        result = self.collector.generateEmbededDocument(row)
        self.assertIn("restaurants", result)

    def test_should_return_empty_string_when_row_is_empty(self):
        result = self.collector.generateEmbededDocument({})
        self.assertEqual(result, "")

    def test_should_exclude_none_values_from_dynamic_fields(self):
        row = {"someField": None}
        result = self.collector.generateEmbededDocument(row)
        self.assertNotIn("someField", result)

    def test_should_exclude_nan_string_from_dynamic_fields(self):
        row = {"someField": "nan"}
        result = self.collector.generateEmbededDocument(row)
        self.assertNotIn("someField", result)

    # ------------------------------------------------------------------
    # persist()
    # ------------------------------------------------------------------

    def test_should_call_embed_documents_to_vectors_when_persisting(self):
        data = [{"city": "Warsaw"}]
        self.collector.persist(data)
        self.mock_embedder.embed_documents_to_vectors.assert_called_once_with(data)

    def test_should_call_save_vectors_in_store_when_persisting(self):
        data = [{"city": "Warsaw"}]
        mock_vectors = MagicMock()
        self.mock_embedder.embed_documents_to_vectors.return_value = mock_vectors
        self.collector.persist(data)
        self.mock_embedder.save_vectors_in_store.assert_called_once_with(mock_vectors)

    # ------------------------------------------------------------------
    # context manager
    # ------------------------------------------------------------------

    def test_should_support_context_manager_protocol(self):
        with patch("agentP.src.model.embedder.Embedder", self.mock_embedder_cls):
            from housing.housing_data_collector import housing_data_collector
            with housing_data_collector() as collector:
                self.assertIsInstance(collector, housing_data_collector)

    # ------------------------------------------------------------------
    # stream_csv_files()
    # ------------------------------------------------------------------

    @patch("housing.housing_data_collector.pd")
    def test_should_yield_rows_from_csv_when_streaming(self, mock_pd):
        mock_dir = MagicMock(spec=Path)
        mock_file = MagicMock()
        mock_file.name = "data.csv"
        mock_dir.glob.return_value = [mock_file]

        mock_chunk = MagicMock()
        mock_chunk.to_dict.return_value = [{"city": "Warsaw"}]
        mock_pd.read_csv.return_value = [mock_chunk]

        results = list(self.collector.stream_csv_files(mock_dir))
        self.assertEqual(results, [{"city": "Warsaw"}])
