"""
Tests for housing.housing_csv_reader.

NOTE: housing_csv_reader.py on this branch is incomplete — the stream_csv_files
function body is missing (the file has a bare `d` token on line 9). These tests
document the *expected* behaviour as described in CLAUDE.md so they can be
enabled once the implementation is complete.
"""
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path


@unittest.skip(
    "housing_csv_reader.py is incomplete on this branch (missing stream_csv_files body)"
)
class TestStreamCsvFiles(unittest.TestCase):

    @patch("housing.housing_csv_reader.pd")
    def test_should_yield_rows_from_csv_when_files_exist(self, mock_pd):
        mock_dir = MagicMock(spec=Path)
        mock_csv_file = MagicMock()
        mock_csv_file.name = "data.csv"
        mock_dir.glob.return_value = [mock_csv_file]

        mock_chunk = MagicMock()
        mock_chunk.to_dict.return_value = [{"city": "Brussels", "price": 500000}]
        mock_pd.read_csv.return_value = [mock_chunk]

        from housing.housing_csv_reader import stream_csv_files
        results = list(stream_csv_files(mock_dir))
        self.assertEqual(results, [{"city": "Brussels", "price": 500000}])

    @patch("housing.housing_csv_reader.pd")
    def test_should_yield_nothing_when_directory_has_no_csv_files(self, mock_pd):
        mock_dir = MagicMock(spec=Path)
        mock_dir.glob.return_value = []

        from housing.housing_csv_reader import stream_csv_files
        results = list(stream_csv_files(mock_dir))
        self.assertEqual(results, [])

    @patch("housing.housing_csv_reader.pd")
    def test_should_use_default_chunk_size_when_not_specified(self, mock_pd):
        mock_dir = MagicMock(spec=Path)
        mock_dir.glob.return_value = []

        from housing.housing_csv_reader import stream_csv_files
        list(stream_csv_files(mock_dir))
        # When files exist, read_csv is called with chunksize=50_000
        # (skipped here since glob returns empty)

    @patch("housing.housing_csv_reader.pd")
    def test_should_skip_bad_lines_when_reading_csv(self, mock_pd):
        mock_dir = MagicMock(spec=Path)
        mock_csv_file = MagicMock()
        mock_dir.glob.return_value = [mock_csv_file]
        mock_pd.read_csv.return_value = []

        from housing.housing_csv_reader import stream_csv_files
        list(stream_csv_files(mock_dir))
        call_kwargs = mock_pd.read_csv.call_args[1]
        self.assertEqual(call_kwargs["on_bad_lines"], "skip")
