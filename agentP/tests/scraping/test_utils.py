import unittest

from scraping.utils import listings_to_documents


class TestListingsToDocuments(unittest.TestCase):

    def test_should_return_empty_list_when_given_no_listings(self):
        result = listings_to_documents([])
        self.assertEqual(result, [])

    def test_should_return_one_document_per_listing(self):
        listings = [
            {"type": "Apartment", "city": "Warsaw", "rent": 1200, "description": "Nice flat"},
            {"type": "House", "city": "Krakow", "rent": 800, "description": "Cozy house"},
        ]
        result = listings_to_documents(listings)
        self.assertEqual(len(result), 2)

    def test_should_format_page_content_with_type_city_and_rent(self):
        listings = [{"type": "Apartment", "city": "Brussels", "rent": 1500, "description": ""}]
        doc = listings_to_documents(listings)[0]
        self.assertIn("Apartment", doc.page_content)
        self.assertIn("Brussels", doc.page_content)
        self.assertIn("1500", doc.page_content)

    def test_should_include_description_in_page_content(self):
        listings = [{"type": "Flat", "city": "Ghent", "rent": 900, "description": "Spacious flat"}]
        doc = listings_to_documents(listings)[0]
        self.assertIn("Spacious flat", doc.page_content)

    def test_should_include_type_in_metadata(self):
        listings = [{"type": "House", "city": "Antwerp", "rent": 1100}]
        doc = listings_to_documents(listings)[0]
        self.assertEqual(doc.metadata["type"], "House")

    def test_should_include_city_in_metadata(self):
        listings = [{"type": "Apartment", "city": "Liège", "rent": 750}]
        doc = listings_to_documents(listings)[0]
        self.assertEqual(doc.metadata["city"], "Liège")

    def test_should_include_rent_in_metadata(self):
        listings = [{"type": "Apartment", "city": "Namur", "rent": 650}]
        doc = listings_to_documents(listings)[0]
        self.assertEqual(doc.metadata["rent"], 650)

    def test_should_include_url_in_metadata(self):
        listings = [{"url": "http://example.com/listing/1"}]
        doc = listings_to_documents(listings)[0]
        self.assertEqual(doc.metadata["url"], "http://example.com/listing/1")

    def test_should_use_defaults_when_fields_are_missing(self):
        listings = [{}]
        doc = listings_to_documents(listings)[0]
        self.assertIn("Unknown property", doc.page_content)
        self.assertIn("Unknown city", doc.page_content)
        self.assertIn("Unknown rent", doc.page_content)

    def test_should_set_default_type_in_metadata_when_type_missing(self):
        listings = [{}]
        doc = listings_to_documents(listings)[0]
        self.assertEqual(doc.metadata["type"], "Unknown property")

    def test_should_set_empty_url_in_metadata_when_url_missing(self):
        listings = [{}]
        doc = listings_to_documents(listings)[0]
        self.assertEqual(doc.metadata["url"], "")

    def test_should_set_empty_description_in_metadata_when_description_missing(self):
        listings = [{}]
        doc = listings_to_documents(listings)[0]
        self.assertEqual(doc.metadata["description"], "")
