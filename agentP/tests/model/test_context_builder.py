import unittest

from model.context_builder import build_context_from_listings, _format_amenities


class TestFormatAmenities(unittest.TestCase):

    def test_should_return_none_string_when_no_amenities_are_present(self):
        listing = {}
        result = _format_amenities(listing)
        self.assertEqual(result, "None")

    def test_should_return_all_amenities_when_all_fields_are_true(self):
        listing = {
            "hasParkingSpace": True,
            "hasBalcony": True,
            "hasElevator": True,
            "hasSecurity": True,
            "hasStorageRoom": True,
        }
        result = _format_amenities(listing)
        self.assertEqual(result, "Parking, Balcony, Elevator, Security, Storage")

    def test_should_return_partial_amenities_when_only_some_fields_are_true(self):
        listing = {"hasBalcony": True, "hasElevator": True}
        result = _format_amenities(listing)
        self.assertEqual(result, "Balcony, Elevator")

    def test_should_exclude_false_fields_when_building_amenities(self):
        listing = {"hasParkingSpace": False, "hasBalcony": True, "hasElevator": False}
        result = _format_amenities(listing)
        self.assertEqual(result, "Balcony")

    def test_should_return_none_string_when_all_amenity_fields_are_false(self):
        listing = {
            "hasParkingSpace": False,
            "hasBalcony": False,
            "hasElevator": False,
            "hasSecurity": False,
            "hasStorageRoom": False,
        }
        result = _format_amenities(listing)
        self.assertEqual(result, "None")


class TestBuildContextFromListings(unittest.TestCase):

    def test_should_return_no_listings_message_when_list_is_empty(self):
        result = build_context_from_listings([])
        self.assertEqual(result, "No listings found.")

    def test_should_format_listing_number_when_single_listing_given(self):
        listing = {"city": "Warsaw", "price": 500000}
        result = build_context_from_listings([listing])
        self.assertIn("Listing 1", result)

    def test_should_include_city_when_city_is_present(self):
        listing = {"city": "Warsaw"}
        result = build_context_from_listings([listing])
        self.assertIn("Location: Warsaw", result)

    def test_should_include_price_with_pln_suffix_when_price_present(self):
        listing = {"price": 500000}
        result = build_context_from_listings([listing])
        self.assertIn("Price: 500000 PLN", result)

    def test_should_return_na_when_field_is_none(self):
        listing = {"city": None}
        result = build_context_from_listings([listing])
        self.assertIn("Location: N/A", result)

    def test_should_return_na_when_field_is_nan_string(self):
        listing = {"price": "nan"}
        result = build_context_from_listings([listing])
        self.assertIn("Price: N/A", result)

    def test_should_return_na_when_field_is_empty_string(self):
        listing = {"city": ""}
        result = build_context_from_listings([listing])
        self.assertIn("Location: N/A", result)

    def test_should_return_na_when_field_is_missing(self):
        listing = {}
        result = build_context_from_listings([listing])
        self.assertIn("Location: N/A", result)
        self.assertIn("Price: N/A", result)

    def test_should_include_surface_with_m2_suffix_when_square_meters_present(self):
        listing = {"squareMeters": 65}
        result = build_context_from_listings([listing])
        self.assertIn("Surface: 65 m²", result)

    def test_should_format_floor_as_fraction_when_floor_and_floor_count_present(self):
        listing = {"floor": 3, "floorCount": 10}
        result = build_context_from_listings([listing])
        self.assertIn("Floor: 3 / 10", result)

    def test_should_number_listings_sequentially_when_multiple_listings_given(self):
        listings = [{"city": "Warsaw"}, {"city": "Krakow"}, {"city": "Gdansk"}]
        result = build_context_from_listings(listings)
        self.assertIn("Listing 1", result)
        self.assertIn("Listing 2", result)
        self.assertIn("Listing 3", result)

    def test_should_separate_listings_with_blank_lines_when_multiple_given(self):
        listings = [{"city": "Warsaw"}, {"city": "Krakow"}]
        result = build_context_from_listings(listings)
        self.assertIn("\n\n", result)

    def test_should_include_amenities_none_when_no_amenities_present(self):
        listing = {}
        result = build_context_from_listings([listing])
        self.assertIn("Amenities: None", result)

    def test_should_include_amenities_list_when_amenities_are_present(self):
        listing = {"hasBalcony": True, "hasParking": True}
        result = build_context_from_listings([listing])
        self.assertIn("Amenities:", result)
