import unittest
from unittest.mock import patch, MagicMock

from persistence.factory import create_vector_store


class TestCreateVectorStore(unittest.TestCase):

    @patch("persistence.factory.FAISSStore")
    def test_should_return_faiss_store_when_store_type_is_local(self, MockFAISS):
        mock_instance = MagicMock()
        MockFAISS.return_value = mock_instance

        result = create_vector_store("local", 384, {})

        self.assertIs(result, mock_instance)

    @patch("persistence.factory.FAISSStore")
    def test_should_pass_dim_to_faiss_store_when_store_type_is_local(self, MockFAISS):
        create_vector_store("local", 512, {})
        MockFAISS.assert_called_once_with(512)

    @patch("persistence.factory.Config")
    @patch("persistence.factory.PineconeStore")
    def test_should_return_pinecone_store_when_store_type_is_pinecone(
        self, MockPinecone, MockConfig
    ):
        MockConfig.PINECONE_INDEX_NAME = "test-index"
        MockConfig.PINECONE_API_KEY = "test-key"
        MockConfig.PINECONE_ENVIRONMENT = "us-east-1"
        mock_instance = MagicMock()
        MockPinecone.return_value = mock_instance

        result = create_vector_store("pinecone", 384, {})

        self.assertIs(result, mock_instance)

    @patch("persistence.factory.Config")
    @patch("persistence.factory.PineconeStore")
    def test_should_pass_correct_config_to_pinecone_store_when_creating(
        self, MockPinecone, MockConfig
    ):
        MockConfig.PINECONE_INDEX_NAME = "property-agent"
        MockConfig.PINECONE_API_KEY = "secret-key"
        MockConfig.PINECONE_ENVIRONMENT = "gcpstart"

        create_vector_store("pinecone", 384, {})

        MockPinecone.assert_called_once_with(
            index_name="property-agent",
            dim=384,
            api_key="secret-key",
            env="gcpstart",
        )

    def test_should_raise_value_error_when_store_type_is_unknown(self):
        with self.assertRaises(ValueError) as ctx:
            create_vector_store("redis", 384, {})
        self.assertIn("redis", str(ctx.exception))

    def test_should_raise_value_error_when_store_type_is_empty_string(self):
        with self.assertRaises(ValueError):
            create_vector_store("", 384, {})
