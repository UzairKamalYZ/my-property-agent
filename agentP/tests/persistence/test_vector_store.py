import unittest
import numpy as np

from persistence.vector_store import VectorStore


class TestVectorStoreABC(unittest.TestCase):

    def test_should_raise_type_error_when_instantiated_without_implementing_add(self):
        class IncompleteStore(VectorStore):
            def search(self, query_vector, k):
                return []

        with self.assertRaises(TypeError):
            IncompleteStore()

    def test_should_raise_type_error_when_instantiated_without_implementing_search(self):
        class IncompleteStore(VectorStore):
            def add(self, vectors, metadatas):
                pass

        with self.assertRaises(TypeError):
            IncompleteStore()

    def test_should_raise_type_error_when_instantiated_without_any_methods(self):
        class EmptyStore(VectorStore):
            pass

        with self.assertRaises(TypeError):
            EmptyStore()

    def test_should_not_raise_when_all_abstract_methods_implemented(self):
        class FullStore(VectorStore):
            def add(self, vectors, metadatas):
                pass

            def search(self, query_vector, k):
                return []

        try:
            store = FullStore()
        except TypeError:
            self.fail("FullStore raised TypeError unexpectedly")
        self.assertIsInstance(store, VectorStore)
