import unittest

from model.session_manager import SessionManager


class TestSessionManager(unittest.TestCase):
    """Tests for the in-memory SessionManager on this branch."""

    def setUp(self):
        self.manager = SessionManager()

    def test_should_have_empty_store_when_initialized(self):
        self.assertEqual(self.manager.store, {})

    def test_should_return_history_instance_when_getting_new_session(self):
        history = self.manager.get_session_history("session-1")
        self.assertIsNotNone(history)

    def test_should_create_session_entry_when_session_id_not_in_store(self):
        self.manager.get_session_history("new-session")
        self.assertIn("new-session", self.manager.store)

    def test_should_return_same_history_when_called_with_same_session_id(self):
        history1 = self.manager.get_session_history("session-abc")
        history2 = self.manager.get_session_history("session-abc")
        self.assertIs(history1, history2)

    def test_should_return_different_histories_for_different_session_ids(self):
        history1 = self.manager.get_session_history("session-1")
        history2 = self.manager.get_session_history("session-2")
        self.assertIsNot(history1, history2)

    def test_should_not_overwrite_existing_session_when_called_again(self):
        history1 = self.manager.get_session_history("session-x")
        # Simulate adding a message (the object is still in store)
        self.manager.get_session_history("session-x")
        # The store entry should still be the original object
        self.assertIs(self.manager.store["session-x"], history1)

    def test_should_have_two_entries_when_two_different_sessions_used(self):
        self.manager.get_session_history("s1")
        self.manager.get_session_history("s2")
        self.assertEqual(len(self.manager.store), 2)
