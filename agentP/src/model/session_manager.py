from langchain_core.chat_history import InMemoryChatMessageHistory


class SessionManager:
    """Manages in-memory session history for chat interactions."""

    def __init__(self):
        self.store = {}

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """
        Retrieves the message history for a given session, creating it
        if it doesn't exist.
        """
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]
