import logging
from langchain_core.chat_history import InMemoryChatMessageHistory

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages in-memory session history for chat interactions."""

    def __init__(self):
        self.store = {}
        logger.debug("SessionManager initialized")

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """
        Retrieves the message history for a given session, creating it
        if it doesn't exist.
        """
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
            logger.debug("New session created: session_id=%s", session_id)
        else:
            logger.debug("Resuming existing session: session_id=%s", session_id)
        return self.store[session_id]
