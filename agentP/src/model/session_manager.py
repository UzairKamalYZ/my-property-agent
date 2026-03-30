from langchain_community.chat_message_histories import SQLChatMessageHistory

from agentP.src.config.config import Config


class SessionManager:
    """Manages persistent session history for chat interactions backed by SQLite."""

    def __init__(self):
        self._db_url = f"sqlite:///{Config.SESSION_DB_FILE}"

    def get_session_history(self, session_id: str) -> SQLChatMessageHistory:
        """
        Retrieves the message history for a given session.
        History is persisted to SQLite and survives process restarts.
        """
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=self._db_url,
        )
