from langchain_core.tools import tool

from .rag_context_manager import RagContextManager


def make_property_search_tool(rag_context_manager: RagContextManager):
    """
    Returns a @tool-decorated function closed over the given RagContextManager.
    Called once during LlmModelGraph.__init__.
    """

    @tool
    def property_search(query: str) -> str:
        """Search the property listings database for apartments or houses
        matching the given query. Returns formatted listing details
        including price, location, rooms, surface area, and amenities.
        Use this tool when the user asks about properties, flats, or apartments."""
        return rag_context_manager.get_context(query)

    return property_search
