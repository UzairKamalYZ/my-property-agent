import logging

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..config.config import Config
from .rag_context_manager import RagContextManager


class _PropertySearchInput(BaseModel):
    """Input schema for the property_search tool.

    An explicit Pydantic schema ensures the LLM always receives a correctly-typed
    object schema (not a bare string schema), which prevents the validation error
    "input must be a valid string" that occurs when the LLM passes a JSON object
    to a tool that was only declared to accept a plain str.
    """

    query: str = Field(
        description=(
            "The property search query in natural language. "
            "Examples: '2 bedroom flat in Warsaw under €800', "
            "'studio apartment near city centre', "
            "'3-room house with garden to rent'."
        )
    )

logger = logging.getLogger(__name__)

_TOOL_DESCRIPTION = (
    "Search the property listings database for real estate matching the given query. "
    "Returns formatted listing details including price, location, number of rooms, "
    "surface area, and amenities. "
    "Use this tool whenever the user asks about apartments, flats, houses, studios, "
    "duplexes, penthouses, rooms, rent, buying, listings, accommodation, location, "
    "price, cost, or any other property-related topic."
)


def make_property_search_tool(rag_context_manager: RagContextManager) -> StructuredTool:
    """
    Returns a StructuredTool closed over the given RagContextManager.
    Called once during LlmModelGraph.__init__.

    Using StructuredTool.from_function with an explicit name avoids the tool-name
    collision that occurs when the @tool decorator is applied to a nested function
    and make_property_search_tool() is called more than once.
    """
    k = Config.RAG_K

    def _search(query: str) -> str:
        logger.debug("property_search: query=%r k=%d", query, k)
        try:
            result = rag_context_manager.get_context(query, k=k)
            logger.info(
                "property_search: returned %d chars for query_len=%d",
                len(result),
                len(query),
            )
            return result
        except Exception:
            logger.exception("property_search: failed for query=%r", query)
            return "Property search is currently unavailable. Please try again later."

    return StructuredTool.from_function(
        func=_search,
        name="property_search",
        description=_TOOL_DESCRIPTION,
        args_schema=_PropertySearchInput,
    )
