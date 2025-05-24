"""Web search tool for PydanticAI agents to search the web."""

from pydantic_ai.common_tools.duckduckgo import DDGS, DuckDuckGoSearchTool
from pydantic_ai.tools import Tool


def web_search_tool(duckduckgo_client: DDGS | None = None, max_results: int | None = None) -> Tool:
	"""Creates a DuckDuckGo search tool.

	Args:
	    duckduckgo_client: The DuckDuckGo search client.
	    max_results: The maximum number of results. If None, returns results only from the first response.
	"""
	return Tool(
		DuckDuckGoSearchTool(client=duckduckgo_client or DDGS(), max_results=max_results).__call__,
		name="web_search",
		description="Searches the web for the given query and returns the results.",
	)
