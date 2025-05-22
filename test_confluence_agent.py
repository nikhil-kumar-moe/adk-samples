import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock # For mocking

import httpx # For type hinting AsyncClient and for spec in mock

from confluence_agent import ConfluenceAgent
from agent_schemas import (
    QueryInput,
    ConfluenceSearchOutput,
    ContextDetail,
    confluence_search_runbooks_tool_schema # For checking schema registration if needed
)

@pytest.fixture
async def http_client_mock():
    """Provides a mock httpx.AsyncClient."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    # mock_client.aclose = AsyncMock() # Ensure aclose is an AsyncMock if not automatically by spec
    return mock_client

@pytest.fixture
async def confluence_agent_fixture(http_client_mock: AsyncMock):
    """
    Initializes and yields a ConfluenceAgent instance with mocked dependencies.
    Handles agent.close() in teardown.
    """
    # Mocking embedding_service and qdrant_client as they are part of the
    # __init__ signature but not strictly used by the current placeholder logic.
    mock_embedding_service = MagicMock(name="MockEmbeddingService")
    mock_qdrant_client = MagicMock(name="MockQdrantClient")
    test_collection_name = "test_runbook_collection"

    agent = ConfluenceAgent(
        model_name="test-model",
        # embedding_service=mock_embedding_service, # Passed but not used by placeholder
        # qdrant_client=mock_qdrant_client,         # Passed but not used by placeholder
        confluence_collection_name=test_collection_name,
        httpx_client=http_client_mock
    )
    yield agent # The test runs at this point
    await agent.close() # Teardown: close the agent and its httpx_client

@pytest.mark.asyncio
async def test_search_runbooks_success(confluence_agent_fixture: ConfluenceAgent):
    """Tests successful placeholder execution of search_runbooks."""
    agent = confluence_agent_fixture
    query_text = "how to fix API errors"
    query_input = QueryInput(query=query_text)

    # The placeholder search_runbooks uses asyncio.sleep, no external calls to mock for its direct execution
    result = await agent.search_runbooks(query_input)

    assert isinstance(result, ConfluenceSearchOutput)
    assert result.query == query_text
    assert len(result.results) == 1 # Based on placeholder data
    first_result = result.results[0]
    assert isinstance(first_result, ContextDetail)
    assert first_result.source_collection == agent.confluence_collection_name
    assert "Runbook: Resolve API Gateway Errors" in first_result.title
    assert "This is a sample runbook content" in first_result.text

@pytest.mark.asyncio
async def test_search_runbooks_missing_services_placeholder_behavior(http_client_mock: AsyncMock):
    """
    Tests search_runbooks behavior when embedding_service or qdrant_client are None.
    Note: The current placeholder in ConfluenceAgent does NOT change its behavior (e.g., return empty results)
    if these services are None, as it doesn't use them. This test reflects the *current* placeholder behavior.
    A future implementation that relies on these services would need different assertions here (e.g., empty list).
    """
    test_collection_name = "test_runbook_collection_no_services"
    # Initialize agent with None for services that might be optional in a real implementation
    agent = ConfluenceAgent(
        model_name="test-model-no-services",
        # embedding_service=None, # Explicitly None
        # qdrant_client=None,     # Explicitly None
        confluence_collection_name=test_collection_name,
        httpx_client=http_client_mock
    )
    query_text = "query with missing services"
    query_input = QueryInput(query=query_text)
    result = await agent.search_runbooks(query_input)

    # Current placeholder behavior: still returns dummy data
    assert isinstance(result, ConfluenceSearchOutput)
    assert len(result.results) == 1
    assert result.results[0].source_collection == test_collection_name
    # If the requirement was to return empty results, this assertion would be:
    # assert len(result.results) == 0
    await agent.close()


@pytest.mark.asyncio
async def test_process_method_with_query_input(confluence_agent_fixture: ConfluenceAgent):
    """Tests the process method with QueryInput."""
    agent = confluence_agent_fixture
    query_text = "process this query input"
    query_input = QueryInput(query=query_text)

    result = await agent.process(query_input)

    assert isinstance(result, ConfluenceSearchOutput)
    assert result.query == query_text
    assert len(result.results) == 1 # Based on placeholder data

@pytest.mark.asyncio
async def test_process_method_with_string_input(confluence_agent_fixture: ConfluenceAgent):
    """Tests the process method with a string input."""
    agent = confluence_agent_fixture
    query_text = "process this string"

    result = await agent.process(query_text)

    assert isinstance(result, ConfluenceSearchOutput)
    assert result.query == query_text
    assert len(result.results) == 1 # Based on placeholder data

@pytest.mark.asyncio
async def test_process_method_with_invalid_input(confluence_agent_fixture: ConfluenceAgent):
    """Tests the process method with an invalid input type."""
    agent = confluence_agent_fixture
    invalid_input = 12345 # An integer, not QueryInput or str

    result = await agent.process(invalid_input)

    assert isinstance(result, str)
    assert "ConfluenceAgent cannot process this type of request" in result

@pytest.mark.asyncio
async def test_agent_initialization_and_tool_registration(confluence_agent_fixture: ConfluenceAgent):
    """Tests if the agent initializes correctly and registers its tool."""
    agent = confluence_agent_fixture
    assert agent is not None
    assert agent.confluence_collection_name == "test_runbook_collection" # From fixture
    assert "search_confluence_runbooks" in agent.tools
    tool = agent.tools["search_confluence_runbooks"]
    assert tool.fn == agent.search_runbooks
    # Check if the schema matches the one imported
    assert tool.schema_dict == confluence_search_runbooks_tool_schema

@pytest.mark.asyncio
async def test_close_method_called_on_http_client(http_client_mock: AsyncMock):
    """
    Tests that the agent's close method also closes the http_client.
    This is implicitly tested via the confluence_agent_fixture teardown.
    We create a new agent here to isolate the check on http_client_mock.
    """
    agent = ConfluenceAgent(httpx_client=http_client_mock)
    await agent.close()
    http_client_mock.aclose.assert_called_once()

# To run these tests: `pytest test_confluence_agent.py`
# Ensure `confluence_agent.py` and `agent_schemas.py` are accessible.
# The tests assume `google.adk` is available in the environment.
# The `embedding_service` and `qdrant_client` parameters in `ConfluenceAgent.__init__`
# are currently placeholders in the agent's logic, so tests for their absence
# reflect this placeholder behavior rather than a hypothetical real-world scenario
# where their absence might lead to empty search results.
# The `test_search_runbooks_missing_services_placeholder_behavior` explicitly calls this out.
