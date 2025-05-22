import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock # AsyncMock for async methods

# Import the agent to be tested
from request_handler_agent import RequestHandlerAgent

# Import dependent agent types for creating mock specs
from zendesk_agent import ZendeskAgent
from confluence_agent import ConfluenceAgent

# Import Pydantic models used in interactions
from agent_schemas import (
    QueryInput,
    SynthesizerInput,
    ZendeskSearchOutput,
    ConfluenceSearchOutput,
    ContextDetail
)

@pytest.fixture
async def mock_zendesk_agent() -> AsyncMock:
    """Provides an AsyncMock for ZendeskAgent."""
    mock = AsyncMock(spec=ZendeskAgent)
    mock.search_tickets = AsyncMock()
    mock.close = AsyncMock() # Ensure close is also an AsyncMock
    return mock

@pytest.fixture
async def mock_confluence_agent() -> AsyncMock:
    """Provides an AsyncMock for ConfluenceAgent."""
    mock = AsyncMock(spec=ConfluenceAgent)
    mock.search_runbooks = AsyncMock()
    mock.close = AsyncMock() # Ensure close is also an AsyncMock
    return mock

@pytest.fixture
async def request_handler_agent_fixture(
    mock_zendesk_agent: AsyncMock,
    mock_confluence_agent: AsyncMock
) -> RequestHandlerAgent:
    """
    Initializes and yields a RequestHandlerAgent instance with mocked sub-agents.
    Handles agent.close() in teardown.
    """
    # Note: The ADK Agent base class might take **kwargs, which could include 'name'.
    # If not specified, a default name is often generated.
    agent = RequestHandlerAgent(
        zendesk_agent_instance=mock_zendesk_agent,
        confluence_agent_instance=mock_confluence_agent,
        name="TestRequestHandlerAgent" # Explicitly naming for predictability
    )
    yield agent
    # Teardown: close the agent, which should trigger close on mocked sub-agents
    await agent.close()


# --- Test Data ---
SAMPLE_QUERY = "Help with login"
ZENDESK_CONTEXT_1 = ContextDetail(text="Zendesk ticket 1 snippet", source_collection="zendesk", ticket_id="ZD001", title="Login issue ZD001")
ZENDESK_CONTEXT_2 = ContextDetail(text="Zendesk ticket 2 snippet", source_collection="zendesk", ticket_id="ZD002", title="Cannot log in ZD002")
CONFLUENCE_CONTEXT_1 = ContextDetail(text="Confluence runbook snippet for login", source_collection="confluence_runbooks", title="Runbook: Login Problems")

@pytest.mark.asyncio
async def test_process_success_both_agents(
    request_handler_agent_fixture: RequestHandlerAgent,
    mock_zendesk_agent: AsyncMock,
    mock_confluence_agent: AsyncMock
):
    query_input = QueryInput(query=SAMPLE_QUERY)

    mock_zendesk_agent.search_tickets.return_value = ZendeskSearchOutput(
        query=SAMPLE_QUERY, results=[ZENDESK_CONTEXT_1, ZENDESK_CONTEXT_2]
    )
    mock_confluence_agent.search_runbooks.return_value = ConfluenceSearchOutput(
        query=SAMPLE_QUERY, results=[CONFLUENCE_CONTEXT_1]
    )

    result_input = await request_handler_agent_fixture.process(query_input)

    mock_zendesk_agent.search_tickets.assert_called_once_with(query_input)
    mock_confluence_agent.search_runbooks.assert_called_once_with(query_input)

    assert isinstance(result_input, SynthesizerInput)
    assert result_input.original_query == SAMPLE_QUERY
    assert len(result_input.zendesk_search_results) == 2
    assert ZENDESK_CONTEXT_1 in result_input.zendesk_search_results
    assert ZENDESK_CONTEXT_2 in result_input.zendesk_search_results
    assert len(result_input.confluence_search_results) == 1
    assert CONFLUENCE_CONTEXT_1 in result_input.confluence_search_results
    assert result_input.zendesk_analyses is None

@pytest.mark.asyncio
async def test_process_zendesk_fails_confluence_succeeds(
    request_handler_agent_fixture: RequestHandlerAgent,
    mock_zendesk_agent: AsyncMock,
    mock_confluence_agent: AsyncMock
):
    query_input = QueryInput(query=SAMPLE_QUERY)
    mock_zendesk_agent.search_tickets.side_effect = Exception("Zendesk API Error")
    mock_confluence_agent.search_runbooks.return_value = ConfluenceSearchOutput(
        query=SAMPLE_QUERY, results=[CONFLUENCE_CONTEXT_1]
    )

    result_input = await request_handler_agent_fixture.process(query_input)

    mock_zendesk_agent.search_tickets.assert_called_once_with(query_input)
    mock_confluence_agent.search_runbooks.assert_called_once_with(query_input)

    assert result_input.original_query == SAMPLE_QUERY
    assert len(result_input.zendesk_search_results) == 0 # Should be empty list
    assert len(result_input.confluence_search_results) == 1
    assert CONFLUENCE_CONTEXT_1 in result_input.confluence_search_results

@pytest.mark.asyncio
async def test_process_confluence_fails_zendesk_succeeds(
    request_handler_agent_fixture: RequestHandlerAgent,
    mock_zendesk_agent: AsyncMock,
    mock_confluence_agent: AsyncMock
):
    query_input = QueryInput(query=SAMPLE_QUERY)
    mock_zendesk_agent.search_tickets.return_value = ZendeskSearchOutput(
        query=SAMPLE_QUERY, results=[ZENDESK_CONTEXT_1]
    )
    mock_confluence_agent.search_runbooks.side_effect = Exception("Confluence API Error")

    result_input = await request_handler_agent_fixture.process(query_input)

    mock_zendesk_agent.search_tickets.assert_called_once_with(query_input)
    mock_confluence_agent.search_runbooks.assert_called_once_with(query_input)

    assert result_input.original_query == SAMPLE_QUERY
    assert len(result_input.zendesk_search_results) == 1
    assert ZENDESK_CONTEXT_1 in result_input.zendesk_search_results
    assert len(result_input.confluence_search_results) == 0 # Should be empty list

@pytest.mark.asyncio
async def test_process_both_agents_fail(
    request_handler_agent_fixture: RequestHandlerAgent,
    mock_zendesk_agent: AsyncMock,
    mock_confluence_agent: AsyncMock
):
    query_input = QueryInput(query=SAMPLE_QUERY)
    mock_zendesk_agent.search_tickets.side_effect = Exception("Zendesk API Error")
    mock_confluence_agent.search_runbooks.side_effect = Exception("Confluence API Error")

    result_input = await request_handler_agent_fixture.process(query_input)

    mock_zendesk_agent.search_tickets.assert_called_once_with(query_input)
    mock_confluence_agent.search_runbooks.assert_called_once_with(query_input)

    assert result_input.original_query == SAMPLE_QUERY
    assert len(result_input.zendesk_search_results) == 0
    assert len(result_input.confluence_search_results) == 0

@pytest.mark.asyncio
async def test_close_method_calls_child_agents_close(
    request_handler_agent_fixture: RequestHandlerAgent, # Fixture will call close on teardown
    mock_zendesk_agent: AsyncMock,
    mock_confluence_agent: AsyncMock
):
    # The request_handler_agent_fixture automatically calls agent.close() during its teardown.
    # So, we just need to assert that the mocks' close methods were called by the time the test finishes.
    # To be absolutely explicit and test the close method directly:
    await request_handler_agent_fixture.close() # Call it again or rely on fixture's call

    mock_zendesk_agent.close.assert_called() # Use assert_called() if it might be called multiple times by fixture + direct call
    mock_confluence_agent.close.assert_called()

    # If fixture ensures it's called once and we don't call it again, then:
    # mock_zendesk_agent.close.assert_called_once()
    # mock_confluence_agent.close.assert_called_once()
    # Given the fixture structure, assert_called() is safer if the test itself also calls close().
    # The fixture calls `close` once. If this test also calls `close` then it's twice.
    # For this test, let's assume the fixture handles the call, and we verify post-fixture.
    # This means the assertions would ideally be outside the fixture's scope, which is tricky.
    # So, an explicit call within the test is clearer for verifying the specific action.
    # The fixture will call it *again* at teardown. AsyncMock is fine with multiple calls.

@pytest.mark.asyncio
async def test_agent_initialization_properties(
    request_handler_agent_fixture: RequestHandlerAgent,
    mock_zendesk_agent: AsyncMock,
    mock_confluence_agent: AsyncMock
):
    agent = request_handler_agent_fixture
    assert agent.name == "TestRequestHandlerAgent" # Name set in fixture
    assert agent.zendesk_agent == mock_zendesk_agent
    assert agent.confluence_agent == mock_confluence_agent

    # Check that the agent has no tools registered
    # The `tools` property of adk.Agent returns a list of tool values from `_tools`
    assert list(agent.tools) == []
    # Alternatively, and more directly for ADK internals:
    assert agent._tools == {}

# To run: pytest test_request_handler_agent.py
# Ensure zendesk_agent.py, confluence_agent.py, agent_schemas.py are in PYTHONPATH
# And google.adk is available.
