import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch # For mocking
from typing import Any, List, Dict, Optional

# Attempt to import Qdrant specifics for spec, fallback to Any or manual mock if not available
try:
    from qdrant_client import QdrantClient # For spec
    from qdrant_client.models import ScoredPoint # For spec and creating mock results
except ImportError:
    QdrantClient = Any # Fallback if qdrant_client is not installed in test env
    ScoredPoint = Any  # Fallback

import httpx # For type hinting AsyncClient

from zendesk_agent import ZendeskAgent
from agent_schemas import (
    QueryInput,
    ZendeskSearchOutput,
    ZendeskAnalyzeInput,
    ZendeskAnalysisOutput,
    ZendeskAddCommentInput,
    ZendeskAddCommentOutput,
    ContextDetail,
    zendesk_search_tickets_tool_schema,
    zendesk_analyze_ticket_conversation_tool_schema,
    zendesk_add_comment_tool_schema
)

# Placeholder for EmbeddingService type
EmbeddingService = Any

# --- Helper for Mock ScoredPoint ---
def create_mock_scored_point(id_val: Any, score_val: float, payload_val: Dict[str, Any], version_val: int = 1) -> MagicMock:
    """Creates a mock ScoredPoint object."""
    mock_point = MagicMock()
    mock_point.id = id_val
    mock_point.version = version_val
    mock_point.score = score_val
    mock_point.payload = payload_val
    return mock_point

# --- Fixtures ---
@pytest.fixture
def zendesk_config_fixture(): # Renamed to avoid potential conflicts if other files use 'zendesk_config'
    """Provides mock Zendesk configuration."""
    return {
        "domain": "https://testcompany.zendesk.com",
        "api_key": "test_api_key", # Though not directly used by Qdrant search, part of config
        "email": "test_agent@example.com" # Same as above
    }

@pytest.fixture
async def http_client_mock_fixture(): # Renamed for clarity
    """Provides a mock httpx.AsyncClient."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    return mock_client

@pytest.fixture
def mock_embedding_service() -> AsyncMock:
    """Provides a mock EmbeddingService."""
    service = AsyncMock(spec=EmbeddingService)
    service.generate_embeddings = AsyncMock()
    return service

@pytest.fixture
def mock_qdrant_client() -> AsyncMock:
    """Provides a mock QdrantClient."""
    client = AsyncMock(spec=QdrantClient)
    client.search = AsyncMock()
    return client

@pytest.fixture
async def zendesk_agent_fixture(
    mock_embedding_service: AsyncMock,
    mock_qdrant_client: AsyncMock,
    zendesk_config_fixture: Dict[str, Any],
    http_client_mock_fixture: AsyncMock
) -> ZendeskAgent:
    """
    Initializes and yields a ZendeskAgent instance with mocked dependencies.
    Handles agent.close() in teardown.
    """
    agent = ZendeskAgent(
        model_name="test-qdrant-model",
        embedding_service=mock_embedding_service,
        qdrant_client=mock_qdrant_client,
        zendesk_collection_name="test_zendesk_docs_collection",
        zendesk_config=zendesk_config_fixture,
        httpx_client=http_client_mock_fixture
    )
    yield agent
    await agent.close()

# --- Initialization Tests ---
def test_initialization_valid(
    mock_embedding_service: AsyncMock,
    mock_qdrant_client: AsyncMock,
    zendesk_config_fixture: Dict[str, Any],
    http_client_mock_fixture: AsyncMock
):
    """Tests successful initialization of ZendeskAgent."""
    agent = ZendeskAgent(
        embedding_service=mock_embedding_service,
        qdrant_client=mock_qdrant_client,
        zendesk_collection_name="valid_collection",
        zendesk_config=zendesk_config_fixture,
        httpx_client=http_client_mock_fixture
    )
    assert agent.embedding_service == mock_embedding_service
    assert agent.qdrant_client == mock_qdrant_client
    assert agent.zendesk_collection_name == "valid_collection"
    assert agent.zendesk_config == zendesk_config_fixture

def test_initialization_missing_dependencies(zendesk_config_fixture, http_client_mock_fixture):
    """Tests that ValueError is raised if dependencies are missing."""
    with pytest.raises(ValueError, match="embedding_service is required"):
        ZendeskAgent(embedding_service=None, qdrant_client=AsyncMock(), zendesk_collection_name="c", zendesk_config=zendesk_config_fixture, httpx_client=http_client_mock_fixture)
    with pytest.raises(ValueError, match="qdrant_client is required"):
        ZendeskAgent(embedding_service=AsyncMock(), qdrant_client=None, zendesk_collection_name="c", zendesk_config=zendesk_config_fixture, httpx_client=http_client_mock_fixture)
    with pytest.raises(ValueError, match="zendesk_collection_name is required"):
        ZendeskAgent(embedding_service=AsyncMock(), qdrant_client=AsyncMock(), zendesk_collection_name="", zendesk_config=zendesk_config_fixture, httpx_client=http_client_mock_fixture)
    with pytest.raises(ValueError, match="zendesk_config is required"):
        ZendeskAgent(embedding_service=AsyncMock(), qdrant_client=AsyncMock(), zendesk_collection_name="c", zendesk_config=None, httpx_client=http_client_mock_fixture)


@pytest.mark.asyncio
async def test_tool_registration(zendesk_agent_fixture: ZendeskAgent):
    """Tests correct tool registration."""
    agent = zendesk_agent_fixture
    assert "search_zendesk_tickets" in agent.tools
    assert agent.tools["search_zendesk_tickets"].fn == agent.search_tickets
    assert agent.tools["search_zendesk_tickets"].schema_dict == zendesk_search_tickets_tool_schema

    assert "analyze_zendesk_ticket_conversation" in agent.tools
    assert agent.tools["analyze_zendesk_ticket_conversation"].fn == agent.analyze_ticket_conversation
    assert agent.tools["analyze_zendesk_ticket_conversation"].schema_dict == zendesk_analyze_ticket_conversation_tool_schema
    
    assert "add_comment_to_zendesk_ticket" in agent.tools
    assert agent.tools["add_comment_to_zendesk_ticket"].fn == agent.add_comment_to_ticket
    assert agent.tools["add_comment_to_zendesk_ticket"].schema_dict == zendesk_add_comment_tool_schema


# --- search_tickets Method Tests (Qdrant Implementation) ---
@pytest.mark.asyncio
async def test_search_tickets_success(
    zendesk_agent_fixture: ZendeskAgent,
    mock_embedding_service: AsyncMock,
    mock_qdrant_client: AsyncMock,
    zendesk_config_fixture: Dict[str, Any]
):
    agent = zendesk_agent_fixture
    query_text = "login issue"
    query_input = QueryInput(query=query_text)
    top_k_val = 3
    mock_embedding_vector = [0.1, 0.2, 0.3]

    mock_embedding_service.generate_embeddings.return_value = [mock_embedding_vector]
    
    mock_qdrant_hits = [
        create_mock_scored_point(id_val="q_doc1", score_val=0.95, payload_val={"text": "Login fix steps for ticket 789...", "ticket_id": "789", "title": "Fix Login ZD789"}),
        create_mock_scored_point(id_val="q_doc2", score_val=0.90, payload_val={"text": "Password reset info for 101...", "ticket_id": "101", "title": "Reset Pass ZD101", "url": "http://custom.url/101"}) # Custom URL in payload
    ]
    mock_qdrant_client.search.return_value = mock_qdrant_hits

    result_output = await agent.search_tickets(query_input, top_k=top_k_val)

    mock_embedding_service.generate_embeddings.assert_called_once_with([query_text])
    mock_qdrant_client.search.assert_called_once_with(
        collection_name=agent.zendesk_collection_name,
        query_vector=mock_embedding_vector,
        limit=top_k_val,
        with_payload=True
    )
    
    assert isinstance(result_output, ZendeskSearchOutput)
    assert result_output.query == query_text
    assert len(result_output.results) == 2
    
    res1_detail = result_output.results[0]
    assert res1_detail.text == "Login fix steps for ticket 789..."
    assert res1_detail.title == "Fix Login ZD789"
    assert res1_detail.ticket_id == "789"
    assert res1_detail.url == f"{zendesk_config_fixture['domain']}/agent/tickets/789" # Constructed URL
    assert res1_detail.score == 0.95
    assert res1_detail.source_collection == agent.zendesk_collection_name

    res2_detail = result_output.results[1]
    assert res2_detail.ticket_id == "101"
    assert res2_detail.url == "http://custom.url/101" # URL from payload preferred if zendesk_config['domain'] is not set, or if payload URL exists. Logic is: constructed if possible, else payload's.
                                                       # Current agent logic: always constructs if domain and ticket_id exist. This test has domain, so it constructs.
                                                       # Let's adjust expectation based on agent code:
    assert res2_detail.url == f"{zendesk_config_fixture['domain']}/agent/tickets/101"


@pytest.mark.asyncio
async def test_search_tickets_top_k_parameter(
    zendesk_agent_fixture: ZendeskAgent,
    mock_embedding_service: AsyncMock,
    mock_qdrant_client: AsyncMock
):
    agent = zendesk_agent_fixture
    query_input = QueryInput(query="test top k")
    # Test default top_k
    mock_embedding_service.generate_embeddings.return_value = [[0.1]]
    mock_qdrant_client.search.return_value = []
    await agent.search_tickets(query_input)
    args, kwargs = mock_qdrant_client.search.call_args
    assert kwargs.get("limit") == 5 # Default top_k

    # Test specified top_k
    await agent.search_tickets(query_input, top_k=10)
    args, kwargs = mock_qdrant_client.search.call_args
    assert kwargs.get("limit") == 10


@pytest.mark.asyncio
async def test_search_tickets_embedding_fails(zendesk_agent_fixture: ZendeskAgent, mock_embedding_service: AsyncMock):
    agent = zendesk_agent_fixture
    mock_embedding_service.generate_embeddings.return_value = None
    output = await agent.search_tickets(QueryInput(query="q"))
    assert len(output.results) == 0

    mock_embedding_service.generate_embeddings.side_effect = Exception("Embedding error")
    output = await agent.search_tickets(QueryInput(query="q"))
    assert len(output.results) == 0

@pytest.mark.asyncio
async def test_search_tickets_qdrant_fails(zendesk_agent_fixture: ZendeskAgent, mock_embedding_service: AsyncMock, mock_qdrant_client: AsyncMock):
    agent = zendesk_agent_fixture
    mock_embedding_service.generate_embeddings.return_value = [[0.1]]
    mock_qdrant_client.search.side_effect = Exception("Qdrant error")
    output = await agent.search_tickets(QueryInput(query="q"))
    assert len(output.results) == 0

@pytest.mark.asyncio
async def test_search_tickets_no_hits(zendesk_agent_fixture: ZendeskAgent, mock_embedding_service: AsyncMock, mock_qdrant_client: AsyncMock):
    agent = zendesk_agent_fixture
    mock_embedding_service.generate_embeddings.return_value = [[0.1]]
    mock_qdrant_client.search.return_value = []
    output = await agent.search_tickets(QueryInput(query="q"))
    assert len(output.results) == 0


# --- Tests for Unchanged Placeholder Methods ---
@pytest.mark.asyncio
async def test_analyze_ticket_conversation_placeholder(zendesk_agent_fixture: ZendeskAgent):
    """Tests placeholder execution of analyze_ticket_conversation."""
    agent = zendesk_agent_fixture
    ticket_id = "ZD78901"
    analysis_input = ZendeskAnalyzeInput(ticket_id=ticket_id)
    result = await agent.analyze_ticket_conversation(analysis_input)
    assert isinstance(result, ZendeskAnalysisOutput)
    assert result.ticket_id == ticket_id
    assert "The user is unable to log in" in result.ai_analysis

@pytest.mark.asyncio
async def test_add_comment_to_ticket_placeholder(zendesk_agent_fixture: ZendeskAgent):
    """Tests placeholder execution of add_comment_to_ticket."""
    agent = zendesk_agent_fixture
    ticket_id = "ZD54321"
    comment_body = "This is a test comment."
    comment_input = ZendeskAddCommentInput(ticket_id=ticket_id, comment_body=comment_body)
    result = await agent.add_comment_to_ticket(comment_input)
    assert isinstance(result, ZendeskAddCommentOutput)
    assert result.ticket_id == ticket_id
    assert "Comment added successfully" in result.status

# --- Process Method Dispatcher Tests ---
@pytest.mark.asyncio
async def test_process_dispatch_search_tickets(zendesk_agent_fixture: ZendeskAgent):
    agent = zendesk_agent_fixture
    query = "find my ticket"
    top_k = 7
    tool_input = {"query": query, "top_k": top_k}
    
    # Mock the actual search_tickets method to isolate testing of process dispatch logic
    mock_search_output = ZendeskSearchOutput(query=query, results=[])
    with patch.object(agent, 'search_tickets', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_search_output
        
        result_dict = await agent.process(tool_to_call="search_zendesk_tickets", tool_input_dict=tool_input)
        
        mock_search.assert_called_once()
        call_args = mock_search.call_args[0]
        assert isinstance(call_args[0], QueryInput)
        assert call_args[0].query == query
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs.get('top_k') == top_k
        assert result_dict == mock_search_output.model_dump()

@pytest.mark.asyncio
async def test_process_dispatch_analyze_ticket(zendesk_agent_fixture: ZendeskAgent):
    agent = zendesk_agent_fixture
    ticket_id = "ticket_for_analysis"
    tool_input = {"ticket_id": ticket_id}
    mock_analyze_output = ZendeskAnalysisOutput(ticket_id=ticket_id, ai_analysis="analysis done")

    with patch.object(agent, 'analyze_ticket_conversation', new_callable=AsyncMock) as mock_analyze:
        mock_analyze.return_value = mock_analyze_output
        result_dict = await agent.process(tool_to_call="analyze_zendesk_ticket_conversation", tool_input_dict=tool_input)
        
        mock_analyze.assert_called_once_with(ZendeskAnalyzeInput(ticket_id=ticket_id))
        assert result_dict == mock_analyze_output.model_dump()

@pytest.mark.asyncio
async def test_process_dispatch_add_comment(zendesk_agent_fixture: ZendeskAgent):
    agent = zendesk_agent_fixture
    ticket_id = "ticket_for_comment"
    comment = "new comment text"
    tool_input = {"ticket_id": ticket_id, "comment_body": comment}
    mock_comment_output = ZendeskAddCommentOutput(ticket_id=ticket_id, status="comment added", comment_id="cmt1")

    with patch.object(agent, 'add_comment_to_ticket', new_callable=AsyncMock) as mock_add_comment:
        mock_add_comment.return_value = mock_comment_output
        result_dict = await agent.process(tool_to_call="add_comment_to_zendesk_ticket", tool_input_dict=tool_input)

        mock_add_comment.assert_called_once_with(ZendeskAddCommentInput(ticket_id=ticket_id, comment_body=comment))
        assert result_dict == mock_comment_output.model_dump()

@pytest.mark.asyncio
async def test_process_missing_params(zendesk_agent_fixture: ZendeskAgent):
    agent = zendesk_agent_fixture
    # Search tickets missing query
    error_res = await agent.process("search_zendesk_tickets", {"top_k": 1})
    assert "error" in error_res
    assert "Missing 'query'" in error_res["error"]

    # Analyze missing ticket_id
    error_res = await agent.process("analyze_zendesk_ticket_conversation", {})
    assert "error" in error_res
    assert "Missing 'ticket_id'" in error_res["error"]

    # Add comment missing ticket_id
    error_res = await agent.process("add_comment_to_zendesk_ticket", {"comment_body": "c"})
    assert "error" in error_res
    assert "Missing 'ticket_id'" in error_res["error"]
    
    # Add comment missing comment_body
    error_res = await agent.process("add_comment_to_zendesk_ticket", {"ticket_id": "t"})
    assert "error" in error_res
    assert "Missing 'comment_body'" in error_res["error"]


@pytest.mark.asyncio
async def test_process_unknown_tool(zendesk_agent_fixture: ZendeskAgent):
    agent = zendesk_agent_fixture
    error_res = await agent.process("non_existent_tool", {})
    assert "error" in error_res
    assert "Unknown tool_to_call" in error_res["error"]

# --- Close Method Test ---
@pytest.mark.asyncio
async def test_close_method(zendesk_agent_fixture: ZendeskAgent, http_client_mock_fixture: AsyncMock):
    """Tests that the close method calls httpx_client.aclose()."""
    agent = zendesk_agent_fixture
    await agent.close() # Called by fixture, but call again for direct test
    http_client_mock_fixture.aclose.assert_called() # Check if client's aclose was called


# To run: pytest test_zendesk_agent.py
# Ensure zendesk_agent.py and agent_schemas.py are accessible.
# And necessary libraries (pytest, qdrant-client if using real spec) are installed.
