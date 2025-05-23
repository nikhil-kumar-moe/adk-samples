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

from confluence_agent import ConfluenceAgent
from agent_schemas import (
    QueryInput,
    ConfluenceSearchOutput,
    ContextDetail,
    confluence_search_runbooks_tool_schema
)

# Placeholder for EmbeddingService type
EmbeddingService = Any

# --- Helper for Mock ScoredPoint ---
def create_mock_scored_point(id_val: Any, score_val: float, payload_val: Dict[str, Any], version_val: int = 1) -> MagicMock:
    """Creates a mock ScoredPoint object."""
    # If ScoredPoint was imported and is a class, MagicMock(spec=ScoredPoint) is better.
    # If it's Any, or for simplicity in environments where it might not be fully available:
    mock_point = MagicMock()
    mock_point.id = id_val
    mock_point.version = version_val
    mock_point.score = score_val
    mock_point.payload = payload_val
    return mock_point

# --- Fixtures ---
@pytest.fixture
def mock_embedding_service() -> AsyncMock:
    """Provides a mock EmbeddingService."""
    service = AsyncMock(spec=EmbeddingService)
    service.generate_embeddings = AsyncMock()
    return service

@pytest.fixture
def mock_qdrant_client() -> AsyncMock:
    """Provides a mock QdrantClient."""
    client = AsyncMock(spec=QdrantClient) # Use spec if QdrantClient is imported
    client.search = AsyncMock()
    return client

@pytest.fixture
async def confluence_agent_fixture(
    mock_embedding_service: AsyncMock,
    mock_qdrant_client: AsyncMock
) -> ConfluenceAgent:
    """
    Initializes and yields a ConfluenceAgent instance with mocked dependencies.
    Handles agent.close() in teardown.
    """
    test_collection_name = "test_confluence_docs"
    agent = ConfluenceAgent(
        model_name="test-model", # Retained, not primary focus of these tests
        embedding_service=mock_embedding_service,
        qdrant_client=mock_qdrant_client,
        confluence_collection_name=test_collection_name
    )
    yield agent
    await agent.close() # Agent's close method is simple logging

# --- Initialization Tests ---
def test_initialization_valid(
    mock_embedding_service: AsyncMock,
    mock_qdrant_client: AsyncMock
):
    """Tests successful initialization of ConfluenceAgent."""
    agent = ConfluenceAgent(
        embedding_service=mock_embedding_service,
        qdrant_client=mock_qdrant_client,
        confluence_collection_name="valid_collection"
    )
    assert agent.embedding_service == mock_embedding_service
    assert agent.qdrant_client == mock_qdrant_client
    assert agent.confluence_collection_name == "valid_collection"

def test_initialization_missing_dependencies():
    """Tests that ValueError is raised if dependencies are missing."""
    with pytest.raises(ValueError, match="embedding_service is required"):
        ConfluenceAgent(embedding_service=None, qdrant_client=AsyncMock(), confluence_collection_name="c")
    with pytest.raises(ValueError, match="qdrant_client is required"):
        ConfluenceAgent(embedding_service=AsyncMock(), qdrant_client=None, confluence_collection_name="c")
    with pytest.raises(ValueError, match="confluence_collection_name is required"):
        ConfluenceAgent(embedding_service=AsyncMock(), qdrant_client=AsyncMock(), confluence_collection_name="")

@pytest.mark.asyncio
async def test_agent_tool_registration_points_to_process(confluence_agent_fixture: ConfluenceAgent):
    """Tests that the 'search_confluence_runbooks' tool is registered and points to agent.process."""
    agent = confluence_agent_fixture
    assert "search_confluence_runbooks" in agent.tools
    tool = agent.tools["search_confluence_runbooks"]
    assert tool.fn == agent.process # Tool should now call process
    assert tool.schema_dict == confluence_search_runbooks_tool_schema


# --- search_runbooks Method Tests ---
@pytest.mark.asyncio
async def test_search_runbooks_success(
    confluence_agent_fixture: ConfluenceAgent,
    mock_embedding_service: AsyncMock,
    mock_qdrant_client: AsyncMock
):
    agent = confluence_agent_fixture
    query_text = "how to fix API errors"
    query_input = QueryInput(query=query_text)
    top_k_val = 3
    mock_embedding_vector = [0.1, 0.2, 0.3, 0.4, 0.5]

    mock_embedding_service.generate_embeddings.return_value = [mock_embedding_vector]
    
    mock_qdrant_hits = [
        create_mock_scored_point(id_val="doc1", score_val=0.95, payload_val={"text": "API error fix steps...", "title": "Fix API Errors", "url": "/fix-api"}),
        create_mock_scored_point(id_val="doc2", score_val=0.90, payload_val={"text": "Troubleshooting guide...", "title": "API Troubleshooting", "url": "/api-troubleshoot"})
    ]
    mock_qdrant_client.search.return_value = mock_qdrant_hits

    result_output = await agent.search_runbooks(query_input, top_k=top_k_val)

    mock_embedding_service.generate_embeddings.assert_called_once_with([query_text])
    mock_qdrant_client.search.assert_called_once_with(
        collection_name=agent.confluence_collection_name,
        query_vector=mock_embedding_vector,
        limit=top_k_val,
        with_payload=True
    )
    
    assert isinstance(result_output, ConfluenceSearchOutput)
    assert result_output.query == query_text
    assert len(result_output.results) == 2
    first_res_detail = result_output.results[0]
    assert isinstance(first_res_detail, ContextDetail)
    assert first_res_detail.text == "API error fix steps..."
    assert first_res_detail.title == "Fix API Errors"
    assert first_res_detail.url == "/fix-api"
    assert first_res_detail.score == 0.95
    assert first_res_detail.source_collection == agent.confluence_collection_name

@pytest.mark.asyncio
async def test_search_runbooks_embedding_fails(
    confluence_agent_fixture: ConfluenceAgent,
    mock_embedding_service: AsyncMock
):
    agent = confluence_agent_fixture
    query_input = QueryInput(query="test query")
    
    mock_embedding_service.generate_embeddings.return_value = None # Simulate embedding failure (empty list or None)
    
    result_output = await agent.search_runbooks(query_input, top_k=5)
    assert len(result_output.results) == 0

    mock_embedding_service.generate_embeddings.side_effect = Exception("Embedding service error")
    result_output_exc = await agent.search_runbooks(query_input, top_k=5)
    assert len(result_output_exc.results) == 0


@pytest.mark.asyncio
async def test_search_runbooks_qdrant_search_fails(
    confluence_agent_fixture: ConfluenceAgent,
    mock_embedding_service: AsyncMock,
    mock_qdrant_client: AsyncMock
):
    agent = confluence_agent_fixture
    query_input = QueryInput(query="test query")
    mock_embedding_service.generate_embeddings.return_value = [[0.1, 0.2]]
    mock_qdrant_client.search.side_effect = Exception("Qdrant DB error")
    
    result_output = await agent.search_runbooks(query_input, top_k=5)
    assert len(result_output.results) == 0

@pytest.mark.asyncio
async def test_search_runbooks_no_qdrant_hits(
    confluence_agent_fixture: ConfluenceAgent,
    mock_embedding_service: AsyncMock,
    mock_qdrant_client: AsyncMock
):
    agent = confluence_agent_fixture
    query_input = QueryInput(query="test query")
    mock_embedding_service.generate_embeddings.return_value = [[0.1, 0.2]]
    mock_qdrant_client.search.return_value = [] # No hits from Qdrant
    
    result_output = await agent.search_runbooks(query_input, top_k=5)
    assert len(result_output.results) == 0


# --- Process Method Tests ---
@pytest.mark.asyncio
async def test_process_method_success(
    confluence_agent_fixture: ConfluenceAgent,
    mock_embedding_service: AsyncMock,
    mock_qdrant_client: AsyncMock
):
    """Tests the process method which calls search_runbooks internally."""
    agent = confluence_agent_fixture
    query_text = "process this query"
    top_k_val = 2
    input_data = {"query": query_text, "top_k": top_k_val}
    
    mock_embedding_vector = [0.3, 0.4, 0.5]
    mock_embedding_service.generate_embeddings.return_value = [mock_embedding_vector]
    
    mock_qdrant_hits = [
        create_mock_scored_point(id_val="proc_doc1", score_val=0.88, payload_val={"text": "Processed content...", "title": "Processed Doc", "url": "/proc/doc"})
    ]
    mock_qdrant_client.search.return_value = mock_qdrant_hits
    
    result_dict = await agent.process(input_data)
    
    # Check that underlying services were called correctly via search_runbooks
    mock_embedding_service.generate_embeddings.assert_called_once_with([query_text])
    mock_qdrant_client.search.assert_called_once_with(
        collection_name=agent.confluence_collection_name,
        query_vector=mock_embedding_vector,
        limit=top_k_val,
        with_payload=True
    )
    
    assert isinstance(result_dict, dict) # process returns a dict (model_dump)
    assert result_dict["query"] == query_text
    assert len(result_dict["results"]) == 1
    assert result_dict["results"][0]["title"] == "Processed Doc"

@pytest.mark.asyncio
async def test_process_method_missing_query(confluence_agent_fixture: ConfluenceAgent):
    """Tests the process method when 'query' is missing from input_data."""
    agent = confluence_agent_fixture
    input_data = {"top_k": 3} # Missing 'query'
    
    # Expected error structure as defined in ConfluenceAgent.process
    expected_error_response = {"error": "Invalid input. 'query' is required."}
    
    result_dict = await agent.process(input_data)
    
    assert result_dict == expected_error_response

@pytest.mark.asyncio
async def test_process_method_default_top_k(
    confluence_agent_fixture: ConfluenceAgent,
    mock_embedding_service: AsyncMock,
    mock_qdrant_client: AsyncMock
):
    """Tests that process method uses default top_k if not provided."""
    agent = confluence_agent_fixture
    query_text = "query with default top_k"
    input_data = {"query": query_text} # top_k not provided, should default to 5

    mock_embedding_service.generate_embeddings.return_value = [[0.5, 0.6]]
    mock_qdrant_client.search.return_value = [] # Don't care about results, just the call

    await agent.process(input_data)

    mock_qdrant_client.search.assert_called_once()
    # Check if limit argument in the call to qdrant_client.search was 5 (the default)
    args, kwargs = mock_qdrant_client.search.call_args
    assert kwargs.get("limit") == 5


# --- Close Method Test ---
@pytest.mark.asyncio
async def test_close_method(confluence_agent_fixture: ConfluenceAgent):
    """Tests that the close method runs without error (it's just logging)."""
    agent = confluence_agent_fixture
    try:
        await agent.close() # Already called by fixture, but call again for explicitness
    except Exception as e:
        pytest.fail(f"ConfluenceAgent.close() raised an exception: {e}")

# To run: pytest test_confluence_agent.py
# Ensure confluence_agent.py and agent_schemas.py are accessible.
# And necessary libraries (pytest, qdrant-client if using real spec) are installed.
