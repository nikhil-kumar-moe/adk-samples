import pytest
import asyncio
import os
from unittest.mock import patch, AsyncMock, MagicMock # Added MagicMock
from typing import Any, List, Dict, Optional # Added for type hints

import httpx # For AsyncClient spec

# Attempt to import Qdrant specifics for spec, fallback to Any or manual mock if not available
try:
    from qdrant_client import QdrantClient # For spec
    from qdrant_client.models import ScoredPoint # For spec and creating mock results
except ImportError:
    QdrantClient = Any # Fallback if qdrant_client is not installed in test env
    ScoredPoint = Any  # Fallback


# Import agent classes
from zendesk_agent import ZendeskAgent
from confluence_agent import ConfluenceAgent
from request_handler_agent import RequestHandlerAgent
from synthesizing_agent import SynthesizingAgent, ACTION_SUGGESTION_MARKER, ACTION_DETAILS_MARKER

# Import Pydantic schemas
from agent_schemas import (
    QueryInput,
    SynthesizerInput, 
    SynthesizerOutput,
    ContextDetail
)

# Placeholder for EmbeddingService type
EmbeddingService = Any

# --- Helper for Mock ScoredPoint ---
def create_mock_scored_point(id_val: Any, score_val: float, payload_val: Dict[str, Any], version_val: int = 1) -> MagicMock:
    """Creates a mock ScoredPoint object."""
    mock_point = MagicMock()
    if ScoredPoint is not Any and ScoredPoint is not None : # pragma: no cover
        # If ScoredPoint is actually available, use it for spec for better type checking in tests
        # However, directly instantiating ScoredPoint might require specific fields not needed for all mocks.
        # For simplicity and to avoid strict Qdrant model validation in tests where only payload/score matter:
        mock_point = MagicMock(spec=ScoredPoint) # This helps if attributes are misspelled
    
    mock_point.id = id_val
    mock_point.version = version_val # Qdrant ScoredPoint has this
    mock_point.score = score_val
    mock_point.payload = payload_val
    return mock_point


@pytest.fixture
async def integrated_agents(
    # Mocks will be created inside the fixture to allow per-test configuration if needed,
    # or default configurations as done here.
):
    """
    Initializes and yields a dictionary of all four agents for integration testing.
    Sets a dummy GOOGLE_API_KEY for SynthesizingAgent default initialization.
    Handles closing of all agents during teardown.
    """
    mock_embedding_service = AsyncMock(spec=EmbeddingService)
    # Default behavior for generate_embeddings, can be overridden in tests if specific vectors are needed
    mock_embedding_service.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3, 0.4, 0.5]]) # Must return list of lists
    
    mock_qdrant_client = AsyncMock(spec=QdrantClient)
    # mock_qdrant_client.search will be configured per-test using side_effect typically

    mock_httpx_client = AsyncMock(spec=httpx.AsyncClient) # For ZendeskAgent

    zd_collection_name = "test_zd_qdrant_collection"
    cf_collection_name = "test_cf_qdrant_collection"
    
    # Patch os.environ for the duration of this fixture setup
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_integration_key_dummy"}):
        zd_agent = ZendeskAgent(
            embedding_service=mock_embedding_service,
            qdrant_client=mock_qdrant_client,
            zendesk_collection_name=zd_collection_name,
            zendesk_config={"domain": "https://testintegrationsubdomain.zendesk.com"}, # Ensure 'domain' is present
            httpx_client=mock_httpx_client,
            model_name="zd_test_integr_model"
        )
        cf_agent = ConfluenceAgent(
            embedding_service=mock_embedding_service,
            qdrant_client=mock_qdrant_client,
            confluence_collection_name=cf_collection_name,
            model_name="cf_test_integr_model"
        )
        rh_agent = RequestHandlerAgent(
            zendesk_agent_instance=zd_agent,
            confluence_agent_instance=cf_agent,
            name="TestIntegrationRequestHandler"
        )
        synth_agent = SynthesizingAgent(model_name="gemini-1.5-flash-test-integration")

    agents = {
        "zendesk": zd_agent,
        "confluence": cf_agent,
        "request_handler": rh_agent,
        "synthesizer": synth_agent,
        "mock_embedding_service": mock_embedding_service, # Expose mocks for test-specific setup
        "mock_qdrant_client": mock_qdrant_client,
        "zd_collection_name": zd_collection_name,
        "cf_collection_name": cf_collection_name
    }

    yield agents

    # Teardown: Close all agents
    close_ops = [
        agents["zendesk"].close(),
        agents["confluence"].close(),
        agents["request_handler"].close(), # This should also close its children
        agents["synthesizer"].close()
    ]
    results = await asyncio.gather(*close_ops, return_exceptions=True)
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            agent_name = list(agents.keys())[i] # Get agent name for logging
            print(f"Error during closing of {agent_name}: {result}")


@pytest.mark.asyncio
async def test_full_flow_qdrant_success(integrated_agents):
    """
    Tests the full flow with Qdrant searches mocked to succeed for both agents.
    SynthesizingAgent is expected to use its placeholder as GOOGLE_API_KEY is a dummy.
    """
    rh_agent: RequestHandlerAgent = integrated_agents["request_handler"]
    synth_agent: SynthesizingAgent = integrated_agents["synthesizer"]
    mock_qdrant_client: AsyncMock = integrated_agents["mock_qdrant_client"]
    mock_embedding_service: AsyncMock = integrated_agents["mock_embedding_service"]
    zd_collection_name = integrated_agents["zd_collection_name"]
    cf_collection_name = integrated_agents["cf_collection_name"]

    query_text = "Integration Qdrant: login issue"
    query_input = QueryInput(query=query_text)

    # Configure Qdrant mock for this test
    mock_zd_payload = {"text": "Zendesk Qdrant result about login", "ticket_id": "QD-123", "title": "ZD Qdrant Login Title"}
    mock_cf_payload = {"text": "Confluence Qdrant result for login", "title": "CF Qdrant Login Runbook", "url": "http://qdrant.cf/login-runbook"}
    
    mock_zd_points = [create_mock_scored_point(id_val="zd_q_1", score_val=0.95, payload_val=mock_zd_payload)]
    mock_cf_points = [create_mock_scored_point(id_val="cf_q_1", score_val=0.90, payload_val=mock_cf_payload)]

    async def qdrant_search_side_effect(*args, **kwargs):
        collection_name_called = kwargs.get("collection_name")
        if collection_name_called == zd_collection_name:
            return mock_zd_points
        elif collection_name_called == cf_collection_name:
            return mock_cf_points
        return [] # Should not happen in this test
    
    mock_qdrant_client.search.side_effect = qdrant_search_side_effect

    # 1. Process through RequestHandlerAgent
    synthesizer_input_payload: SynthesizerInput = await rh_agent.process(query_input)

    # Assert calls to embedding and Qdrant
    # Embedding service called once per agent invoked by RequestHandler
    assert mock_embedding_service.generate_embeddings.call_count == 2 
    mock_embedding_service.generate_embeddings.assert_any_call([query_text]) # Called for ZD and CF

    assert mock_qdrant_client.search.call_count == 2
    mock_qdrant_client.search.assert_any_call(
        collection_name=zd_collection_name, 
        query_vector=mock_embedding_service.generate_embeddings.return_value[0], # Using the default mock embedding
        limit=5, # Default top_k for search_tickets
        with_payload=True
    )
    mock_qdrant_client.search.assert_any_call(
        collection_name=cf_collection_name, 
        query_vector=mock_embedding_service.generate_embeddings.return_value[0],
        limit=5, # Default top_k for search_runbooks
        with_payload=True
    )

    assert synthesizer_input_payload.original_query == query_text
    assert len(synthesizer_input_payload.zendesk_search_results) == 1
    zd_res_detail = synthesizer_input_payload.zendesk_search_results[0]
    assert zd_res_detail.text == mock_zd_payload["text"]
    assert zd_res_detail.ticket_id == mock_zd_payload["ticket_id"]
    assert zd_res_detail.title == mock_zd_payload["title"]
    assert "https://testintegrationsubdomain.zendesk.com/agent/tickets/QD-123" in zd_res_detail.url

    assert len(synthesizer_input_payload.confluence_search_results) == 1
    cf_res_detail = synthesizer_input_payload.confluence_search_results[0]
    assert cf_res_detail.text == mock_cf_payload["text"]
    assert cf_res_detail.title == mock_cf_payload["title"]
    assert cf_res_detail.url == mock_cf_payload["url"]


    # 2. Process through SynthesizingAgent
    final_output: SynthesizerOutput = await synth_agent.process(synthesizer_input_payload)

    assert isinstance(final_output, SynthesizerOutput)
    assert "Placeholder: Based on the context" in final_output.answer
    assert query_text in final_output.answer
    
    expected_ticket_id_for_action = mock_zd_payload["ticket_id"]
    assert ACTION_SUGGESTION_MARKER in final_output.answer
    assert final_output.suggested_action is not None
    assert f"ticket {expected_ticket_id_for_action}" in final_output.suggested_action.lower()
    assert final_output.action_details is not None
    assert final_output.action_details.get("ticket_id") == expected_ticket_id_for_action
    assert len(final_output.sources_used) == 2 # 1 Zendesk + 1 Confluence from Qdrant mocks

@pytest.mark.asyncio
async def test_full_flow_no_google_api_key_for_synthesizer(integrated_agents):
    """
    Tests flow where SynthesizingAgent has no API key, using Qdrant-sourced data.
    """
    rh_agent: RequestHandlerAgent = integrated_agents["request_handler"]
    mock_qdrant_client: AsyncMock = integrated_agents["mock_qdrant_client"]
    # mock_embedding_service is implicitly used by rh_agent

    query_text = "Integration Qdrant: no API key for synthesizer"
    query_input = QueryInput(query=query_text)

    # Configure Qdrant mock (similar to success test, can be simpler if details don't matter for this test focus)
    mock_zd_payload = {"text": "ZD data for no_api_key test", "ticket_id": "QD-NOKEY", "title": "ZD NoKey Title"}
    mock_zd_points = [create_mock_scored_point(id_val="zd_nk_1", score_val=0.9, payload_val=mock_zd_payload)]
    
    async def qdrant_search_side_effect(*args, **kwargs):
        # Simplified: For this test, only Zendesk data might be enough to check Synthesizer's placeholder
        if kwargs.get("collection_name") == integrated_agents["zd_collection_name"]:
            return mock_zd_points
        return [] 
    mock_qdrant_client.search.side_effect = qdrant_search_side_effect
    
    synthesizer_input_payload: SynthesizerInput = await rh_agent.process(query_input)
    assert len(synthesizer_input_payload.zendesk_search_results) == 1

    new_synth_agent = None
    with patch.dict(os.environ, {}, clear=True):
        new_synth_agent = SynthesizingAgent(model_name="gemini-1.5-flash-no-key-qdrant")
    
    assert new_synth_agent is not None
    assert new_synth_agent.generative_model is None

    final_output: SynthesizerOutput = await new_synth_agent.process(synthesizer_input_payload)
    await new_synth_agent.close()

    assert isinstance(final_output, SynthesizerOutput)
    assert "Placeholder: Based on the context" in final_output.answer
    assert query_text in final_output.answer
    assert ACTION_SUGGESTION_MARKER in final_output.answer
    assert final_output.action_details.get("ticket_id") == mock_zd_payload["ticket_id"]

@pytest.mark.asyncio
async def test_full_flow_zendesk_qdrant_fails(integrated_agents):
    """
    Tests flow where Zendesk Qdrant search fails, but Confluence succeeds.
    """
    rh_agent: RequestHandlerAgent = integrated_agents["request_handler"]
    synth_agent: SynthesizingAgent = integrated_agents["synthesizer"]
    mock_qdrant_client: AsyncMock = integrated_agents["mock_qdrant_client"]
    zd_collection_name = integrated_agents["zd_collection_name"]
    cf_collection_name = integrated_agents["cf_collection_name"]

    query_text = "Integration: Zendesk Qdrant Failure"
    query_input = QueryInput(query=query_text)

    mock_cf_payload = {"text": "Confluence data when ZD Qdrant fails", "title": "CF Survives ZD Failure", "url": "http://qdrant.cf/cf-survives", "id": "CFP_SURVIVE"}
    mock_cf_points = [create_mock_scored_point(id_val="cf_s_1", score_val=0.85, payload_val=mock_cf_payload)]

    async def qdrant_search_side_effect_zd_fail(*args, **kwargs):
        collection_name = kwargs.get("collection_name")
        if collection_name == zd_collection_name:
            raise Exception("Simulated Qdrant client error for Zendesk")
        elif collection_name == cf_collection_name:
            return mock_cf_points
        return []
    mock_qdrant_client.search.side_effect = qdrant_search_side_effect_zd_fail
    
    synthesizer_input_payload: SynthesizerInput = await rh_agent.process(query_input)

    assert len(synthesizer_input_payload.zendesk_search_results) == 0
    assert len(synthesizer_input_payload.confluence_search_results) == 1
    assert synthesizer_input_payload.confluence_search_results[0].title == mock_cf_payload["title"]

    final_output: SynthesizerOutput = await synth_agent.process(synthesizer_input_payload)

    assert isinstance(final_output, SynthesizerOutput)
    assert "Placeholder: Based on the context" in final_output.answer
    assert query_text in final_output.answer
    assert ACTION_SUGGESTION_MARKER in final_output.answer # Synthesizer placeholder should adapt
    # Action should now be based on Confluence data
    assert mock_cf_payload["title"] in final_output.suggested_action
    assert final_output.action_details.get("page_id") == mock_cf_payload.get("id") # From CF placeholder logic

# To run: pytest test_integration_flow.py
# Ensure all agent files and agent_schemas.py are in PYTHONPATH
# And necessary libraries (pytest, google-generativeai, httpx, qdrant-client etc.) are installed.
