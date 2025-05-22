import pytest
import asyncio
import os
from unittest.mock import patch, AsyncMock

# Import agent classes
from zendesk_agent import ZendeskAgent
from confluence_agent import ConfluenceAgent
from request_handler_agent import RequestHandlerAgent
from synthesizing_agent import SynthesizingAgent, ACTION_SUGGESTION_MARKER, ACTION_DETAILS_MARKER

# Import Pydantic schemas
from agent_schemas import (
    QueryInput,
    SynthesizerInput, # To inspect intermediate state
    SynthesizerOutput,
    ContextDetail
)

@pytest.fixture
async def integrated_agents():
    """
    Initializes and yields a dictionary of all four agents for integration testing.
    Sets a dummy GOOGLE_API_KEY for SynthesizingAgent default initialization.
    Handles closing of all agents during teardown.
    """
    # Patch os.environ for the duration of this fixture setup
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_integration_key_dummy"}):
        # Initialize agents with minimal valid config for their placeholders
        # These configs are for the placeholder logic within the agents themselves
        zd_agent = ZendeskAgent(zendesk_config={"domain": "testintegrationsubdomain"})
        cf_agent = ConfluenceAgent(confluence_collection_name="test_integration_runbooks")
        rh_agent = RequestHandlerAgent(
            zendesk_agent_instance=zd_agent,
            confluence_agent_instance=cf_agent
        )
        # SynthesizingAgent will try to init with the dummy API key.
        # Its internal LLM call is not mocked here, so it will likely fallback to its own placeholder.
        synth_agent = SynthesizingAgent(model_name="gemini-1.5-flash-test-integration")

    agents = {
        "zendesk": zd_agent,
        "confluence": cf_agent,
        "request_handler": rh_agent,
        "synthesizer": synth_agent
    }

    yield agents

    # Teardown: Close all agents
    close_ops = [
        agents["zendesk"].close(),
        agents["confluence"].close(),
        agents["request_handler"].close(), # This should also close its children, but explicit is fine
        agents["synthesizer"].close()
    ]
    results = await asyncio.gather(*close_ops, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            print(f"Error during agent close in teardown: {result}")


@pytest.mark.asyncio
async def test_full_flow_with_placeholder_data(integrated_agents):
    """
    Tests the full flow from RequestHandlerAgent to SynthesizingAgent using their
    respective placeholder data generation logic.
    SynthesizingAgent is expected to use its placeholder as GOOGLE_API_KEY is a dummy.
    """
    rh_agent: RequestHandlerAgent = integrated_agents["request_handler"]
    synth_agent: SynthesizingAgent = integrated_agents["synthesizer"]

    query_text = "Integration test: login issue"
    query_input = QueryInput(query=query_text)

    # 1. Process through RequestHandlerAgent
    synthesizer_input_payload: SynthesizerInput = await rh_agent.process(query_input)

    assert synthesizer_input_payload.original_query == query_text
    # ZendeskAgent placeholder returns 2 results
    assert len(synthesizer_input_payload.zendesk_search_results) == 2
    assert synthesizer_input_payload.zendesk_search_results[0].ticket_id == "ZD12345" # From ZendeskAgent placeholder
    assert "testintegrationsubdomain" in synthesizer_input_payload.zendesk_search_results[0].url
    # ConfluenceAgent placeholder returns 1 result
    assert len(synthesizer_input_payload.confluence_search_results) == 1
    assert "Runbook: Resolve API Gateway Errors" in synthesizer_input_payload.confluence_search_results[0].title # From ConfluenceAgent placeholder
    assert synthesizer_input_payload.confluence_search_results[0].source_collection == "test_integration_runbooks"

    # 2. Process through SynthesizingAgent
    # SynthesizingAgent with a dummy API key will fail actual LLM call and use its placeholder.
    final_output: SynthesizerOutput = await synth_agent.process(synthesizer_input_payload)

    assert isinstance(final_output, SynthesizerOutput)
    # Check for SynthesizingAgent's placeholder response elements
    assert "Placeholder: Based on the context" in final_output.answer
    assert query_text in final_output.answer # Original query should be in the answer

    # SynthesizingAgent placeholder should pick the first Zendesk ticket for action
    expected_ticket_id_for_action = synthesizer_input_payload.zendesk_search_results[0].ticket_id
    assert ACTION_SUGGESTION_MARKER in final_output.answer # Placeholder output includes markers
    assert final_output.suggested_action is not None
    assert f"ticket {expected_ticket_id_for_action}" in final_output.suggested_action.lower()
    assert final_output.action_details is not None
    assert final_output.action_details.get("ticket_id") == expected_ticket_id_for_action
    assert len(final_output.sources_used) == 3 # 2 Zendesk + 1 Confluence

@pytest.mark.asyncio
async def test_full_flow_no_google_api_key_for_synthesizer(integrated_agents):
    """
    Tests the flow where SynthesizingAgent is explicitly initialized without an API key,
    forcing its no-LLM placeholder logic.
    """
    rh_agent: RequestHandlerAgent = integrated_agents["request_handler"]
    # Original synthesizer from fixture (has dummy key) is not used here.

    query_text = "Integration test: no API key for synthesizer"
    query_input = QueryInput(query=query_text)

    # 1. Get input for synthesizer
    synthesizer_input_payload: SynthesizerInput = await rh_agent.process(query_input)
    assert len(synthesizer_input_payload.zendesk_search_results) == 2 # Still get data from previous agents

    # 2. Re-initialize SynthesizingAgent in a context where GOOGLE_API_KEY is not set
    new_synth_agent = None
    with patch.dict(os.environ, {}, clear=True): # Clear all env vars, including GOOGLE_API_KEY
        new_synth_agent = SynthesizingAgent(model_name="gemini-1.5-flash-no-key-test")
    
    assert new_synth_agent is not None
    assert new_synth_agent.generative_model is None # Crucial check

    final_output: SynthesizerOutput = await new_synth_agent.process(synthesizer_input_payload)
    await new_synth_agent.close() # Explicit close for this locally created agent

    assert isinstance(final_output, SynthesizerOutput)
    assert "Placeholder: Based on the context" in final_output.answer # No-LLM placeholder
    assert query_text in final_output.answer
    assert ACTION_SUGGESTION_MARKER in final_output.answer
    assert final_output.suggested_action is not None
    assert final_output.action_details is not None
    assert final_output.action_details.get("ticket_id") == synthesizer_input_payload.zendesk_search_results[0].ticket_id

@pytest.mark.asyncio
async def test_full_flow_zendesk_fails_in_requesthandler(integrated_agents):
    """
    Tests the flow where ZendeskAgent fails within RequestHandlerAgent.
    SynthesizingAgent should then process only Confluence data.
    """
    rh_agent: RequestHandlerAgent = integrated_agents["request_handler"]
    synth_agent: SynthesizingAgent = integrated_agents["synthesizer"]
    zd_agent: ZendeskAgent = integrated_agents["zendesk"] # Get the original Zendesk agent

    query_text = "Integration test: Zendesk failure"
    query_input = QueryInput(query=query_text)

    # Patch the specific ZendeskAgent instance's method
    original_search_tickets = zd_agent.search_tickets
    zd_agent.search_tickets = AsyncMock(side_effect=Exception("Mocked Zendesk API Failure"))

    try:
        # 1. Process through RequestHandlerAgent - expecting Zendesk to fail
        synthesizer_input_payload: SynthesizerInput = await rh_agent.process(query_input)

        assert synthesizer_input_payload.original_query == query_text
        assert len(synthesizer_input_payload.zendesk_search_results) == 0 # Zendesk results should be empty
        assert len(synthesizer_input_payload.confluence_search_results) == 1 # Confluence should still work
        expected_confluence_title = "Runbook: Resolve API Gateway Errors" # From ConfluenceAgent placeholder
        assert synthesizer_input_payload.confluence_search_results[0].title == expected_confluence_title

        # 2. Process through SynthesizingAgent
        final_output: SynthesizerOutput = await synth_agent.process(synthesizer_input_payload)

        assert isinstance(final_output, SynthesizerOutput)
        assert "Placeholder: Based on the context" in final_output.answer # Synthesizer's placeholder
        assert query_text in final_output.answer

        # SynthesizingAgent placeholder should now pick Confluence data for action
        # if Zendesk data is absent.
        assert ACTION_SUGGESTION_MARKER in final_output.answer
        assert final_output.suggested_action is not None
        # Check if the action is related to the Confluence page
        assert expected_confluence_title in final_output.suggested_action
        assert final_output.action_details is not None
        # Placeholder for Confluence provides page_id from raw_data if available
        expected_page_id = synthesizer_input_payload.confluence_search_results[0].raw_data.get("id")
        assert final_output.action_details.get("page_id") == expected_page_id
        assert len(final_output.sources_used) == 1 # Only Confluence

    finally:
        # Restore the original method to avoid affecting other tests if agent instances were reused (not in this setup)
        zd_agent.search_tickets = original_search_tickets

# To run: pytest test_integration_flow.py
# Ensure all agent files and agent_schemas.py are in PYTHONPATH
# And necessary libraries (pytest, google-generativeai, etc.) are installed.
