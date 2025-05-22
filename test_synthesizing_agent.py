import pytest
import asyncio
import os
import json
from unittest.mock import AsyncMock, MagicMock, patch

# Import the agent and schemas
from synthesizing_agent import SynthesizingAgent, ACTION_SUGGESTION_MARKER, ACTION_DETAILS_MARKER
from agent_schemas import (
    SynthesizerInput,
    SynthesizerOutput,
    ContextDetail,
    ZendeskTicketAnalysis
)

# For mocking genai parts
import google.generativeai as genai # To help with spec for mocks

# --- Test Data Samples ---
ZD_RESULT_1 = ContextDetail(
    text="Zendesk result 1 text about login.",
    source_collection="zendesk",
    score=0.9,
    ticket_id="ZD101",
    title="Login Issue ZD101",
    url="http://zendesk/101"
)
CF_RESULT_1 = ContextDetail(
    text="Confluence result 1 text about API.",
    source_collection="confluence_runbooks",
    score=0.85,
    title="API Usage Runbook",
    url="http://confluence/api-runbook",
    raw_data={"id": "CFP123"}
)
ZD_ANALYSIS_1 = ZendeskTicketAnalysis(
    ticket_id="ZD101",
    conversation_summary=[{"user": "I cannot login"}, {"agent": "Try reset password"}],
    ai_analysis="User is having login issues. Password reset suggested."
)

@pytest.fixture
def synthesizer_input_empty() -> SynthesizerInput:
    return SynthesizerInput(
        original_query="test query empty",
        zendesk_search_results=[],
        zendesk_analyses=[],
        confluence_search_results=[]
    )

@pytest.fixture
def synthesizer_input_with_data() -> SynthesizerInput:
    return SynthesizerInput(
        original_query="test query with data",
        zendesk_search_results=[ZD_RESULT_1],
        zendesk_analyses=[ZD_ANALYSIS_1],
        confluence_search_results=[CF_RESULT_1]
    )

@pytest.fixture
def mock_generative_model_instance() -> AsyncMock:
    """Provides a mock for the genai.GenerativeModel instance."""
    mock_model = AsyncMock(spec=genai.GenerativeModel)
    mock_model.generate_content_async = AsyncMock() # This is the method we'll be calling
    return mock_model

@pytest.fixture(autouse=True)
def mock_genai_setup(mock_generative_model_instance: AsyncMock):
    """
    Auto-used fixture to mock genai configuration, model instantiation,
    and environment variable for GOOGLE_API_KEY.
    """
    with patch('synthesizing_agent.os.getenv') as mock_getenv, \
         patch('synthesizing_agent.genai.configure') as mock_genai_configure, \
         patch('synthesizing_agent.genai.GenerativeModel') as mock_GenModelClass:

        mock_getenv.return_value = "test_api_key" # Simulate API key presence by default
        mock_GenModelClass.return_value = mock_generative_model_instance # Make class return our mock instance
        yield mock_getenv, mock_genai_configure, mock_GenModelClass # Provide mocks to tests if needed

@pytest.fixture
async def synthesizing_agent_fixture(mock_genai_setup) -> SynthesizingAgent: # mock_genai_setup is autouse but can be explicit
    """Initializes and yields a SynthesizingAgent instance."""
    agent = SynthesizingAgent(model_name="test-gemini-model")
    # The mock_genai_setup fixture ensures that genai.configure and GenerativeModel
    # are already mocked during the agent's __init__ call.
    yield agent
    await agent.close()


# --- Initialization Tests ---
@pytest.mark.asyncio
async def test_initialization_with_api_key(
    synthesizing_agent_fixture: SynthesizingAgent,
    mock_genai_setup, # Access the mocks from mock_genai_setup
    mock_generative_model_instance: AsyncMock
):
    _mock_getenv, mock_genai_configure, mock_GenModelClass = mock_genai_setup
    agent = synthesizing_agent_fixture # Agent is already initialized

    mock_getenv.assert_called_with("GOOGLE_API_KEY")
    mock_genai_configure.assert_called_once_with(api_key="test_api_key")
    mock_GenModelClass.assert_called_once_with(agent.model_name)
    assert agent.generative_model == mock_generative_model_instance
    assert list(agent.tools) == [] # Check for no tools

@pytest.mark.asyncio
async def test_initialization_without_api_key(mock_genai_setup): # No agent fixture needed here
    mock_getenv, _, _ = mock_genai_setup
    mock_getenv.return_value = None # Override default behavior for this test

    agent_no_key = SynthesizingAgent(model_name="test-model-no-key")

    assert agent_no_key.generative_model is None
    # genai.configure and genai.GenerativeModel should not have been called
    # mock_genai_setup[1] is mock_genai_configure, mock_genai_setup[2] is mock_GenModelClass
    mock_genai_setup[1].assert_not_called()
    mock_genai_setup[2].assert_not_called()
    await agent_no_key.close()

@pytest.mark.asyncio
async def test_initialization_genai_exception(mock_genai_setup):
    mock_getenv, mock_genai_configure, _ = mock_genai_setup
    mock_getenv.return_value = "test_api_key"
    mock_genai_configure.side_effect = Exception("GenAI config error")

    agent_exc = SynthesizingAgent(model_name="test-model-exc")
    assert agent_exc.generative_model is None
    await agent_exc.close()


# --- _format_context_for_prompt Tests (Non-Async) ---
def test_format_context_empty(synthesizing_agent_fixture: SynthesizingAgent, synthesizer_input_empty: SynthesizerInput):
    formatted_str = synthesizing_agent_fixture._format_context_for_prompt(synthesizer_input_empty)
    assert formatted_str == "No context provided."

def test_format_context_with_data(synthesizing_agent_fixture: SynthesizingAgent, synthesizer_input_with_data: SynthesizerInput):
    formatted_str = synthesizing_agent_fixture._format_context_for_prompt(synthesizer_input_with_data)
    assert "--- Zendesk Ticket Search Results ---" in formatted_str
    assert ZD_RESULT_1.text in formatted_str
    assert f"Ticket ID: {ZD_RESULT_1.ticket_id}" in formatted_str
    assert "--- Zendesk Ticket Analyses ---" in formatted_str
    assert ZD_ANALYSIS_1.ai_analysis in formatted_str
    assert f"Ticket ID: {ZD_ANALYSIS_1.ticket_id}" in formatted_str
    assert "--- Confluence Runbook Search Results ---" in formatted_str
    assert CF_RESULT_1.text in formatted_str
    assert CF_RESULT_1.title in formatted_str
    assert "--- End of Zendesk Ticket Search Results ---" in formatted_str # Check for end markers too

# --- Process Method Tests ---
@pytest.mark.asyncio
async def test_process_llm_answer_only(
    synthesizing_agent_fixture: SynthesizingAgent,
    synthesizer_input_with_data: SynthesizerInput,
    mock_generative_model_instance: AsyncMock
):
    agent = synthesizing_agent_fixture
    llm_response_text = "This is a direct answer from the LLM."
    mock_generative_model_instance.generate_content_async.return_value = MagicMock(text=llm_response_text)

    output = await agent.process(synthesizer_input_with_data)

    mock_generative_model_instance.generate_content_async.assert_called_once()
    assert output.answer == llm_response_text
    assert output.suggested_action is None
    assert output.action_details is None
    assert len(output.sources_used) == 2 # ZD_RESULT_1 and CF_RESULT_1

@pytest.mark.asyncio
async def test_process_llm_answer_with_action(
    synthesizing_agent_fixture: SynthesizingAgent,
    synthesizer_input_with_data: SynthesizerInput,
    mock_generative_model_instance: AsyncMock
):
    agent = synthesizing_agent_fixture
    action_desc = "Add comment to ZD101"
    action_json_str = json.dumps({"ticket_id": "ZD101", "comment_body": "LLM suggested comment"})
    llm_response_text = (
        f"This is the main answer.\n"
        f"{ACTION_SUGGESTION_MARKER} {action_desc}\n"
        f"{ACTION_DETAILS_MARKER} {action_json_str}"
    )
    mock_generative_model_instance.generate_content_async.return_value = MagicMock(text=llm_response_text)

    output = await agent.process(synthesizer_input_with_data)

    assert output.answer == "This is the main answer."
    assert output.suggested_action == action_desc
    assert output.action_details == json.loads(action_json_str)

@pytest.mark.asyncio
async def test_process_llm_answer_with_malformed_action_json(
    synthesizing_agent_fixture: SynthesizingAgent,
    synthesizer_input_with_data: SynthesizerInput,
    mock_generative_model_instance: AsyncMock
):
    agent = synthesizing_agent_fixture
    action_desc = "Try something"
    malformed_json_str = '{"ticket_id": "ZD101", "comment_body": "Missing quote}' # Invalid JSON
    llm_response_text = (
        f"Main answer part.\n"
        f"{ACTION_SUGGESTION_MARKER} {action_desc}\n"
        f"{ACTION_DETAILS_MARKER} {malformed_json_str}"
    )
    mock_generative_model_instance.generate_content_async.return_value = MagicMock(text=llm_response_text)

    output = await agent.process(synthesizer_input_with_data)

    assert output.answer == "Main answer part."
    assert output.suggested_action == action_desc
    assert output.action_details is not None
    assert output.action_details["error"] == "Failed to parse action details JSON"
    assert output.action_details["raw_details"] == malformed_json_str

@pytest.mark.asyncio
async def test_process_llm_call_failure(
    synthesizing_agent_fixture: SynthesizingAgent,
    synthesizer_input_with_data: SynthesizerInput,
    mock_generative_model_instance: AsyncMock
):
    agent = synthesizing_agent_fixture
    mock_generative_model_instance.generate_content_async.side_effect = Exception("LLM API Error")

    output = await agent.process(synthesizer_input_with_data) # Should use placeholder

    assert "Placeholder: Based on the context" in output.answer
    assert ZD_RESULT_1.ticket_id in output.answer # Placeholder uses first ZD ticket
    assert ACTION_SUGGESTION_MARKER in output.answer # Placeholder suggests an action
    assert output.suggested_action is not None
    assert output.action_details is not None
    assert output.action_details["ticket_id"] == ZD_RESULT_1.ticket_id

@pytest.mark.asyncio
async def test_process_no_llm_placeholder_with_data(
    mock_genai_setup, # To override API key for this agent instance
    synthesizer_input_with_data: SynthesizerInput
):
    mock_getenv, _, _ = mock_genai_setup
    mock_getenv.return_value = None # Simulate no API key for this specific test run

    # Create agent instance *after* os.getenv is patched for no API key
    agent_no_llm = SynthesizingAgent(model_name="no-llm-model")
    assert agent_no_llm.generative_model is None # Confirm no model

    output = await agent_no_llm.process(synthesizer_input_with_data)

    assert "Placeholder: Based on the context" in output.answer
    assert ZD_RESULT_1.ticket_id in output.answer
    assert ACTION_SUGGESTION_MARKER in output.answer
    assert output.suggested_action is not None
    assert output.action_details is not None
    assert output.action_details["ticket_id"] == ZD_RESULT_1.ticket_id
    await agent_no_llm.close()

@pytest.mark.asyncio
async def test_process_no_llm_placeholder_empty_input(
    mock_genai_setup, # To override API key
    synthesizer_input_empty: SynthesizerInput
):
    mock_getenv, _, _ = mock_genai_setup
    mock_getenv.return_value = None

    agent_no_llm_empty = SynthesizingAgent(model_name="no-llm-model-empty")
    assert agent_no_llm_empty.generative_model is None

    output = await agent_no_llm_empty.process(synthesizer_input_empty)

    assert "Placeholder: No information found" in output.answer
    assert ACTION_SUGGESTION_MARKER not in output.answer # No action if no info
    assert output.suggested_action is None
    assert output.action_details is None

@pytest.mark.asyncio
async def test_close_method(synthesizing_agent_fixture: SynthesizingAgent):
    """Tests that the close method runs without error."""
    agent = synthesizing_agent_fixture
    try:
        await agent.close() # Already called by fixture, but call again for explicitness
    except Exception as e:
        pytest.fail(f"SynthesizingAgent.close() raised an exception: {e}")

# To run: pytest test_synthesizing_agent.py
# Ensure synthesizing_agent.py and agent_schemas.py are accessible.
# And google.adk, google.generativeai are available.
