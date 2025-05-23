import logging
import os
import json
from typing import Optional, List, Dict, Any, Union

from google.adk.agent import Agent as AdkAgent
import google.generativeai as genai

from agent_schemas import (
    SynthesizerInput,
    SynthesizerOutput,
    ContextDetail,
    ZendeskTicketAnalysis # Though not directly added to sources_used, it's part of input
)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Markers for parsing LLM output
ACTION_SUGGESTION_MARKER = "ACTION_SUGGESTION:"
ACTION_DETAILS_MARKER = "ACTION_DETAILS:"

class SynthesizingAgent(AdkAgent):
    """
    An agent that takes structured input from various sources (Zendesk, Confluence),
    uses a generative model to synthesize an answer, and suggests potential actions.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash-latest", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.generative_model: Optional[genai.GenerativeModel] = None
        self._tools: Dict[str, Any] = {} # No tools registered by this agent itself

        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.generative_model = genai.GenerativeModel(self.model_name)
                logger.info(f"SynthesizingAgent initialized with GenerativeModel: {self.model_name}")
            else:
                logger.warning(
                    "GOOGLE_API_KEY not found in environment. "
                    "SynthesizingAgent will use placeholder responses."
                )
        except Exception as e:
            logger.error(f"Error during GenerativeModel initialization: {e}", exc_info=True)
            self.generative_model = None # Fallback to placeholder

    def _format_context_for_prompt(self, synth_input: SynthesizerInput) -> str:
        """Formats the combined context from Zendesk and Confluence into a single string."""
        context_parts: List[str] = []

        if synth_input.zendesk_search_results:
            context_parts.append("--- Zendesk Ticket Search Results ---")
            for i, zd_res in enumerate(synth_input.zendesk_search_results):
                context_parts.append(
                    f"Zendesk Result {i+1}:\n"
                    f"  Title: {zd_res.title or 'N/A'}\n"
                    f"  Ticket ID: {zd_res.ticket_id or 'N/A'}\n"
                    f"  URL: {zd_res.url or 'N/A'}\n"
                    f"  Snippet: {zd_res.text}\n"
                    f"  Score: {zd_res.score or 'N/A'}"
                )
            context_parts.append("--- End of Zendesk Ticket Search Results ---")

        if synth_input.zendesk_analyses:
            context_parts.append("\n--- Zendesk Ticket Analyses ---")
            for i, zd_analysis in enumerate(synth_input.zendesk_analyses):
                context_parts.append(
                    f"Zendesk Analysis {i+1} for Ticket ID: {zd_analysis.ticket_id}:\n"
                    f"  AI Summary: {zd_analysis.ai_analysis or 'N/A'}\n"
                    f"  Conversation Highlights: {json.dumps(zd_analysis.conversation_summary, indent=2) if zd_analysis.conversation_summary else 'N/A'}"
                )
            context_parts.append("--- End of Zendesk Ticket Analyses ---")

        if synth_input.confluence_search_results:
            context_parts.append("\n--- Confluence Runbook Search Results ---")
            for i, cf_res in enumerate(synth_input.confluence_search_results):
                context_parts.append(
                    f"Confluence Result {i+1}:\n"
                    f"  Title: {cf_res.title or 'N/A'}\n"
                    f"  URL: {cf_res.url or 'N/A'}\n"
                    f"  Snippet: {cf_res.text}\n"
                    f"  Score: {cf_res.score or 'N/A'}"
                )
            context_parts.append("--- End of Confluence Runbook Search Results ---")

        if not context_parts:
            return "No context provided."

        return "\n\n".join(context_parts)

    async def process(self, synth_input: SynthesizerInput) -> SynthesizerOutput:
        """
        Generates a synthesized answer and suggests actions based on the input.
        """
        logger.info(f"SynthesizingAgent received query: {synth_input.original_query}")
        formatted_context = self._format_context_for_prompt(synth_input)

        system_prompt = (
            "You are a helpful AI assistant. Based on the provided context and the user's query, "
            "formulate a comprehensive answer. \n"
            "If you identify a clear, actionable step that can be taken (e.g., adding a comment to a Zendesk ticket), "
            "suggest this action using the following format ON SEPARATE LINES:\n"
            f"{ACTION_SUGGESTION_MARKER} <Describe the suggested action clearly and concisely.>\n"
            f"{ACTION_DETAILS_MARKER} <Provide a JSON string with necessary details for the action. "
            'For example: {"ticket_id": "12345", "comment_body": "This is a suggested comment."} '
            'or {"page_id": "54321", "comment": "Relevant finding for this page."}. '
            "If no specific details are needed or the action is generic, provide an empty JSON object {}.>\n"
            "If no specific action is appropriate or identifiable from the context, do not include the "
            f"ACTION_SUGGESTION or {ACTION_DETAILS_MARKER} lines.\n"
            "Focus on directly answering the user's query first, then optionally add the action suggestion."
        )

        full_prompt = f"{system_prompt}\n\n--- Provided Context ---\n{formatted_context}\n\n--- User's Original Query ---\n{synth_input.original_query}\n\n--- Answer ---\n"
        generated_text: Optional[str] = None
        llm_error: Optional[str] = None

        if self.generative_model:
            try:
                logger.info("Calling GenerativeModel to generate content...")
                response = await self.generative_model.generate_content_async(full_prompt)
                generated_text = response.text
                logger.info("Successfully received response from GenerativeModel.")
            except Exception as e:
                logger.error(f"Error during LLM call: {e}", exc_info=True)
                llm_error = str(e)
                generated_text = None # Fallback to placeholder

        if generated_text is None:
            logger.info("Using placeholder response logic.")
            if not synth_input.zendesk_search_results and not synth_input.confluence_search_results and not synth_input.zendesk_analyses:
                generated_text = f"Placeholder: No information found for your query: '{synth_input.original_query}'. There are no search results or analyses to process."
            else:
                placeholder_answer = (
                    f"Placeholder: Based on the context for '{synth_input.original_query}', "
                    "here's a summary. Further details would be generated by the LLM.\n"
                )
                if synth_input.zendesk_search_results:
                    first_zd_ticket = synth_input.zendesk_search_results[0]
                    placeholder_answer += (
                        f"\nAn action could be to comment on Zendesk ticket {first_zd_ticket.ticket_id}.\n"
                        f"{ACTION_SUGGESTION_MARKER} Add a comment to Zendesk ticket {first_zd_ticket.ticket_id} summarizing findings.\n"
                        f"{ACTION_DETAILS_MARKER} {json.dumps({'ticket_id': first_zd_ticket.ticket_id, 'comment_body': 'Placeholder: Further investigation details here.'})}"
                    )
                elif synth_input.confluence_search_results: # Fallback if no Zendesk results
                     first_cf_page = synth_input.confluence_search_results[0]
                     placeholder_answer += (
                        f"\nAn action could be to review Confluence page '{first_cf_page.title}'.\n"
                        f"{ACTION_SUGGESTION_MARKER} Review Confluence page '{first_cf_page.title}' for more details.\n"
                        f"{ACTION_DETAILS_MARKER} {json.dumps({'page_id': first_cf_page.raw_data.get('id') if first_cf_page.raw_data else 'N/A', 'action_type': 'review_confluence'})}"
                     )
                generated_text = placeholder_answer

        # Parse the generated text for answer, action suggestion, and details
        answer = generated_text
        suggested_action: Optional[str] = None
        action_details: Optional[Dict[str, Any]] = None

        if ACTION_SUGGESTION_MARKER in answer:
            parts = answer.split(ACTION_SUGGESTION_MARKER, 1)
            answer = parts[0].strip()
            suggestion_part = parts[1]

            if ACTION_DETAILS_MARKER in suggestion_part:
                detail_parts = suggestion_part.split(ACTION_DETAILS_MARKER, 1)
                suggested_action = detail_parts[0].strip()
                try:
                    action_details_str = detail_parts[1].strip()
                    action_details = json.loads(action_details_str)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse ACTION_DETAILS JSON: {detail_parts[1].strip()}. Error: {e}")
                    action_details = {"error": "Failed to parse action details JSON", "raw_details": detail_parts[1].strip()}
                except Exception as e: # Catch any other error during parsing
                    logger.error(f"Unexpected error parsing ACTION_DETAILS: {e}", exc_info=True)
                    action_details = {"error": "Unexpected error parsing action details", "raw_details": detail_parts[1].strip()}
            else:
                suggested_action = suggestion_part.strip()
        
        # Consolidate sources_used
        sources_used: List[ContextDetail] = []
        if synth_input.zendesk_search_results:
            sources_used.extend(synth_input.zendesk_search_results)
        if synth_input.confluence_search_results:
            sources_used.extend(synth_input.confluence_search_results)
        # Note: ZendeskTicketAnalysis objects are not ContextDetail, so not added here.

        return SynthesizerOutput(
            answer=answer.strip(),
            suggested_action=suggested_action,
            action_details=action_details,
            sources_used=sources_used,
            # error_message=llm_error # Could be added to SynthesizerOutput if needed
        )

    async def close(self):
        """Placeholder for any cleanup if needed in the future."""
        logger.info("SynthesizingAgent close method called. No specific resources to release for GenAI model.")
        pass

# Note: The `main` function and `if __name__ == "__main__":` block
# are intentionally omitted as per the instructions.
# The GOOGLE_API_KEY needs to be set as an environment variable for actual LLM calls.
# Placeholder logic is crucial for operation without the key or if LLM fails.
