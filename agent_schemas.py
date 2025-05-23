from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# --- General Utility Schemas ---

class QueryInput(BaseModel):
    query: str = Field(description="The user query.")

class ContextDetail(BaseModel):
    text: str = Field(description="The retrieved text snippet.")
    source_collection: str = Field(description="The name of the source (e.g., 'zendesk', 'confluence').")
    score: Optional[float] = Field(default=None, description="The relevance score of the snippet.")
    ticket_id: Optional[str] = Field(default=None, description="The Zendesk ticket ID, if applicable.")
    title: Optional[str] = Field(default=None, description="The title of the document or ticket.")
    url: Optional[str] = Field(default=None, description="The URL of the source, if available.")
    raw_data: Optional[Dict[str, Any]] = Field(default=None, description="Original data from the source.") # For richer data passing

class ZendeskTicketAnalysis(BaseModel):
    ticket_id: str
    conversation_summary: Optional[List[Dict[str, Any]]] = Field(default=None, description="Summary of comments or key points.")
    ai_analysis: Optional[str] = Field(default=None, description="AI-generated summary or suggested solution for the ticket.")

# --- Request Handling Agent ---

class RequestHandlerOutput(BaseModel):
    original_query: str
    # This agent might not produce direct textual output but rather pass structured data
    # For simplicity, let's assume it gathers results that are then passed on.
    # Or, it could simply pass the query to the next stage if orchestration happens outside.
    # For now, let's keep it simple. It initiates the process.
    message: str = Field(description="Status message from the request handler.")

# --- Zendesk Agent ---

# Input for searching tickets is QueryInput

class ZendeskSearchOutput(BaseModel):
    query: str
    results: List[ContextDetail] = Field(description="List of relevant Zendesk ticket snippets.")

# Input for analyzing a ticket
class ZendeskAnalyzeInput(BaseModel):
    ticket_id: str = Field(description="The Zendesk ticket ID to analyze.")
    # query: Optional[str] = Field(None, description="Original user query, if context is needed for analysis")

class ZendeskAnalysisOutput(ZendeskTicketAnalysis): # Inherits from the analysis model
    pass

# Input for adding a comment
class ZendeskAddCommentInput(BaseModel):
    ticket_id: str = Field(description="The Zendesk ticket ID.")
    comment_body: str = Field(description="The text of the comment to add.")
    # public: bool = Field(default=False, description="Whether the comment should be public or internal.") # Example extra field

class ZendeskAddCommentOutput(BaseModel):
    ticket_id: str
    comment_id: Optional[str] = Field(default=None, description="The ID of the newly created comment.")
    status: str = Field(description="Status of the operation (e.g., 'Comment added successfully').")
    error: Optional[str] = Field(default=None, description="Error message if the operation failed.")

# --- Confluence Agent ---

# Input for searching runbooks is QueryInput

class ConfluenceSearchOutput(BaseModel):
    query: str
    results: List[ContextDetail] = Field(description="List of relevant Confluence runbook snippets.")

# --- Synthesizing Agent ---

class SynthesizerInput(BaseModel):
    original_query: str
    zendesk_search_results: Optional[List[ContextDetail]] = Field(default=None)
    zendesk_analyses: Optional[List[ZendeskTicketAnalysis]] = Field(default=None) # If specific tickets were analyzed
    confluence_search_results: Optional[List[ContextDetail]] = Field(default=None)

class SynthesizerOutput(BaseModel):
    answer: str = Field(description="The final synthesized answer for the user.")
    suggested_action: Optional[str] = Field(default=None, description="A description of a suggested action, e.g., 'Add comment to ticket X'.")
    action_details: Optional[Dict[str, Any]] = Field(default=None, description="Details needed to perform the action, e.g., {'ticket_id': '123', 'comment': '...'}.")
    sources_used: List[ContextDetail] = Field(description="List of sources used to formulate the answer.")

# --- Tool Schemas (for registration with ADK Agent) ---

# For ZendeskAgent
zendesk_search_tickets_tool_schema = {
    "name": "search_zendesk_tickets",
    "description": "Searches Zendesk tickets for information relevant to the user's query.",
    "input_schema": QueryInput.model_json_schema(),
    "output_schema": ZendeskSearchOutput.model_json_schema()
}

zendesk_analyze_ticket_conversation_tool_schema = {
    "name": "analyze_zendesk_ticket_conversation",
    "description": "Analyzes a specific Zendesk ticket's conversation history using an AI model.",
    "input_schema": ZendeskAnalyzeInput.model_json_schema(),
    "output_schema": ZendeskAnalysisOutput.model_json_schema()
}

zendesk_add_comment_tool_schema = {
    "name": "add_comment_to_zendesk_ticket",
    "description": "Adds a comment to a specified Zendesk ticket.",
    "input_schema": ZendeskAddCommentInput.model_json_schema(),
    "output_schema": ZendeskAddCommentOutput.model_json_schema()
}

# For ConfluenceAgent
confluence_search_runbooks_tool_schema = {
    "name": "search_confluence_runbooks",
    "description": "Searches Confluence runbooks for information relevant to the user's query.",
    "input_schema": QueryInput.model_json_schema(),
    "output_schema": ConfluenceSearchOutput.model_json_schema()
}
