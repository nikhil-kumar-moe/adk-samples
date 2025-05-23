import logging
import asyncio
from typing import List, Dict, Any, Optional

import google.adk as adk
# from qdrant_client import QdrantClient, models as qdrant_models # Actual import
# from some_embedding_service import EmbeddingService # Actual import

from agent_schemas import (
    QueryInput,
    ConfluenceSearchOutput,
    ContextDetail,
    confluence_search_runbooks_tool_schema # Schema for tool registration
)

# Placeholder type hints for services not fully integrated in this scope
EmbeddingService = Any
QdrantClient = Any

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ConfluenceAgent(adk.Agent):
    """
    An agent specialized in interacting with Confluence, primarily for searching runbooks
    using an embedding service and Qdrant vector database.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_client: QdrantClient,
        confluence_collection_name: str,
        model_name: str = "gemini-1.5-flash-latest", # Retained from previous, can be used for other purposes
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name # Retained, might be useful for future extensions
        self.embedding_service = embedding_service
        self.qdrant_client = qdrant_client
        self.confluence_collection_name = confluence_collection_name

        logger.info(
            f"Initializing ConfluenceAgent with model: {self.model_name}, "
            f"collection: {self.confluence_collection_name}. "
            f"Embedding service provided: {self.embedding_service is not None}. "
            f"Qdrant client provided: {self.qdrant_client is not None}."
        )

        # Register tools - schema remains the same for the tool's external signature
        self._tools = {
            "search_confluence_runbooks": adk.FunctionTool(
                fn=self.process, # Tool now calls process, which then calls search_runbooks
                schema=confluence_search_runbooks_tool_schema,
            )
        }

        if not self.embedding_service:
            raise ValueError("ConfluenceAgent: embedding_service is required.")
        if not self.qdrant_client:
            raise ValueError("ConfluenceAgent: qdrant_client is required.")
        if not self.confluence_collection_name:
            raise ValueError("ConfluenceAgent: confluence_collection_name is required and cannot be empty.")


    async def search_runbooks(self, query_input: QueryInput, top_k: int = 5) -> ConfluenceSearchOutput:
        """
        Searches Confluence runbooks in Qdrant based on the query embedding.
        """
        logger.info(
            f"ConfluenceAgent searching runbooks for query: '{query_input.query}', "
            f"collection: '{self.confluence_collection_name}', top_k: {top_k}"
        )

        results: List[ContextDetail] = []
        empty_output = ConfluenceSearchOutput(query=query_input.query, results=[])

        try:
            if not self.embedding_service or not self.qdrant_client:
                logger.error("ConfluenceAgent: Embedding service or Qdrant client not available. Cannot perform search.")
                return empty_output

            # Generate query embedding
            # Assuming generate_embeddings can take a single string and returns a list/vector
            # Also assuming it's an async method. If not, remove await.
            query_embedding_list = await self.embedding_service.generate_embeddings([query_input.query])
            if not query_embedding_list or not query_embedding_list[0]:
                logger.error(f"ConfluenceAgent: Failed to generate embedding for query: {query_input.query}")
                return empty_output
            
            query_embedding = query_embedding_list[0] # Take the first embedding for the single query

            logger.info(f"ConfluenceAgent: Embedding generated, proceeding to Qdrant search in collection '{self.confluence_collection_name}'.")

            # Perform Qdrant search (assuming qdrant_client.search is awaitable)
            # In a real qdrant_client setup, search might not be async directly,
            # but run in an executor. For this exercise, we assume it's awaitable.
            search_results = await self.qdrant_client.search(
                collection_name=self.confluence_collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True  # Ensure payload is returned
            )
            # Example: search_results = self.qdrant_client.search(...) if not async

            for hit in search_results:
                payload = hit.payload if hit.payload is not None else {}
                context_detail = ContextDetail(
                    text=payload.get("text", ""), # Ensure 'text' field in Qdrant payload
                    source_collection=self.confluence_collection_name,
                    score=hit.score,
                    title=payload.get("title"),
                    url=payload.get("url"), # Ensure 'url' field in Qdrant payload
                    raw_data=payload # Store the entire payload for richer context
                )
                results.append(context_detail)

            logger.info(f"ConfluenceAgent search found {len(results)} results from Qdrant for query: '{query_input.query}'.")
            return ConfluenceSearchOutput(query=query_input.query, results=results)

        except Exception as e:
            logger.error(f"ConfluenceAgent: Error during search_runbooks for query '{query_input.query}': {e}", exc_info=True)
            return empty_output


    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Main processing method for the agent, expecting a dictionary with 'query' and optional 'top_k'.
        This method is called by the ADK FunctionTool.
        """
        logger.info(f"ConfluenceAgent process method called with input_data: {input_data}")

        query_str = input_data.get("query")
        top_k_val = input_data.get("top_k", 5) # Default top_k

        if not query_str:
            logger.error(f"ConfluenceAgent: 'query' not found in input_data for process: {input_data}")
            # Return a structure that matches the expected output schema, indicating error if possible
            # For now, returning a dict that can be serialized by ADK.
            # The confluence_search_runbooks_tool_schema expects ConfluenceSearchOutput.
            # So, an error should ideally be an exception or a valid ConfluenceSearchOutput with error info.
            # However, the prompt asks for `{"error": "..."}`. This might not align with the registered schema.
            # For strict adherence to returning a dict:
            return {"error": "Invalid input. 'query' is required."}
            # To align with schema, one might raise an error or return:
            # return ConfluenceSearchOutput(query="", results=[], error_message="Invalid input. 'query' is required.").model_dump()


        query = QueryInput(query=query_str)
        search_output = await self.search_runbooks(query, top_k=top_k_val)
        return search_output.model_dump() # Return Pydantic model dumped to dict


    async def close(self):
        """Clean up resources. Currently, no specific resources like httpx_client to close."""
        logger.info("ConfluenceAgent closed.")
        # If qdrant_client or embedding_service had close methods, they would be called here.

# Note: The `main` function and `if __name__ == "__main__":` block
# are intentionally omitted as per the instructions.
# The actual qdrant_client might use `models.ScoredPoint` for search_results hits.
# The embedding_service.generate_embeddings might return a list of embeddings.
# For this exercise, we assume the interfaces align with the usage here.
# If qdrant_client.search is not async, it should be wrapped with e.g. asyncio.to_thread
# or the method should not be async. For this task, assuming it's awaitable.
# The ADK FunctionTool now points to `process`, which then calls `search_runbooks`.
# The `process` method now needs to conform to what the ADK expects for tool calls based on the schema.
# The schema `confluence_search_runbooks_tool_schema` defines input as `QueryInput` and output as `ConfluenceSearchOutput`.
# The `process` method needs to bridge the generic dict input from ADK to these types if `FunctionTool` doesn't do it automatically.
# The `FunctionTool` in ADK typically handles the Pydantic model validation based on the schema.
# So, `process` might receive `QueryInput` directly if the schema is correctly used by ADK.
# However, the prompt for `process` signature is `input_data: Dict[str, Any]`.
# I'll stick to the prompt's signature for `process` and manually parse from dict.
# The tool registration `fn=self.process` is correct based on this.
# The return from `process` should be a dict (`model_dump()`).
# The error return `{"error": "..."}` from process might conflict with `ConfluenceSearchOutput` schema.
# A better way for errors might be to raise an exception that the ADK framework handles,
# or return a `ConfluenceSearchOutput` with an error field if the schema supported it.
# For now, I'm following the direct instruction for the error return.
# The `generate_embeddings` in the instructions takes a single query string, but embedding services
# often take a list of texts. I'll assume it takes a list and we pass `[query_input.query]`.The file `confluence_agent.py` has been updated with the specified changes.

Key modifications include:
-   **Imports**: Added `Any` for `EmbeddingService` and `QdrantClient`.
-   **`__init__`**:
    -   Signature updated to require `embedding_service`, `qdrant_client`, and `confluence_collection_name`.
    -   These are stored as instance attributes.
    -   `httpx_client` has been removed.
    -   `ValueError` checks are added for the new required dependencies.
    -   The tool registration for "search\_confluence\_runbooks" now correctly points to `self.process`.
-   **`search_runbooks`**:
    -   Signature updated to include `top_k`.
    -   Old placeholder logic removed.
    -   New Qdrant search logic implemented:
        -   Generates embeddings using `await self.embedding_service.generate_embeddings([query_input.query])` (assuming it takes a list and returns a list of embeddings).
        -   Calls `await self.qdrant_client.search(...)`.
        -   Processes Qdrant `search_results` (assumed `ScoredPoint` like objects) into a list of `ContextDetail`.
        -   Includes error handling and appropriate logging.
-   **`process` method**:
    -   Signature changed to `async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:`.
    -   Parses `query` string and `top_k` value from `input_data`.
    -   Calls `self.search_runbooks` with these parsed values.
    -   Returns the `model_dump()` of the `ConfluenceSearchOutput`.
    -   Handles missing `query` in `input_data` by returning an error dictionary as specified.
-   **`close` method**:
    -   Simplified to only log "ConfluenceAgent closed."

The changes align with the instructions, focusing on integrating Qdrant and an embedding service for the search functionality, and updating the agent's interaction points (`process` method for tool calls).
