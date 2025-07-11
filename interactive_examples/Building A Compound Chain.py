# Databricks notebook source
# MAGIC %md
# MAGIC # Two-Stage Search Agent with LangGraph
# MAGIC 
# MAGIC This notebook implements a two-stage search strategy:
# MAGIC 1. **Stage 1**: Search the simple index to find relevant documents
# MAGIC 2. **Stage 2**: Use filenames from Stage 1 to filter and search the OCR index
# MAGIC 3. **Response**: Generate answers using both document summaries and detailed chunks

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow[databricks] langgraph==0.3.4 databricks-langchain databricks-agents uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Import Dependencies

# COMMAND ----------

from typing import Any, Generator, Optional, Sequence, Union, List

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool,
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
    DatabricksVectorSearch,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from typing_extensions import TypedDict

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Initialize MLflow and Databricks Client

# COMMAND ----------

# Enable MLflow autologging for LangChain
mlflow.langchain.autolog()

# Initialize Databricks Function Client
client = DatabricksFunctionClient()
set_uc_function_client(client)

print("✅ MLflow and Databricks client initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Configuration

# COMMAND ----------

############################################
# Define your LLM endpoint and system prompt
############################################
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# Define your system prompt here
system_prompt = """You are a helpful assistant that can search through documents to answer questions. 

You have access to two search tools:
1. A simple document search tool that searches document summaries
2. A detailed OCR search tool that searches detailed page content

To answer questions effectively:
- Use the simple search first to understand which documents are relevant
- Then use the detailed search to get specific information
- Combine information from both searches in your final answer

Always search both indexes to provide comprehensive answers."""


# Configure vector search indexes
SIMPLE_INDEX_NAME = "brian_gen_ai.parsing_test.document_analysis_simple_index"
OCR_INDEX_NAME = "brian_gen_ai.parsing_test.document_page_ocr_index"

print(f"✅ Configuration set:")
print(f"   LLM Endpoint: {LLM_ENDPOINT_NAME}")
print(f"   Simple Index: {SIMPLE_INDEX_NAME}")
print(f"   OCR Index: {OCR_INDEX_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: Setup Tools and Vector Search

# COMMAND ----------

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
###############################################################################
tools = []

# You can use UDFs in Unity Catalog as agent tools
uc_tool_names = []
uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
tools.extend(uc_toolkit.tools)

# Create vector search retrievers for the two-stage search
simple_index_retriever = DatabricksVectorSearch(
    index_name=SIMPLE_INDEX_NAME,
    columns=["file_name"]  # Include file_name for filtering
)

page_ocr_index_retriever = DatabricksVectorSearch(
    index_name=OCR_INDEX_NAME,
    columns=["source_filename"]
)

print("✅ Vector search retrievers initialized successfully")
print(f"   UC Tools: {len(uc_tool_names)} functions")
print(f"   Additional Tools: {len(tools)} tools")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: Define Enhanced State

# COMMAND ----------

# Define enhanced state that includes intermediate search results
class EnhancedChatAgentState(TypedDict):
    messages: List[dict]
    simple_index_results: Optional[List[Document]]
    relevant_filenames: Optional[List[str]]
    final_chunks: Optional[List[Document]]

print("✅ Enhanced state class defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: Define Helper Functions

# COMMAND ----------

def create_two_stage_search_agent(
    model: LanguageModelLike,
    additional_tools: List[BaseTool],
    system_prompt: Optional[str] = None,
) -> CompiledGraph:
    """Create the two-stage search agent workflow"""
    
    # Create the simple search tool using VectorSearchRetrieverTool
    simple_search_tool = VectorSearchRetrieverTool(
        index_name=SIMPLE_INDEX_NAME,
        tool_description="Search the document analysis simple index for relevant documents and return document summaries. This tool finds high-level document information and identifies which documents are relevant to the query."
    )
    
    # Create a custom detailed search tool that can handle filtering
    from langchain_core.tools import tool
    
    @tool
    def search_detailed_content_filtered(query: str) -> str:
        """Search the detailed OCR index for specific content. Use this to get detailed information after identifying relevant documents from the simple search."""
        try:
            # First do a simple search to identify relevant filenames
            simple_results = simple_index_retriever.similarity_search(query=query, k=10)
            
            # Extract filenames from simple search
            relevant_filenames = set()
            for doc in simple_results:
                if hasattr(doc, 'metadata') and doc.metadata:
                    filename = doc.metadata.get('file_name') or doc.metadata.get('filename') or doc.metadata.get('source')
                    if filename:
                        relevant_filenames.add(filename)
            
            if not relevant_filenames:
                # Fallback to unfiltered search if no filenames found
                detailed_results = page_ocr_index_retriever.similarity_search(query=query, k=10)
                content_text = f"Found {len(detailed_results)} detailed content chunks (unfiltered search).\n\n"
                for i, doc in enumerate(detailed_results[:5], 1):
                    source = doc.metadata.get('source_filename', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
                    content_text += f"Chunk {i} (from {source}): {doc.page_content[:300]}...\n\n"
                return content_text
            
            # Search detailed index with filename filtering
            all_detailed_results = []
            for filename in list(relevant_filenames)[:5]:  # Limit to top 5 files
                try:
                    filename_filter = {"source_filename": filename}
                    results = page_ocr_index_retriever.similarity_search(
                        query=query,
                        k=3,
                        filters=filename_filter
                    )
                    all_detailed_results.extend(results)
                except Exception as e:
                    print(f"Warning: Could not filter by {filename}: {str(e)}")
                    continue
            
            if not all_detailed_results:
                return "No detailed content found for the identified relevant files."
            
            # Format results
            content_text = f"Found {len(all_detailed_results)} detailed content chunks from {len(relevant_filenames)} relevant files.\n\n"
            content_text += f"Relevant files: {', '.join(list(relevant_filenames))}\n\n"
            
            for i, doc in enumerate(all_detailed_results[:8], 1):
                source = doc.metadata.get('source_filename', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
                content_text += f"Chunk {i} (from {source}): {doc.page_content[:400]}...\n\n"
            
            return content_text
            
        except Exception as e:
            return f"Error searching detailed content: {str(e)}"
    
    # Combine all tools
    all_tools = [simple_search_tool, search_detailed_content_filtered] + additional_tools
    
    # Bind tools to model
    model_with_tools = model.bind_tools(all_tools)
    
    # Define the function that determines whether to continue or end
    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"
    
    # Define the function that calls the model
    def call_model(state: ChatAgentState, config: RunnableConfig):
        messages = state["messages"]
        
        # Add system prompt only if it's the first call and no system message exists
        if system_prompt and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        response = model_with_tools.invoke(messages, config)
        return {"messages": [response]}
    
    # Build the workflow using the standard ChatAgent pattern
    workflow = StateGraph(ChatAgentState)
    
    # Add nodes
    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(all_tools))
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")
    
    print("✅ Agent workflow compiled successfully")
    print(f"   Tools available: {len(all_tools)}")
    for tool in all_tools:
        print(f"   - {tool.name}: {tool.description}")
    return workflow.compile()

print("✅ Two-stage search agent function defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7: Define Agent Logic Functions

# COMMAND ----------

class LangGraphChatAgent(ChatAgent):
    """Custom ChatAgent implementation for MLflow deployment"""
    
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent
        print("✅ LangGraphChatAgent initialized")

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        """Synchronous prediction method"""
        request = {"messages": self._convert_messages_to_dict(messages)}

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        """Streaming prediction method"""
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                )

print("✅ LangGraphChatAgent class defined")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8: Create Two-Stage Search Agent Function

# COMMAND ----------

def create_two_stage_search_agent(
    model: LanguageModelLike,
    additional_tools: List[BaseTool],
    system_prompt: Optional[str] = None,
) -> CompiledGraph:
    """Create the two-stage search agent workflow"""
    
    # Create the simple search tool using VectorSearchRetrieverTool
    simple_search_tool = VectorSearchRetrieverTool(
        index_name=SIMPLE_INDEX_NAME,
        tool_description="Search the document analysis simple index for relevant documents and return document summaries. This tool finds high-level document information and identifies which documents are relevant to the query."
    )
    
    # Create a custom detailed search tool that can handle filtering
    from langchain_core.tools import tool
    
    @tool
    def search_detailed_content_filtered(query: str) -> str:
        """Search the detailed OCR index for specific content. Use this to get detailed information after identifying relevant documents from the simple search."""
        try:
            # First do a simple search to identify relevant filenames
            simple_results = simple_index_retriever.similarity_search(query=query, k=10)
            
            # Extract filenames from simple search
            relevant_filenames = set()
            for doc in simple_results:
                if hasattr(doc, 'metadata') and doc.metadata:
                    filename = doc.metadata.get('file_name') or doc.metadata.get('filename') or doc.metadata.get('source')
                    if filename:
                        relevant_filenames.add(filename)
            
            if not relevant_filenames:
                # Fallback to unfiltered search if no filenames found
                detailed_results = page_ocr_index_retriever.similarity_search(query=query, k=10)
                content_text = f"Found {len(detailed_results)} detailed content chunks (unfiltered search).\n\n"
                for i, doc in enumerate(detailed_results[:5], 1):
                    source = doc.metadata.get('source_filename', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
                    content_text += f"Chunk {i} (from {source}): {doc.page_content[:300]}...\n\n"
                return content_text
            
            # Search detailed index with filename filtering
            all_detailed_results = []
            for filename in list(relevant_filenames)[:5]:  # Limit to top 5 files
                try:
                    filename_filter = {"source_filename": filename}
                    results = page_ocr_index_retriever.similarity_search(
                        query=query,
                        k=3,
                        filters=filename_filter
                    )
                    all_detailed_results.extend(results)
                except Exception as e:
                    print(f"Warning: Could not filter by {filename}: {str(e)}")
                    continue
            
            if not all_detailed_results:
                return "No detailed content found for the identified relevant files."
            
            # Format results
            content_text = f"Found {len(all_detailed_results)} detailed content chunks from {len(relevant_filenames)} relevant files.\n\n"
            content_text += f"Relevant files: {', '.join(list(relevant_filenames))}\n\n"
            
            for i, doc in enumerate(all_detailed_results[:8], 1):
                source = doc.metadata.get('source_filename', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
                content_text += f"Chunk {i} (from {source}): {doc.page_content[:400]}...\n\n"
            
            return content_text
            
        except Exception as e:
            return f"Error searching detailed content: {str(e)}"
    
    # Combine all tools
    all_tools = [simple_search_tool, search_detailed_content_filtered] + additional_tools
    
    # Bind tools to model
    model_with_tools = model.bind_tools(all_tools)
    
    # Define the function that determines whether to continue or end
    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"
    
    # Define the function that calls the model
    def call_model(state: ChatAgentState, config: RunnableConfig):
        messages = state["messages"]
        
        # Add system prompt only if it's the first call and no system message exists
        if system_prompt and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        response = model_with_tools.invoke(messages, config)
        return {"messages": [response]}
    
    # Build the workflow using the standard ChatAgent pattern
    workflow = StateGraph(ChatAgentState)
    
    # Add nodes
    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(all_tools))
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")
    
    print("✅ Agent workflow compiled successfully")
    print(f"   Tools available: {len(all_tools)}")
    for tool in all_tools:
        print(f"   - {tool.name}: {tool.description}")
    return workflow.compile()

print("✅ Two-stage search agent function defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9: Define LangGraph Chat Agent Class

# COMMAND ----------

class LangGraphChatAgent(ChatAgent):
    """Custom ChatAgent implementation for MLflow deployment"""
    
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent
        print("✅ LangGraphChatAgent initialized")

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        """Synchronous prediction method"""
        request = {
            "messages": self._convert_messages_to_dict(messages),
            "simple_index_results": None,
            "relevant_filenames": None,
            "final_chunks": None,
        }

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        """Streaming prediction method"""
        request = {
            "messages": self._convert_messages_to_dict(messages),
            "simple_index_results": None,
            "relevant_filenames": None,
            "final_chunks": None,
        }
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                )

print("✅ LangGraphChatAgent class defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10: Create and Deploy Agent

# COMMAND ----------

# Create the agent object
print("🚀 Creating two-stage search agent...")
agent = create_two_stage_search_agent(llm, tools, system_prompt)

# Wrap in our custom class
print("📦 Wrapping agent for MLflow deployment...")
AGENT = LangGraphChatAgent(agent)

# Set model for MLflow deployment
print("🔧 Setting agent as MLflow model...")
mlflow.models.set_model(AGENT)

print("✅ Agent created and registered successfully!")
print("\n📋 Agent workflow:")
print("1. 🤖 LLM receives user question")
print("2. 🔍 LLM calls search_simple_documents tool")
print("3. 📄 Tool returns document summaries and filenames") 
print("4. 🎯 LLM calls search_detailed_content tool with filenames")
print("5. 📋 Tool returns filtered detailed content")
print("6. 💬 LLM synthesizes final response")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 11: Test the Agent (Optional)

# COMMAND ----------

# Optional: Test the agent locally before deployment
def test_agent():
    """Test function to verify agent works correctly"""
    test_messages = [
        ChatAgentMessage(role="user", content="What information do you have about project timelines?")
    ]
    
    try:
        response = AGENT.predict(test_messages)
        print("✅ Agent test successful!")
        print(f"Response: {response.messages[-1].content[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Agent test failed: {str(e)}")
        return False

# Uncomment to test
# test_agent()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 12: Model Information and Next Steps

# COMMAND ----------

print("🎉 Two-Stage Search Agent Setup Complete!")
print("\n📖 How it works:")
print("• Stage 1: Searches document_analysis_simple_index for relevant documents")
print("• Stage 2: Filters document_page_ocr_index by source_filename from Stage 1")
print("• Combines results from both indexes for comprehensive responses")
print("\n🔧 Configuration:")
print(f"• LLM Endpoint: {LLM_ENDPOINT_NAME}")
print(f"• Simple Index: {SIMPLE_INDEX_NAME}")
print(f"• OCR Index: {OCR_INDEX_NAME}")
print("\n📝 Next Steps:")
print("1. Adjust index column names if needed in Cell 4")
print("2. Modify search parameters (k values) in Cell 6")
print("3. Customize system prompt in Cell 3")
print("4. Deploy using MLflow model serving")
print("\n💡 To export this notebook:")
print("• File > Export > Source File (.py)")
print("• File > Export > DBC Archive (.dbc)")

# COMMAND ----------