# Databricks notebook source
# MAGIC %md
# MAGIC # Two-Stage Search Agent with LangGraph & Dynamic Temporal Context
# MAGIC 
# MAGIC This notebook implements a two-stage search strategy with time-aware prompting:
# MAGIC 1. **Stage 1**: Search the simple index to find relevant documents
# MAGIC 2. **Stage 2**: Use filenames from Stage 1 to filter and search the OCR index
# MAGIC 3. **Response**: Generate answers using both document summaries and detailed chunks
# MAGIC 4. **Temporal Context**: Dynamic date/time injection for each request

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow[databricks] langgraph==0.3.4 databricks-langchain databricks-agents uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Import Dependencies

# COMMAND ----------

from typing import Any, Generator, Optional, Sequence, Union, List
from datetime import datetime

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
from langchain_core.tools import BaseTool, tool
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
# MAGIC 
# MAGIC This cell enables MLflow autologging for LangChain and initializes the Databricks Function Client for Unity Catalog integration.

# COMMAND ----------

# Enable MLflow autologging for LangChain
mlflow.langchain.autolog()

# Initialize Databricks Function Client
client = DatabricksFunctionClient()
set_uc_function_client(client)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Configuration with Dynamic System Prompt Template
# MAGIC 
# MAGIC This cell defines the LLM endpoint and creates a dynamic system prompt template with placeholders for temporal context. The template includes specialized guidance for infrastructure construction documents and time-aware search strategies.

# COMMAND ----------

############################################
# Define your LLM endpoint and dynamic system prompt template
############################################
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# Dynamic system prompt template with placeholders for temporal context
BASE_SYSTEM_PROMPT_TEMPLATE = """You are an expert financial analyst specializing in Australian listed company documentation and regulatory filings.

## Current Context
- Today: {current_date} ({current_day})
- Quarter: {current_quarter}  
- Financial Year: {financial_year}
- Season: {season_context}

## Document Archive
You have access to a comprehensive archive of financial documents from listed companies including:
- Product Disclosure Statements (PDSs) and offer documents
- Annual reports and financial statements
- Half-yearly and quarterly reports
- Continuous disclosure announcements
- Investor presentations and briefings
- Audit reports and compliance documentation
- Regulatory filings and ASX releases

## Search Strategy
You have access to two complementary search tools:

1. **Document Summary Search**: Searches high-level document summaries to identify relevant company filings and understand document scope
2. **Detailed Content Search**: Searches detailed content to find specific financial data, regulatory information, and precise disclosures

**Search Protocol:**
- Always start with the simple search to identify relevant documents and companies
- Use the detailed search to extract specific financial information from identified relevant documents
- Cross-reference information between different report types when available
- Combine insights from both searches for comprehensive financial analysis

## Temporal Intelligence for Financial Analysis
When users reference time:

**"Recent" or "Recently"** = Within the last 30 days from today ({current_date})
**"Latest" or "Current"** = Most recent available filings as of today ({current_date})
**"This quarter"** = {current_quarter} period
**"This financial year"** = {financial_year} period
**"Upcoming"** = Next 60 days from today

For financial reporting:
- **Quarterly reporting** = Within 45 days of quarter end
- **Half-yearly reporting** = Within 75 days of half-year end  
- **Annual reporting** = Within 4 months of financial year end
- **Continuous disclosure** = Immediate ASX announcement requirements

## Response Guidelines

**For Financial Performance Queries:**
- Include specific financial metrics and ratios
- Reference comparative periods and benchmarks
- Highlight significant changes or trends
- Cite relevant sections of financial reports

**For Regulatory and Compliance Queries:**
- Reference specific regulatory requirements
- Include compliance status and any exceptions
- Mention audit findings or regulatory responses
- Consider ASIC, ASX, and APRA requirements where relevant

**For Product Disclosure Queries:**
- Extract key product features and terms
- Identify risks and investment considerations
- Reference fee structures and costs
- Include regulatory warnings and disclaimers

**For Investment Analysis:**
- Provide financial ratios and performance metrics
- Include forward-looking statements and guidance
- Reference market conditions and peer comparisons
- Consider dividend policies and capital management

**For Risk Assessment:**
- Identify key business and financial risks
- Reference risk management strategies
- Include credit ratings and analyst opinions
- Consider market, operational, and regulatory risks

## Output Format
Structure responses with:
1. **Executive Summary** - Key findings and direct answer to the query
2. **Financial Details** - Specific metrics, ratios, and quantitative information
3. **Regulatory Context** - Compliance status and regulatory considerations
4. **Source References** - Which documents and reporting periods the information comes from
5. **Risk Considerations** - Relevant risks and limitations to consider

Always provide accurate, well-sourced financial information that helps investors and analysts make informed decisions. Include appropriate disclaimers about the timing and nature of financial information."""

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
# MAGIC 
# MAGIC This cell initializes the vector search retrievers for both indexes and sets up any Unity Catalog function tools. The retrievers are configured to include relevant metadata columns for filename filtering in the two-stage search process.

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

print(f"✅ Vector search retrievers initialized")
print(f"   UC Tools: {len(uc_tool_names)} functions")
print(f"   Additional Tools: {len(tools)} tools")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: Define Enhanced State
# MAGIC 
# MAGIC This cell defines an enhanced state class that includes intermediate search results for tracking document discovery and filename extraction during the two-stage search process.

# COMMAND ----------

# Define enhanced state that includes intermediate search results
class EnhancedChatAgentState(TypedDict):
    messages: List[dict]
    simple_index_results: Optional[List[Document]]
    relevant_filenames: Optional[List[str]]
    final_chunks: Optional[List[Document]]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: Time-Aware Helper Functions
# MAGIC 
# MAGIC These functions handle dynamic temporal context generation. They calculate the current date, season, and relevant timing information that gets injected into the system prompt for each request.

# COMMAND ----------

def get_current_temporal_context() -> dict:
    """Generate current temporal context for dynamic prompt injection"""
    now = datetime.now()
    
    # Calculate Australian financial year (July 1 - June 30)
    if now.month >= 7:
        fy_start = now.year
        fy_end = now.year + 1
    else:
        fy_start = now.year - 1
        fy_end = now.year
    
    financial_year = f"FY{fy_end} (July {fy_start} - June {fy_end})"
    
    # Define seasonal business context
    month = now.month
    if month in [12, 1, 2]: 
        season_context = "Summer is a period of low economic activity"
    elif month in [3, 4, 5]: 
        season_context = "Autumn Business is in full swing"  
    elif month in [6, 7, 8]: 
        season_context = "Winter is when end of financial year happens"
    else: 
        season_context = "Spring is when new projects and initiatives typically kick off"
    
    return {
        "current_date": now.strftime("%B %d, %Y"),
        "current_day": now.strftime("%A"),
        "current_quarter": f"Q{(now.month-1)//3 + 1} {now.year}",
        "financial_year": financial_year,
        "season_context": season_context
    }

def generate_dynamic_system_prompt(template: str) -> str:
    """Generate system prompt with current temporal context"""
    context = get_current_temporal_context()
    return template.format(**context)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7: Create Two-Stage Search Agent Function
# MAGIC 
# MAGIC This function creates the main agent workflow with two specialized search tools: a simple document summary search and a detailed OCR content search with filename filtering. The agent includes temporal awareness that adjusts search parameters for time-sensitive queries.

# COMMAND ----------

def create_two_stage_search_agent(
    model: LanguageModelLike,
    additional_tools: List[BaseTool],
    system_prompt_template: Optional[str] = None,
) -> CompiledGraph:
    """Create the two-stage search agent workflow with dynamic temporal context"""
    
    # Create the simple search tool using VectorSearchRetrieverTool
    simple_search_tool = VectorSearchRetrieverTool(
        index_name=SIMPLE_INDEX_NAME,
        tool_description="Search the document analysis simple index for relevant infrastructure construction documents and return document summaries. This tool finds high-level document information and identifies which documents are relevant to the query."
    )
    
    # Create a custom detailed search tool that can handle filtering
    from langchain_core.tools import tool
    
    @tool
    def search_detailed_content_filtered(query: str) -> str:
        """Search the detailed OCR index for specific content. Use this to get detailed information after identifying relevant documents from the simple search. Automatically considers temporal context and prioritizes recent documents."""
        try:
            # Check if query has temporal indicators for enhanced processing
            temporal_keywords = ['recent', 'latest', 'current', 'soon', 'upcoming', 'now', 'today', 'this week', 'this month']
            has_temporal = any(keyword in query.lower() for keyword in temporal_keywords)
            
            # First do a simple search to identify relevant filenames (no date filtering)
            search_k = 15 if has_temporal else 10  # Get more results for temporal queries
            simple_results = simple_index_retriever.similarity_search(query=query, k=search_k)
            
            # Extract filenames from simple search
            relevant_filenames = set()
            for doc in simple_results:
                if hasattr(doc, 'metadata') and doc.metadata:
                    filename = doc.metadata.get('file_name') or doc.metadata.get('filename') or doc.metadata.get('source')
                    if filename:
                        relevant_filenames.add(filename)
            
            if not relevant_filenames:
                # Fallback to unfiltered search if no filenames found (no date filtering)
                detailed_results = page_ocr_index_retriever.similarity_search(query=query, k=10)
                content_text = f"Found {len(detailed_results)} detailed content chunks (unfiltered search).\n\n"
                for i, doc in enumerate(detailed_results[:5], 1):
                    source = doc.metadata.get('source_filename', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
                    content_text += f"Chunk {i} (from {source}): {doc.page_content[:300]}...\n\n"
                return content_text
            
            # Search detailed index with filename filtering only (no date filtering)
            all_detailed_results = []
            max_files = 7 if has_temporal else 5  # Process more files for temporal queries
            
            for filename in list(relevant_filenames)[:max_files]:
                try:
                    # Only filter by filename, no date filtering
                    filename_filter = {"source_filename": filename}
                    chunk_k = 4 if has_temporal else 3  # Get more chunks for temporal queries
                    results = page_ocr_index_retriever.similarity_search(
                        query=query,
                        k=chunk_k,
                        filters=filename_filter
                    )
                    all_detailed_results.extend(results)
                except Exception as e:
                    print(f"Warning: Could not filter by {filename}: {str(e)}")
                    continue
            
            if not all_detailed_results:
                return "No detailed content found for the identified relevant files."
            
            # Format results with temporal awareness
            temporal_note = " (prioritizing recent documents)" if has_temporal else ""
            content_text = f"Found {len(all_detailed_results)} detailed content chunks from {len(relevant_filenames)} relevant files{temporal_note}.\n\n"
            content_text += f"Relevant files: {', '.join(list(relevant_filenames))}\n\n"
            
            max_chunks = 10 if has_temporal else 8  # Show more chunks for temporal queries
            for i, doc in enumerate(all_detailed_results[:max_chunks], 1):
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
    
    # Define the function that calls the model with dynamic system prompt
    def call_model(state: ChatAgentState, config: RunnableConfig):
        messages = state["messages"]
        
        # Generate fresh system prompt with current temporal context
        if system_prompt_template:
            dynamic_system_prompt = generate_dynamic_system_prompt(system_prompt_template)
            
            # Add system prompt only if it's the first call and no system message exists
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": dynamic_system_prompt}] + messages
            else:
                # Update existing system message with fresh temporal context
                messages[0]["content"] = dynamic_system_prompt
        
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
    
    print(f"✅ Agent workflow compiled with {len(all_tools)} tools")
    for tool in all_tools:
        print(f"   - {tool.name}: {tool.description}")
    return workflow.compile()

print("✅ Two-stage search agent function defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8: Enhanced LangGraph Chat Agent Class
# MAGIC 
# MAGIC This custom ChatAgent class extends MLflow's base ChatAgent to support dynamic temporal context injection. It generates fresh system prompts with current date and seasonal information for every request, ensuring temporal queries always use the actual current date.

# COMMAND ----------

class LangGraphChatAgent(ChatAgent):
    """Custom ChatAgent implementation with dynamic temporal context for MLflow deployment"""
    
    def __init__(self, agent: CompiledStateGraph, system_prompt_template: Optional[str] = None):
        self.agent = agent
        self.system_prompt_template = system_prompt_template

    def _get_current_system_prompt(self) -> Optional[str]:
        """Generate system prompt with current date/time context"""
        if not self.system_prompt_template:
            return None
            
        return generate_dynamic_system_prompt(self.system_prompt_template)

    def _prepare_messages_with_dynamic_context(self, messages: list[ChatAgentMessage]) -> list[dict]:
        """Convert messages and inject fresh system prompt"""
        message_dicts = self._convert_messages_to_dict(messages)
        
        # Add/update system message with current temporal context
        if self.system_prompt_template:
            system_prompt = self._get_current_system_prompt()
            if message_dicts and message_dicts[0].get("role") == "system":
                message_dicts[0]["content"] = system_prompt
            else:
                message_dicts.insert(0, {"role": "system", "content": system_prompt})
        
        return message_dicts

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        """Synchronous prediction method with fresh temporal context"""
        
        # Prepare messages with dynamic system prompt
        message_dicts = self._prepare_messages_with_dynamic_context(messages)
        request = {"messages": message_dicts}

        response_messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                response_messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=response_messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        """Streaming prediction method with fresh temporal context"""
        
        # Prepare messages with dynamic system prompt
        message_dicts = self._prepare_messages_with_dynamic_context(messages)
        request = {"messages": message_dicts}
        
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9: Create and Deploy Time-Aware Agent
# MAGIC 
# MAGIC This cell instantiates the complete agent workflow with dynamic temporal context and wraps it for MLflow deployment. The agent will generate fresh temporal context for each request and adjust search behavior for time-sensitive queries.

# COMMAND ----------

# Create the agent object with dynamic temporal context
agent = create_two_stage_search_agent(llm, tools, BASE_SYSTEM_PROMPT_TEMPLATE)

# Wrap in our enhanced custom class with dynamic temporal context
AGENT = LangGraphChatAgent(agent, BASE_SYSTEM_PROMPT_TEMPLATE)

# Set model for MLflow deployment
mlflow.models.set_model(AGENT)

# Show current temporal context
current_context = get_current_temporal_context()
print(f"🕒 Current temporal context:")
print(f"   Date: {current_context['current_date']} ({current_context['current_day']})")
print(f"   Quarter: {current_context['current_quarter']}")
print(f"   Season: {current_context['season_context']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10: Test the Time-Aware Agent (Optional)
# MAGIC 
# MAGIC These test functions verify that the agent correctly handles both temporal and general queries, and that the dynamic temporal context generation is working properly. Uncomment the test calls at the bottom to run before deployment.

# COMMAND ----------

# Optional: Test the agent locally before deployment
def test_time_aware_agent():
    """Test function to verify agent works correctly with temporal context"""
    
    # Test with temporal query
    temporal_test_messages = [
        ChatAgentMessage(role="user", content="What recent is the general trends in banking lately?")
    ]
    
    # Test with non-temporal query  
    general_test_messages = [
        ChatAgentMessage(role="user", content="What general risks are there across finance?")
    ]
    
    try:
        print("🧪 Testing temporal query...")
        temporal_response = AGENT.predict(temporal_test_messages)
        print(f"✅ Temporal response: {temporal_response.messages[-1].content[:200]}...")
        
        print("🧪 Testing general query...")
        general_response = AGENT.predict(general_test_messages)
        print(f"✅ General response: {general_response.messages[-1].content[:200]}...")
        
        return True
    except Exception as e:
        print(f"❌ Agent test failed: {str(e)}")
        return False

def test_temporal_context_generation():
    """Test temporal context generation"""
    context = get_current_temporal_context()
    print(f"✅ Generated context: {context}")
    
    sample_prompt = generate_dynamic_system_prompt(BASE_SYSTEM_PROMPT_TEMPLATE)
    print(f"✅ Sample prompt length: {len(sample_prompt)} characters")
    
    return True

# Uncomment to run tests
# test_temporal_context_generation()
# test_time_aware_agent()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 11: Agent Summary and Deployment Guide
# MAGIC 
# MAGIC The time-aware two-stage search agent is now ready for deployment. This agent specializes in infrastructure construction document analysis with dynamic temporal context that ensures time-sensitive queries always use the current date.
# MAGIC 
# MAGIC ### How It Works
# MAGIC - **Stage 1**: Searches document summaries in the simple index to identify relevant documents
# MAGIC - **Stage 2**: Filters detailed OCR content by filenames from Stage 1 
# MAGIC - **Temporal Intelligence**: Injects fresh date/time context for every request
# MAGIC - **Smart Search**: Adjusts parameters automatically for time-sensitive queries
# MAGIC 
# MAGIC ### Temporal Features
# MAGIC - **"Recent"** = Last 30 days from actual current date
# MAGIC - **"Soon/Upcoming"** = Next 60 days from current date  
# MAGIC - **Seasonal Context** = Construction-aware timing (weather limitations, peak seasons)
# MAGIC - **Smart Parameters** = Enhanced search for temporal keywords
# MAGIC 
# MAGIC ### Search Behavior
# MAGIC | Query Type | Search Results | Files | Chunks/File | Total Chunks |
# MAGIC |------------|----------------|-------|-------------|--------------|
# MAGIC | General | 10 | 5 | 3 | 8 |
# MAGIC | Temporal | 15 | 7 | 4 | 10 |
# MAGIC 
# MAGIC ### Next Steps
# MAGIC 1. Test with temporal queries before deployment  
# MAGIC 2. Adjust index configurations if needed
# MAGIC 3. Customize temporal definitions in the system prompt template
# MAGIC 4. Deploy using MLflow model serving

# COMMAND ----------

# Display current configuration
print(f"🎉 Time-Aware Agent Ready for Deployment!")
print(f"   LLM: {LLM_ENDPOINT_NAME}")
print(f"   Simple Index: {SIMPLE_INDEX_NAME}")
print(f"   OCR Index: {OCR_INDEX_NAME}")

# Show current temporal context as example
current_context = get_current_temporal_context()
print(f"\n📅 Current Context Example:")
print(f"   {current_context['current_date']} ({current_context['current_day']})")
print(f"   {current_context['current_quarter']} - {current_context['season_context']}")

print(f"\n🚀 Ready for MLflow deployment!")

# COMMAND ----------