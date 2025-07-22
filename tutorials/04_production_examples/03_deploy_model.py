# Databricks notebook source
# MAGIC %md
# MAGIC #Tool-calling Agent
# MAGIC
# MAGIC This notebook uses [Mosaic AI Agent Framework](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/build-genai-apps) to recreate your agent from the AI Playground. It  demonstrates how to develop, manually test, evaluate, log, and deploy a tool-calling agent in LangGraph.
# MAGIC
# MAGIC The agent code implements [MLflow's ChatAgent](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatAgent) interface, a Databricks-recommended open-source standard that simplifies authoring multi-turn conversational agents, and is fully compatible with Mosaic AI agent framework functionality.
\
# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow-skinny[databricks] langgraph==0.3.4 databricks-langchain databricks-agents uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Configuring Index and Endpoint Settings

# COMMAND ----------

import mlflow
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint, DatabricksVectorSearchIndex
from pkg_resources import get_distribution

CATALOG = 'brian_gen_ai'
SCHEMA = 'parsing_test'

LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
EMBEDDING_ENDPOINT_NAME = "databricks-gte-large-en"
SIMPLE_INDEX_NAME = f"{CATALOG}.{SCHEMA}.document_analysis_simple_index"
OCR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.document_page_ocr_index"

# COMMAND ----------

# MAGIC %md
# MAGIC # Logging and Model Registration 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Input Example and Resources for on-behalf-of 

# COMMAND ----------

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "Hello how are you?"
        }
    ]
}

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
             DatabricksVectorSearchIndex(index_name=SIMPLE_INDEX_NAME),
             DatabricksVectorSearchIndex(index_name=OCR_INDEX_NAME),
             DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Model As Code 

# COMMAND ----------

with mlflow.start_run() as run:
    
    logging_run_id = run.info.run_id    
    
    logged_agent_info = mlflow.pyfunc.log_model(
        name="compound_agent",
        python_model="Building A Compound Chain",
        input_example=input_example,
        resources=resources,
        pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"mlflow=={get_distribution('mlflow').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
            f"langgraph=={get_distribution('langgraph').version}",
        ],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Logged Model    
    
# COMMAND ----------
    
# Test the model
chain_input = {
    "messages": [
        {
            "role": "user",
            "content": "How did cba perform?", # Replace with a question relevant to your use case
        }
    ]
}
chain = mlflow.pyfunc.load_model(logged_agent_info.model_uri)
chain.predict(chain_input)


# COMMAND ----------

# MAGIC %md
# MAGIC # Assess and Evaluate Our Model

# COMMAND ----------

# Concise MLflow Evaluation for LangGraph Agent

from mlflow.genai.scorers import RelevanceToQuery, Safety, RetrievalRelevance, RetrievalGroundedness
from mlflow.types.agent import ChatAgentMessage
import mlflow

# Evaluation dataset - inputs must be dict with param names as keys
eval_dataset = [
    {
        "inputs": {
            "messages": [{"role": "user", "content": "Tell me about how CBA is performing?"}]
        },
        "expected_response": None
    },
    {
        "inputs": {
            "messages": [{"role": "user", "content": "What are the key financial highlights?"}]
        },
        "expected_response": None
    }
]

# Prediction function - parameter name must match inputs dictionary key
def predict_with_agent(messages):
    try:
        chat_messages = [ChatAgentMessage(role=msg["role"], content=msg["content"]) 
                        for msg in messages]
        response = chain.predict(chat_messages)
        assistant_messages = [msg for msg in response.messages if msg.role == "assistant"]
        return assistant_messages[-1].content if assistant_messages else "No response"
    except Exception as e:
        return f"Error: {str(e)}"

# Run evaluation
eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=predict_with_agent,
    scorers=[RelevanceToQuery(), Safety(), RetrievalRelevance(), RetrievalGroundedness()],
    model_id=logged_agent_info.model_id
)

print("✅ Evaluation completed!")
print(eval_results)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
model_name = "baseline_chain"

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents

agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version)