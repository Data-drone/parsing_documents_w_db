# Databricks notebook source
# MAGIC %md
# MAGIC # Document Summarisation
# MAGIC In order to best find the right document, we should summarise each
# MAGIC

# COMMAND ----------

# MAGIC %pip install -U mlflow pymupdf4llm databricks-langchain --quiet
# MAGIC %restart_python

# COMMAND ----------

### Config Vars

CATALOG = "brian_gen_ai"
SCHEMA = "parsing_test"
TABLE = "document_store"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Analysis Function

# COMMAND ----------

# document list
document_table = spark.table(f"{CATALOG}.{SCHEMA}.{TABLE}")

display(document_table)

# COMMAND ----------

# 

# COMMAND ----------

import os
import pymupdf4llm
from databricks_langchain import ChatDatabricks
from typing import Optional

def process_pdf_with_databricks(
    file_path: str,
    databricks_endpoint: str = "databricks-dbrx-instruct",
    databricks_host: Optional[str] = None,
    databricks_token: Optional[str] = None,
    prompt_template: Optional[str] = None
) -> str:
    """
    Process a PDF file using PyMuPDF4LLM and Databricks LangChain model.
    
    Args:
        file_path (str): Path to the PDF file to process
        databricks_endpoint (str): Databricks model endpoint name
        databricks_host (str, optional): Databricks workspace URL
        databricks_token (str, optional): Databricks access token
        prompt_template (str, optional): Custom prompt template for processing
        
    Returns:
        str: Processed results from the Databricks model
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        Exception: For other processing errors
    """
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract markdown from PDF using pymupdf4llm
        print(f"Extracting text from {file_path}...")
        markdown_text = pymupdf4llm.to_markdown(file_path)
        
        if not markdown_text.strip():
            return "No text content extracted from the PDF file."
        
        # Set up Databricks credentials if not provided
        # if databricks_host is None:
        #     databricks_host = os.getenv("DATABRICKS_HOST")
        # if databricks_token is None:
        #     databricks_token = os.getenv("DATABRICKS_TOKEN")
            
        # if not databricks_host or not databricks_token:
        #     raise ValueError(
        #         "Databricks credentials required. Set DATABRICKS_HOST and DATABRICKS_TOKEN "
        #         "environment variables or pass them as parameters."
        #     )
        
        # Initialize Databricks LLM
        print("Initializing Databricks model...")
        llm = ChatDatabricks(
            endpoint=databricks_endpoint,
            databricks_host=databricks_host,
            databricks_token=databricks_token,
            max_tokens=4000,
            temperature=0.1
        )
        
        # Prepare the prompt
        if prompt_template is None:
            prompt_template = """
Please analyze and summarize the following document content:

{content}

Provide a succint analysis including:
1. Key Topics
2. Overall Summary
3. Why someone would need to check this document

Analysis:"""
        
        prompt = prompt_template.format(content=markdown_text)
        
        # Process with Databricks model
        print("Processing with Databricks model...")
        result = llm.invoke(prompt)
        
        return result
        
    except FileNotFoundError:
        raise
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")

# COMMAND ----------

# Test out the responses
result = process_pdf_with_databricks(
    file_path="/Volumes/brian_gen_ai/parsing_test/raw_data/nab-home-and-contents-insurance-pds.pdf",
    databricks_endpoint="databricks-claude-sonnet-4"
)

display(result.content)

# COMMAND ----------

# Scale up the summarisation
