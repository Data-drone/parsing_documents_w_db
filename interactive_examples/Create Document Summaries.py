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

# MAGIC %md
# MAGIC # Build out the parsing functions

# COMMAND ----------

import os
import re
import pandas as pd
import pymupdf4llm
from databricks_langchain import ChatDatabricks
from typing import Optional, Tuple, List

def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (approximately 4 characters per token).
    """
    return len(text) // 4


def chunk_markdown_by_sections(markdown_text: str, max_chunk_tokens: int = 100000) -> List[str]:
    """
    Intelligently chunk markdown by sections/headers while respecting token limits.
    
    Args:
        markdown_text: The markdown text to chunk
        max_chunk_tokens: Maximum tokens per chunk (leaving room for prompt)
        
    Returns:
        List of markdown chunks
    """
    # Split by major headers first (# and ##)
    sections = re.split(r'\n(?=#{1,2}\s)', markdown_text)
    
    chunks = []
    current_chunk = ""
    
    for section in sections:
        section_tokens = estimate_tokens(section)
        current_tokens = estimate_tokens(current_chunk)
        
        # If adding this section would exceed limit, start new chunk
        if current_tokens + section_tokens > max_chunk_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = section
        else:
            current_chunk += ("\n" if current_chunk else "") + section
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # If we still have oversized chunks, split them by paragraphs
    final_chunks = []
    for chunk in chunks:
        if estimate_tokens(chunk) > max_chunk_tokens:
            final_chunks.extend(chunk_markdown_by_paragraphs(chunk, max_chunk_tokens))
        else:
            final_chunks.append(chunk)
    
    return final_chunks


def chunk_markdown_by_paragraphs(markdown_text: str, max_chunk_tokens: int = 100000) -> List[str]:
    """
    Fallback chunking by paragraphs when section-based chunking still produces oversized chunks.
    """
    paragraphs = markdown_text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph_tokens = estimate_tokens(paragraph)
        current_tokens = estimate_tokens(current_chunk)
        
        # If single paragraph is too large, truncate it
        if paragraph_tokens > max_chunk_tokens:
            paragraph = paragraph[:max_chunk_tokens * 4]  # Rough character limit
        
        if current_tokens + paragraph_tokens > max_chunk_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += ("\n\n" if current_chunk else "") + paragraph
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def process_large_document(
    markdown_text: str,
    llm,
    prompt_template: str,
    max_tokens: int = 120000  # Leave room for prompt and response
) -> str:
    """
    Process a large document by chunking and combining results.
    
    Args:
        markdown_text: The full markdown text
        llm: The configured LLM instance
        prompt_template: The prompt template
        max_tokens: Maximum tokens per chunk (including prompt)
        
    Returns:
        Combined analysis result
    """
    # Estimate prompt overhead (roughly)
    prompt_overhead = estimate_tokens(prompt_template.replace("{content}", ""))
    max_content_tokens = max_tokens - prompt_overhead - 4000  # Leave room for response
    
    # Check if document fits in one chunk
    if estimate_tokens(markdown_text) <= max_content_tokens:
        prompt = prompt_template.format(content=markdown_text)
        return llm.invoke(prompt)
    
    # Chunk the document
    chunks = chunk_markdown_by_sections(markdown_text, max_content_tokens)
    
    # Process each chunk
    chunk_results = []
    for i, chunk in enumerate(chunks, 1):
        chunk_prompt = f"""
Analyze this document chunk ({i}/{len(chunks)}):

{chunk}

Provide analysis focusing on:
1. Key Topics in this section
2. Important information
3. Notable details

Analysis:"""
        
        try:
            result = llm.invoke(chunk_prompt)
            chunk_results.append(f"--- Chunk {i}/{len(chunks)} ---\n{result}")
        except Exception as e:
            chunk_results.append(f"--- Chunk {i}/{len(chunks)} ---\nError processing chunk: {str(e)}")
    
    # Combine results with a summary
    combined_analysis = "\n\n".join(chunk_results)
    
    # If we have multiple chunks, create a final summary
    if len(chunks) > 1:
        summary_prompt = f"""
Based on the following analyses of different sections of a document, provide a comprehensive summary:

{combined_analysis}

Provide a unified analysis including:
1. Overall Key Topics across all sections
2. Main insights and findings
3. Why someone would need to check this document
4. Comprehensive summary

Final Analysis:"""
        
        # Check if summary prompt fits
        if estimate_tokens(summary_prompt) <= max_tokens - 4000:
            try:
                final_summary = llm.invoke(summary_prompt)
                return f"{final_summary}\n\n--- Detailed Section Analysis ---\n{combined_analysis}"
            except Exception as e:
                return f"Combined analysis from {len(chunks)} chunks:\n\n{combined_analysis}"
        else:
            return f"Combined analysis from {len(chunks)} chunks:\n\n{combined_analysis}"
    
    return combined_analysis


# COMMAND ----------

# Test out the responses
markdown_text, result = extract_pdf_to_markdown_udf(
    file_path=pd.Series(["/Volumes/brian_gen_ai/parsing_test/raw_data/nab-home-and-contents-insurance-pds.pdf"])
)

print(result)



# COMMAND ----------

# Scale up the summarisation
# Create the Pandas UDF
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import pandas_udf, col, lit

max_tokens = 128000
databricks_endpoint = 'databricks-meta-llama-3-3-70b-instruct'

# =============================================================================
# UDF 1: PDF to Markdown Extraction
# =============================================================================

@pandas_udf(
    returnType=StructType([
        StructField("file_path", StringType(), True),
        StructField("markdown_text", StringType(), True),
        StructField("estimated_tokens", StringType(), True),
        StructField("extraction_status", StringType(), True),
        StructField("error_message", StringType(), True)
    ])
)
def extract_pdf_to_markdown_udf(file_paths: pd.Series) -> pd.DataFrame:
    """
    UDF to extract markdown from PDF files.
    Fast I/O-bound operation that can be done independently.
    """
    results = []
    
    for file_path in file_paths:
        # Handle DBFS paths
        processed_path = file_path.replace('dbfs:', '') if file_path.startswith('dbfs:') else file_path
        
        try:
            # Check if file exists
            if not os.path.exists(processed_path):
                results.append({
                    "file_path": file_path,
                    "markdown_text": "",
                    "estimated_tokens": "0",
                    "extraction_status": "FAILED",
                    "error_message": f"File not found: {processed_path}"
                })
                continue
            
            # Extract markdown from PDF
            print(f"Extracting text from {processed_path}...")
            markdown_text = pymupdf4llm.to_markdown(processed_path)
            
            if not markdown_text.strip():
                results.append({
                    "file_path": file_path,
                    "markdown_text": "",
                    "estimated_tokens": "0",
                    "extraction_status": "NO_CONTENT",
                    "error_message": "No text content extracted from PDF"
                })
            else:
                estimated_tokens = estimate_tokens(markdown_text)
                results.append({
                    "file_path": file_path,
                    "markdown_text": markdown_text,
                    "estimated_tokens": str(estimated_tokens),
                    "extraction_status": "SUCCESS",
                    "error_message": None
                })
                
        except Exception as e:
            results.append({
                "file_path": file_path,
                "markdown_text": "",
                "estimated_tokens": "0",
                "extraction_status": "ERROR",
                "error_message": str(e)
            })
    
    return pd.DataFrame(results)


# =============================================================================
# UDF 2: Markdown Analysis with LLM
# =============================================================================

def create_markdown_analysis_udf(
    databricks_endpoint: str = "databricks-meta-llama-3-3-70b-instruct",
    prompt_template: Optional[str] = None,
    max_tokens: int = 120000
):
    """
    Factory function to create a markdown analysis UDF with specific model configuration.
    """
    
    @pandas_udf(
        returnType=StructType([
            StructField("markdown_text", StringType(), True),
            StructField("analysis_result", StringType(), True),
            StructField("model_used", StringType(), True),
            StructField("was_chunked", StringType(), True),
            StructField("processing_status", StringType(), True),
            StructField("error_message", StringType(), True)
        ])
    )
    def analyze_markdown_udf(markdown_texts: pd.Series) -> pd.DataFrame:
        """
        UDF to analyze markdown text using LLM.
        Handles chunking for large documents automatically.
        """
        results = []
        
        # Initialize LLM once per batch
        try:
            llm = ChatDatabricks(
                endpoint=databricks_endpoint,
                max_tokens=4000,
                temperature=0.1
            )
        except Exception as e:
            # If LLM initialization fails, return errors for all inputs
            for markdown_text in markdown_texts:
                results.append({
                    "markdown_text": markdown_text,
                    "analysis_result": "",
                    "model_used": databricks_endpoint,
                    "was_chunked": "ERROR",
                    "processing_status": "LLM_INIT_FAILED", 
                    "error_message": f"Failed to initialize LLM: {str(e)}"
                })
            return pd.DataFrame(results)
        
        # Default prompt template
        if prompt_template is None:
            default_prompt = """
Please analyze and summarize the following document content:
{content}
Provide a succint analysis including:
1. Key Topics
2. Overall Summary
3. Why someone would need to check this document
Analysis:"""
        else:
            default_prompt = prompt_template
        
        # Process each markdown text
        for markdown_text in markdown_texts:
            try:
                # Skip empty texts
                if not markdown_text or not markdown_text.strip():
                    results.append({
                        "markdown_text": markdown_text,
                        "analysis_result": "No content to analyze",
                        "model_used": databricks_endpoint,
                        "was_chunked": "NO",
                        "processing_status": "SKIPPED",
                        "error_message": "Empty or null markdown text"
                    })
                    continue
                
                # Check document size and process accordingly
                estimated_tokens = estimate_tokens(markdown_text)
                prompt_overhead = estimate_tokens(default_prompt.replace("{content}", ""))
                total_estimated = estimated_tokens + prompt_overhead
                
                print(f"Processing markdown: {estimated_tokens:,} tokens (with prompt: {total_estimated:,})")
                
                if total_estimated > max_tokens:
                    print(f"Document too large ({total_estimated:,} tokens), using chunking strategy...")
                    analysis_result = process_large_document(markdown_text, llm, default_prompt, max_tokens)
                    was_chunked = "YES"
                else:
                    print("Processing document in single request...")
                    prompt = default_prompt.format(content=markdown_text)
                    result = llm.invoke(prompt)
                    analysis_result = result.content
                    was_chunked = "NO"
                
                results.append({
                    "markdown_text": markdown_text,
                    "analysis_result": analysis_result,
                    "model_used": databricks_endpoint,
                    "was_chunked": was_chunked,
                    "processing_status": "SUCCESS",
                    "error_message": None
                })
                
            except Exception as e:
                results.append({
                    "markdown_text": markdown_text,
                    "analysis_result": "",
                    "model_used": databricks_endpoint,
                    "was_chunked": "ERROR",
                    "processing_status": "ANALYSIS_FAILED",
                    "error_message": str(e)
                })
        
        return pd.DataFrame(results)
    
    return analyze_markdown_udf

# COMMAND ----------

result_df = document_table .select(
        col("*"),
        extract_pdf_to_markdown_udf(col("volume_path")).alias("extraction_result")
    ).select(
        col("*"),
        col("extraction_result.file_path").alias("extracted_file_path"),
        col("extraction_result.markdown_text"),
        col("extraction_result.estimated_tokens"),
        col("extraction_result.extraction_status"),
        col("extraction_result.error_message").alias("extraction_error")
    )
  
# COMMAND ----------

display(result_df)

# COMMAND ----------