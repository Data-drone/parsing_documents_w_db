# Databricks notebook source
# MAGIC %md
# MAGIC # Document Markdown Extraction with PyMuPDF4LLM
# MAGIC 
# MAGIC ## Overview
# MAGIC This notebook demonstrates PDF markdown extraction using a **two-tier approach**:
# MAGIC 
# MAGIC - **Simple Examples**: Use `pymupdf4llm` for easy single-file processing and learning
# MAGIC - **Production UDF**: Use PyMuPDF's `fitz` API combined with `pymupdf4llm` for robust binary stream processing in distributed environments
# MAGIC 
# MAGIC This progression helps you understand the concepts with simple tools, then scales to production-ready binary stream processing.
# MAGIC 
# MAGIC ## Why Two Different Approaches?
# MAGIC 
# MAGIC ### 🎓 **pymupdf4llm (Learning & Simple Cases)**
# MAGIC - **Pros**: Simple API, great for learning and single files
# MAGIC - **Cons**: Can have issues with binary streams in distributed processing
# MAGIC - **Use for**: Examples, tutorials, single-file processing
# MAGIC 
# MAGIC ### 🏭 **PyMuPDF + pymupdf4llm (Production & UDFs)**  
# MAGIC - **Pros**: Reliable binary stream processing, full control, best markdown extraction quality
# MAGIC - **Cons**: Slightly more setup than pure pymupdf4llm
# MAGIC - **Use for**: Pandas UDFs, production pipelines, binary data processing
# MAGIC 
# MAGIC ## Benefits of This Approach
# MAGIC - 🚀 **Optimized Processing**: Leverage Spark's pandas UDF for vectorized operations with better performance than RDD map
# MAGIC - 📝 **High-Quality Markdown**: Combines PyMuPDF's robust binary handling with pymupdf4llm's superior markdown extraction
# MAGIC - 🔄 **Batch Processing**: Process hundreds or thousands of documents efficiently
# MAGIC - 💾 **Persistent Results**: Store extracted content in Delta Lake for reuse
# MAGIC - 🛡️ **Error Handling**: Robust processing with detailed error tracking

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Setup and Configuration
# MAGIC 
# MAGIC Install required libraries and set up workspace configuration.

# COMMAND ----------

# Install both pymupdf4llm (simple API) and PyMuPDF (robust binary processing)
%pip install pymupdf4llm PyMuPDF python-dotenv
%restart_python

# COMMAND ----------

# Load environment variables from .env (if present) **before** we read them via os.getenv
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # returns True if a .env is found and parsed

import os
import io
import traceback
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
import pymupdf4llm  # Simple API for examples
import pymupdf  # PyMuPDF for robust binary stream processing in UDFs (same as fitz)
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType, BooleanType

# -----------------------------------------------------------------------------
# 1️⃣  Runtime configuration via widgets
# -----------------------------------------------------------------------------
# Pull defaults from environment variables (populated via `.env` or workspace)
import os

# Create Databricks widgets so users can override at run-time
dbutils.widgets.text("catalog_name", os.getenv("CATALOG_NAME", "my_catalog"), "Catalog Name")
dbutils.widgets.text("schema_name",  os.getenv("SCHEMA_NAME",  "tutorials"),   "Schema Name")
dbutils.widgets.text("volume_name",  os.getenv("VOLUME_NAME",  "sample_docs"), "Volume Name")

# Read values back from the widgets
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA  = dbutils.widgets.get("schema_name")
VOLUME  = dbutils.widgets.get("volume_name")

# Construct commonly-used paths / table names
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.document_store"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.document_markdown"
VOLUME_PATH  = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

print(f"Source table : {SOURCE_TABLE}")
print(f"Output table : {OUTPUT_TABLE}")
print(f"Volume path  : {VOLUME_PATH}")
print("Note: This will process all PDF documents in the document store")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 🔍 Quick data check
# MAGIC 
# MAGIC Below we display the first file we can find in the configured `VOLUME_PATH`.
# MAGIC This helps ensure your widget values (and environment defaults) point to a
# MAGIC location that actually contains data before kicking off the heavy parsing job.

# COMMAND ----------

# List files in the configured volume path (non-recursive)
try:
    files = dbutils.fs.ls(VOLUME_PATH)
    if files:
        first_path = files[0].path
        # Strip the `dbfs:/` prefix so we get a direct FUSE path like /Volumes/...
        first_path = first_path.replace("dbfs:/", "/")
        print(f"First file in volume: {first_path}")
    else:
        print(f"⚠️  No files found in {VOLUME_PATH}. Please verify your volume contains PDFs.")
except Exception as e:
    print(f"Error accessing {VOLUME_PATH}: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Convert a Single PDF to Markdown
# MAGIC 
# MAGIC Now that we have a sample file (`first_path`), let's walk through how to
# MAGIC use **PyMuPDF4LLM** to extract its contents to Markdown.  This is useful
# MAGIC if you just want the text from a single document without running the full
# MAGIC Spark pipeline.

# COMMAND ----------

# Convert the DBFS URI (dbfs:/...) to a local FUSE path (/dbfs/...) so that
# pymupdf4llm can read the file like any other local file.

if 'first_path' not in globals():
    raise ValueError("`first_path` was not defined – please run the previous cell.")

# `first_path` is already a direct `/Volumes/...` path that can be read
local_pdf_path = first_path
print(f"Reading: {local_pdf_path}")

# Extract to markdown using pymupdf4llm (simple API)
markdown_content = pymupdf4llm.to_markdown(local_pdf_path)

# Preview the first ~1 000 characters (feel free to adjust)
preview_len = 1000
print("\nMarkdown preview (first", preview_len, "chars):\n")
print("-" * 80)
print(markdown_content[:preview_len])
if len(markdown_content) > preview_len:
    print("\n… (truncated)")
print("-" * 80)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2️⃣ Visual Comparison – Page vs. Markdown
# MAGIC 
# MAGIC For a quick sanity-check we can render the first page of the PDF and show it
# MAGIC side-by-side with the extracted markdown from that same page.
#
# COMMAND ----------
# Set comparison parameters
compare_path = local_pdf_path  # 👉 change this to any PDF path you like
page_index = 0                 # 👉 zero-based page number to visualise

print(f"Comparing {compare_path}, page {page_index}")

# COMMAND ----------
import base64

# Open the chosen document & page
doc = pymupdf.open(compare_path)
page = doc.load_page(page_index)
pix = page.get_pixmap(dpi=150)

# Encode page image to base64 for HTML embedding
img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

# Extract plaintext from that single page
page_text = page.get_text("text")
page_md = f"""
{page_text}
"""

# Build side-by-side HTML
page_num_disp = page_index + 1
html = f"""
<div style='display:flex; gap:16px;'>
  <div style='flex:1;'>
    <h3 style='margin-top:0;'>PDF Page {page_num_disp}</h3>
    <img src='data:image/png;base64,{img_b64}' style='max-width:100%; border:1px solid #ccc;' />
  </div>
  <div style='flex:1; white-space:pre-wrap; font-family:monospace; overflow:auto;'>
    <h3 style='margin-top:0;'>Extracted Markdown</h3>
    {page_md.replace('<','&lt;').replace('>','&gt;')}
  </div>
</div>
"""

displayHTML(html)

# Clean up
doc.close()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Production-Ready Distributed Processing with Pandas UDF
# MAGIC 
# MAGIC Now we'll switch to combining PyMuPDF's binary stream handling with pymupdf4llm's superior markdown extraction.
# MAGIC **Why combine pymupdf with pymupdf4llm for the UDF?**
# MAGIC 
# MAGIC ### 🏗️ **Production Requirements for Binary Stream Processing:**
# MAGIC - **Reliable binary handling**: `pymupdf.open(stream=binary_data)` works directly with bytes
# MAGIC - **Superior markdown quality**: `pymupdf4llm.to_markdown()` provides better formatted output
# MAGIC - **No file system dependencies**: Processes data purely in memory
# MAGIC - **Better error handling**: More predictable behavior with malformed PDFs
# MAGIC - **Distributed processing**: Designed to work reliably across Spark partitions
# MAGIC 
# MAGIC ### 🎯 **Benefits of This Combined UDF Approach:**
# MAGIC - 🚀 **Pandas UDF Performance**: Vectorized processing with better performance than regular UDFs
# MAGIC - 🛡️ **Robust Binary Processing**: Direct stream processing without temporary files
# MAGIC - 📊 **High-Quality Markdown**: Best formatting and structure preservation
# MAGIC - 🔧 **Production Ready**: Handles edge cases and distributed processing requirements

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, col, lit, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType, BooleanType
from datetime import datetime
import pandas as pd
import os

# Define the output schema to match the full parsing pipeline
markdown_extraction_schema = StructType([
    StructField("extraction_success", BooleanType(), True),
    StructField("markdown_content", StringType(), True),
    StructField("error_message", StringType(), True),
    StructField("processing_time_seconds", StringType(), True),
    StructField("character_count", LongType(), True),
    StructField("page_count", LongType(), True),
    StructField("processing_timestamp", TimestampType(), True)
])

def extract_markdown_from_binary(binary_data, file_name, file_extension):
    """
    Helper function to extract markdown from PDF binary data using PyMuPDF + pymupdf4llm.
    
    This function combines the reliability of PyMuPDF's binary stream processing 
    with the superior markdown quality of pymupdf4llm.
    
    Args:
        binary_data: The binary content of the PDF file
        file_name: Name of the file for error reporting
        file_extension: File extension to check if it's a PDF
    
    Returns:
        Dictionary with extraction results including detailed debug info
    """
    start_time = datetime.now()
    debug_info = []
    
    try:
        debug_info.append(f"Step 1: Starting processing for {file_name}")
        
        # Step 1: Check file extension
        if file_extension.lower() != '.pdf':
            debug_info.append(f"Step 1 FAILED: Unsupported file type {file_extension}")
            return {
                "extraction_success": False,
                "markdown_content": None,
                "error_message": f"Unsupported file type: {file_extension}. Debug: {'; '.join(debug_info)}",
                "processing_time_seconds": "0.0",
                "character_count": 0,
                "page_count": 0
            }
        
        # Step 2: Check binary data
        if binary_data is None:
            debug_info.append(f"Step 2 FAILED: No binary data provided")
            return {
                "extraction_success": False,
                "markdown_content": None,
                "error_message": f"No binary data provided. Debug: {'; '.join(debug_info)}",
                "processing_time_seconds": "0.0",
                "character_count": 0,
                "page_count": 0
            }
        
        debug_info.append(f"Step 2: Binary data type={type(binary_data).__name__}, size={len(binary_data)} bytes")
        
        # Step 3: Create BytesIO stream and open PDF
        try:
            debug_info.append(f"Step 3: Creating BytesIO stream and opening PDF")
            pdf_stream = io.BytesIO(binary_data)
            doc = pymupdf.open(stream=pdf_stream, filetype="pdf")
            debug_info.append(f"Step 3 SUCCESS: Opened PDF, page_count={doc.page_count}")
        except Exception as pdf_open_error:
            debug_info.append(f"Step 3 FAILED: {type(pdf_open_error).__name__}: {str(pdf_open_error)}")
            full_traceback = traceback.format_exc()
            error_msg = f"Failed to open PDF stream. Debug: {'; '.join(debug_info)}. Traceback: {full_traceback}"
            return {
                "extraction_success": False,
                "markdown_content": None,
                "error_message": error_msg,
                "processing_time_seconds": str(round((datetime.now() - start_time).total_seconds(), 3)),
                "character_count": 0,
                "page_count": 0
            }
        
        # Step 4: Extract markdown using pymupdf4llm
        try:
            debug_info.append(f"Step 4: Extracting markdown using pymupdf4llm from {doc.page_count} pages")
            
            # Use pymupdf4llm for high-quality markdown extraction
            markdown_content = pymupdf4llm.to_markdown(doc)
            
            actual_page_count = doc.page_count
            debug_info.append(f"Step 4 SUCCESS: Extracted {len(markdown_content)} characters from {actual_page_count} pages")
            
        except Exception as extraction_error:
            doc.close()  # Clean up before returning
            debug_info.append(f"Step 4 FAILED: {type(extraction_error).__name__}: {str(extraction_error)}")
            full_traceback = traceback.format_exc()
            error_msg = f"Failed during pymupdf4llm markdown extraction. Debug: {'; '.join(debug_info)}. Traceback: {full_traceback}"
            return {
                "extraction_success": False,
                "markdown_content": None,
                "error_message": error_msg,
                "processing_time_seconds": str(round((datetime.now() - start_time).total_seconds(), 3)),
                "character_count": 0,
                "page_count": 0
            }
        
        # Step 5: Clean up and calculate metrics
        try:
            doc.close()
            debug_info.append(f"Step 5: PDF closed successfully")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            character_count = len(markdown_content) if markdown_content else 0
            
            debug_info.append(f"Step 5 SUCCESS: {character_count} chars, {actual_page_count} pages, {processing_time:.3f}s")
            
            # Create success message with debug info for troubleshooting
            success_debug = f"Processing completed successfully using pymupdf4llm. Debug: {'; '.join(debug_info[-3:])}"  # Last 3 debug messages
            
            return {
                "extraction_success": True,
                "markdown_content": markdown_content,
                "error_message": success_debug,  # Include debug info even for success
                "processing_time_seconds": str(round(processing_time, 3)),
                "character_count": character_count,
                "page_count": actual_page_count
            }
            
        except Exception as cleanup_error:
            debug_info.append(f"Step 5 FAILED: {type(cleanup_error).__name__}: {str(cleanup_error)}")
            full_traceback = traceback.format_exc()
            error_msg = f"Error during cleanup. Debug: {'; '.join(debug_info)}. Traceback: {full_traceback}"
            return {
                "extraction_success": False,
                "markdown_content": None,
                "error_message": error_msg,
                "processing_time_seconds": str(round((datetime.now() - start_time).total_seconds(), 3)),
                "character_count": 0,
                "page_count": 0
            }
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        debug_info.append(f"UNEXPECTED ERROR: {type(e).__name__}: {str(e)}")
        
        # Get full traceback for debugging
        full_traceback = traceback.format_exc()
        
        error_msg = f"Unexpected error in {file_name}. Debug: {'; '.join(debug_info)}. Traceback: {full_traceback}"
        
        return {
            "extraction_success": False,
            "markdown_content": None,
            "error_message": error_msg,
            "processing_time_seconds": str(round(processing_time, 3)),
            "character_count": 0,
            "page_count": 0
        }

@pandas_udf(returnType=markdown_extraction_schema)
def extract_markdown_udf(binary_data_series: pd.Series, file_name_series: pd.Series, file_extension_series: pd.Series) -> pd.DataFrame:
    """
    Production-ready pandas UDF using PyMuPDF + pymupdf4llm for binary stream processing.
    
    **Why combine pymupdf with pymupdf4llm?**
    - pymupdf.open(stream=binary_data) handles binary streams reliably
    - pymupdf4llm.to_markdown() provides superior markdown formatting
    - No file system dependencies - pure in-memory processing
    - Better error handling for malformed PDFs in distributed environments
    - More predictable behavior across different Spark executors
    
    Args:
        binary_data_series: Pandas Series containing binary PDF data
        file_name_series: Pandas Series containing file names
        file_extension_series: Pandas Series containing file extensions
    
    Returns:
        Pandas DataFrame with extraction results matching the production pipeline schema
    """
    results = []
    processing_timestamp = datetime.now()
    
    for i in range(len(binary_data_series)):
        binary_data = binary_data_series.iloc[i]
        file_name = file_name_series.iloc[i]
        file_extension = file_extension_series.iloc[i]
        
        print(f"Processing: {file_name}")
        
        # Extract markdown using PyMuPDF + pymupdf4llm binary stream processing
        extraction_result = extract_markdown_from_binary(binary_data, file_name, file_extension)
        
        # Create result record matching the schema
        result = {
            "extraction_success": extraction_result["extraction_success"],
            "markdown_content": extraction_result["markdown_content"],
            "error_message": extraction_result["error_message"],
            "processing_time_seconds": extraction_result["processing_time_seconds"],
            "character_count": extraction_result["character_count"],
            "page_count": extraction_result["page_count"],
            "processing_timestamp": processing_timestamp
        }
        
        results.append(result)
    
    return pd.DataFrame(results)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Load Document Store and Process Documents
# MAGIC 
# MAGIC Load the document store table created in the setup tutorial and process all PDF documents.

# COMMAND ----------

# Load the document store table
document_store_table = f"{CATALOG}.{SCHEMA}.document_store"
print(f"Loading documents from: {document_store_table}")

try:
    source_df = spark.table(document_store_table)
    total_docs = source_df.count()
    pdf_docs = source_df.filter(col("file_extension") == ".pdf").count()
    
    print(f"📊 Found {total_docs:,} total documents")
    print(f"📄 Found {pdf_docs:,} PDF documents to process")
    
    if pdf_docs == 0:
        print("⚠️  No PDF documents found. Please run the document store creation tutorial first.")
    
except Exception as e:
    print(f"❌ Error loading document store: {e}")
    print("Please run the '02_create_document_store' tutorial first to create the document store.")

# COMMAND ----------

# Process documents with the UDF if we have PDFs to process
if pdf_docs > 0:
    print(f"🚀 Processing {pdf_docs:,} PDF documents...")
    
    # Apply the pandas UDF and create the output table with consistent schema
    processed_df = source_df.select(
        # Original columns from document store
        col("file_name"),
        col("volume_path"), 
        col("file_extension"),
        col("file_size_bytes"),
        col("modification_time"),
        col("directory"),
        # Apply pandas UDF to extract markdown
        extract_markdown_udf(
            col("binary_content"), 
            col("file_name"), 
            col("file_extension")
        ).alias("extraction_results")
    ).select(
        # Flatten the structure to match the full pipeline schema
        col("file_name"),
        col("volume_path"),
        col("file_extension"),
        col("file_size_bytes"),
        col("modification_time"),
        col("directory"),
        col("extraction_results.extraction_success"),
        col("extraction_results.markdown_content"),
        col("extraction_results.error_message"),
        col("extraction_results.processing_time_seconds"),
        col("extraction_results.character_count"),
        col("extraction_results.page_count"),
        col("extraction_results.processing_timestamp")
    )
    
    # Save to output table with same name format as full pipeline
    output_table = f"{CATALOG}.{SCHEMA}.document_markdown_simple"
    processed_df.write.mode("overwrite").saveAsTable(output_table)
    
    print(f"✅ Results saved to: {output_table}")
    processed_df = spark.table(output_table)

    # Show processing summary
    success_count = processed_df.filter(col("extraction_success") == True).count()
    total_processed = processed_df.count()
    success_rate = (success_count / total_processed * 100) if total_processed > 0 else 0
    
    print(f"\n📊 Processing Summary:")
    print(f"  ✅ Successful extractions: {success_count:,}")
    print(f"  ❌ Failed extractions: {total_processed - success_count:,}")
    print(f"  📈 Success rate: {success_rate:.1f}%")

# COMMAND ----------

# Show sample results
print(f"\n📝 Sample results:")
display(processed_df.select(
        "file_name", 
        "extraction_success", 
        "character_count", 
        "page_count",
        "processing_time_seconds"
))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 📚 Summary: Two-Tier PDF Processing Strategy
# MAGIC 
# MAGIC This notebook demonstrated a **practical progression** from simple to production-ready PDF processing:
# MAGIC 
# MAGIC ### 🎓 **Learning Phase: pymupdf4llm**
# MAGIC ```python
# MAGIC # Simple, intuitive API for single files
# MAGIC markdown_content = pymupdf4llm.to_markdown(pdf_path)
# MAGIC ```
# MAGIC 
# MAGIC ### 🏭 **Production Phase: PyMuPDF + pymupdf4llm**
# MAGIC ```python
# MAGIC # Combine robust binary handling with superior markdown extraction
# MAGIC pdf_stream = io.BytesIO(binary_data)
# MAGIC doc = pymupdf.open(stream=pdf_stream, filetype="pdf")
# MAGIC markdown_content = pymupdf4llm.to_markdown(doc)
# MAGIC ```
# MAGIC 
# MAGIC ### 🎯 **When to Use Each:**
# MAGIC - **pymupdf4llm alone**: Tutorials, examples, single-file scripts, learning
# MAGIC - **pymupdf + pymupdf4llm**: Production UDFs, binary data processing, distributed systems
# MAGIC 
# MAGIC This approach gives you **the best of both worlds**: simple learning tools and robust production capabilities with superior markdown quality.

# COMMAND ----------
# MAGIC %md
# MAGIC ### 🔍 Debugging UDF Processing
# MAGIC 
# MAGIC Since print statements don't work within pandas UDFs (they run on worker nodes), 
# MAGIC we capture debug information in the error messages. Here's how to inspect any issues:

# COMMAND ----------

# Check for any processing errors or debug information
if 'processed_df' in locals():
    print("🔍 Checking for errors and debug information...")
    
    # Show failed extractions with debug info
    failed_df = processed_df.filter(col("extraction_success") == False)
    failed_count = failed_df.count()
    
    if failed_count > 0:
        print(f"❌ Found {failed_count} failed extractions:")
        failed_df.select("file_name", "error_message").show(truncate=False)
    else:
        print("✅ No failed extractions found!")
    
    # Show successful extractions with any debug warnings
    success_df = processed_df.filter(col("extraction_success") == True)
    success_with_issues = success_df.filter(col("error_message").contains("Page errors"))
    issues_count = success_with_issues.count()
    
    if issues_count > 0:
        print(f"⚠️ Found {issues_count} successful extractions with page-level issues:")
        success_with_issues.select("file_name", "error_message").show(truncate=False)
    
    # Show overall processing stats
    total_docs = processed_df.count()
    success_count = success_df.count()
    print(f"\n📊 Processing Summary:")
    print(f"   Total documents: {total_docs}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {failed_count}")
    print(f"   With page issues: {issues_count}")
else:
    print("⚠️ No processed_df found. Run the UDF processing cell first.")
