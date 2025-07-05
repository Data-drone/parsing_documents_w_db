# Databricks notebook source
# MAGIC %md
# MAGIC # Optimized PDF Page Splitting Pipeline
# MAGIC 
# MAGIC This notebook efficiently processes PDFs from a Delta table and splits them into individual pages as PNG images.
# MAGIC 
# MAGIC ## Key Optimizations Made:
# MAGIC - **Removed artificial groupBy**: Eliminated unnecessary shuffling from `groupby("batch_id").applyInPandas()`
# MAGIC - **Direct Delta table reading**: Read from structured Delta table instead of binary file format
# MAGIC - **Simplified pandas UDF**: Use `pandas_udf` directly instead of complex grouping
# MAGIC - **Reduced memory usage**: Removed 4x memory overhead from high-resolution image generation
# MAGIC - **Eliminated external dependencies**: Self-contained processing functions
# MAGIC 
# MAGIC ## Input Schema:
# MAGIC ```
# MAGIC root
# MAGIC |-- file_name: string
# MAGIC |-- volume_path: string  
# MAGIC |-- file_extension: string
# MAGIC |-- file_size_bytes: long
# MAGIC |-- modification_time: timestamp
# MAGIC |-- directory: string
# MAGIC |-- binary_content: binary
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration and Setup

# COMMAND ----------

# MAGIC %pip install pymupdf
# MAGIC %restart_python

# COMMAND ----------

import os
import pandas as pd
import fitz  # PyMuPDF
import logging
import time
from datetime import datetime
import uuid

from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, col

# Configuration
CATALOG = "brian_gen_ai"
INPUT_SCHEMA = "parsing_test" 
SOURCE_TABLE = "document_store_blob"           # Your Delta table with PDFs
OUTPUT_TABLE = "document_page_docs" # Output table for pages

# Processing settings
USE_HIGH_RESOLUTION = False          # Set to True for 2x resolution (uses 4x memory)
MAX_FILE_SIZE_MB = 50               # Skip files larger than this

print(f"Source: {CATALOG}.{INPUT_SCHEMA}.{SOURCE_TABLE}")
print(f"Output: {CATALOG}.{INPUT_SCHEMA}.{OUTPUT_TABLE}")
print(f"High resolution: {USE_HIGH_RESOLUTION}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions
# MAGIC 
# MAGIC Simple, self-contained functions for PDF processing without external dependencies.

# COMMAND ----------

def generate_doc_id(filename: str) -> str:
    """Generate a unique document ID from filename and timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{filename}_{timestamp}_{str(uuid.uuid4())[:8]}"

def create_page_metadata(filename: str, filepath: str, file_size: int, total_pages: int, pdf_metadata: dict) -> dict:
    """Create simplified metadata structure"""
    return {
        "source_filename": filename,
        "source_filepath": filepath, 
        "file_size_bytes": file_size,
        "total_pages": total_pages,
        "processing_timestamp": datetime.now().isoformat(),
        "pdf_title": pdf_metadata.get("title", ""),
        "pdf_author": pdf_metadata.get("author", ""),
        "pdf_subject": pdf_metadata.get("subject", "")
    }

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## PDF Processing Function
# MAGIC 
# MAGIC This function efficiently processes PDF binary data and extracts pages as PNG images.
# MAGIC Uses `mapInPandas` to handle variable output rows (1 PDF → multiple pages).
# MAGIC 
# MAGIC **Performance Notes:**
# MAGIC - Processes PDFs in parallel across partitions
# MAGIC - Uses standard resolution by default (set `USE_HIGH_RESOLUTION=True` for 2x resolution)
# MAGIC - Includes error handling for corrupted/problematic PDFs
# MAGIC - Returns structured data with consistent schema

# COMMAND ----------

# Define output schema for the processing function
pdf_pages_schema = StructType([
    StructField("doc_id", StringType(), True),
    StructField("source_filename", StringType(), True),
    StructField("page_number", IntegerType(), True),
    StructField("page_image_png", BinaryType(), True),
    StructField("total_pages", IntegerType(), True),
    StructField("file_size_bytes", LongType(), True),
    StructField("processing_timestamp", StringType(), True),
    StructField("metadata_json", StringType(), True)
])

def extract_pdf_pages_batch(pdf_batch: pd.DataFrame) -> pd.DataFrame:
    """
    Extract pages from PDF binary data as PNG images.
    
    Input DataFrame columns: file_name, volume_path, binary_content, file_size_bytes
    Returns: DataFrame with one row per page (variable number of output rows)
    """
    results = []
    
    for idx, row in pdf_batch.iterrows():
        filename = row['file_name']
        filepath = row['volume_path'] 
        binary_content = row['binary_content']
        file_size = row['file_size_bytes']
        
        try:
            # Skip if no binary content
            if binary_content is None or len(binary_content) == 0:
                logger.warning(f"No binary content for {filename}")
                continue
                
            # Skip very large files  
            if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                logger.warning(f"Skipping large file {filename}: {file_size/1024/1024:.1f}MB")
                continue
            
            # Open PDF from binary data
            doc = fitz.open(stream=binary_content, filetype="pdf")
            
            if len(doc) == 0:
                logger.warning(f"PDF has no pages: {filename}")
                doc.close()
                continue
            
            # Generate document ID and metadata
            doc_id = generate_doc_id(filename)
            pdf_metadata = doc.metadata if doc.metadata else {}
            total_pages = len(doc)
            
            metadata = create_page_metadata(
                filename, filepath, file_size, total_pages, pdf_metadata
            )
            
            # Process each page
            for page_num in range(total_pages):
                try:
                    page = doc[page_num]
                    
                    # Create page image
                    if USE_HIGH_RESOLUTION:
                        # High resolution (2x) - uses 4x memory
                        matrix = fitz.Matrix(2, 2)
                        pix = page.get_pixmap(matrix=matrix, alpha=False)
                    else:
                        # Standard resolution
                        pix = page.get_pixmap(alpha=False)
                    
                    # Convert to PNG bytes
                    img_bytes = pix.pil_tobytes(format="PNG")
                    
                    results.append({
                        "doc_id": doc_id,
                        "source_filename": filename,
                        "page_number": page_num + 1,  # 1-indexed
                        "page_image_png": img_bytes,
                        "total_pages": total_pages,
                        "file_size_bytes": file_size,
                        "processing_timestamp": datetime.now().isoformat(),
                        "metadata_json": str(metadata)
                    })
                    
                except Exception as page_error:
                    logger.error(f"Error processing page {page_num + 1} of {filename}: {str(page_error)}")
                    continue
            
            doc.close()
            logger.info(f"Successfully processed {filename}: {total_pages} pages")
            
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {str(e)}")
            continue
    
    # Return results as DataFrame
    if not results:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=[field.name for field in pdf_pages_schema.fields])
    
    return pd.DataFrame(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Loading and Validation
# MAGIC 
# MAGIC Load PDF files from the Delta table and validate the data before processing.

# COMMAND ----------

# Load source data
print("=== LOADING SOURCE DATA ===")
source_df = spark.table(f"{CATALOG}.{INPUT_SCHEMA}.{SOURCE_TABLE}")

# Basic validation
total_files = source_df.count()
pdf_files = source_df.filter(col("file_extension") == ".pdf")
pdf_count = pdf_files.count()

print(f"Total files in source table: {total_files:,}")
print(f"PDF files: {pdf_count:,}")

if pdf_count == 0:
    raise ValueError("No PDF files found in source table")

# COMMAND ----------

# Data quality checks
print("=== DATA QUALITY ANALYSIS ===")

# Check for missing binary content
missing_content = pdf_files.filter(col("binary_content").isNull()).count()
print(f"PDFs missing binary content: {missing_content}")

# Size distribution
size_stats = pdf_files.select(
    F.min("file_size_bytes").alias("min_size"),
    F.max("file_size_bytes").alias("max_size"), 
    F.avg("file_size_bytes").alias("avg_size"),
    F.count("*").alias("total_count")
).collect()[0]

print(f"File size stats:")
print(f"  Min: {size_stats['min_size']:,} bytes ({size_stats['min_size']/1024/1024:.1f} MB)")
print(f"  Max: {size_stats['max_size']:,} bytes ({size_stats['max_size']/1024/1024:.1f} MB)")
print(f"  Avg: {size_stats['avg_size']:,.0f} bytes ({size_stats['avg_size']/1024/1024:.1f} MB)")

# Filter to files we'll actually process
processable_files = pdf_files.filter(
    (col("binary_content").isNotNull()) &
    (col("file_size_bytes") <= MAX_FILE_SIZE_MB * 1024 * 1024)
)

processable_count = processable_files.count()
print(f"Files eligible for processing: {processable_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## PDF Processing Execution
# MAGIC 
# MAGIC Process all eligible PDF files to extract pages using `mapInPandas()`.
# MAGIC 
# MAGIC **Why mapInPandas():**
# MAGIC - Allows variable output rows (1 PDF → multiple pages)
# MAGIC - Processes in parallel across partitions  
# MAGIC - More efficient than groupBy().applyInPandas() for this use case
# MAGIC - No artificial grouping needed

# COMMAND ----------

print("=== STARTING PDF PAGE EXTRACTION ===")
start_time = time.time()

# Process PDFs using mapInPandas (allows variable output rows)
# This is the correct approach when 1 input row produces multiple output rows
pages_df = processable_files.select(
    col("file_name"),
    col("volume_path"),
    col("binary_content"),
    col("file_size_bytes")
).repartition(8).mapInPandas(  # Repartition for better parallelism
    extract_pdf_pages_batch,
    schema=pdf_pages_schema
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Results
# MAGIC 
# MAGIC Save the extracted pages to a Delta table partitioned by document ID for efficient querying.

# COMMAND ----------

print("=== SAVING RESULTS ===")

# Save to Delta table with partitioning for better query performance
(pages_df
 .write
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .partitionBy("doc_id")
 .saveAsTable(f"{CATALOG}.{INPUT_SCHEMA}.{OUTPUT_TABLE}")
)

end_time = time.time()
processing_time = end_time - start_time

# COMMAND ----------

# MAGIC %md
# MAGIC ## Processing Summary
# MAGIC 
# MAGIC Display summary statistics and performance metrics for the processing job.

# COMMAND ----------

print("=== PROCESSING SUMMARY ===")

# Get final counts
final_pages_count = spark.table(f"{CATALOG}.{INPUT_SCHEMA}.{OUTPUT_TABLE}").count()
final_docs_count = spark.table(f"{CATALOG}.{INPUT_SCHEMA}.{OUTPUT_TABLE}").select("doc_id").distinct().count()

print(f"Processing completed successfully!")
print(f"")
print(f"Input Statistics:")
print(f"  - PDF files processed: {processable_count:,}")
print(f"  - Total processing time: {processing_time:.1f} seconds")
print(f"")
print(f"Output Statistics:")
print(f"  - Documents created: {final_docs_count:,}")
print(f"  - Total pages extracted: {final_pages_count:,}")
print(f"  - Average pages per document: {final_pages_count/final_docs_count:.1f}")
print(f"")
print(f"Performance Metrics:")
print(f"  - Files per second: {processable_count/processing_time:.2f}")
print(f"  - Pages per second: {final_pages_count/processing_time:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Results
# MAGIC 
# MAGIC Display a sample of the extracted pages to verify the processing worked correctly.

# COMMAND ----------

# Show sample results
sample_results = spark.sql(f"""
SELECT 
    source_filename,
    doc_id,
    page_number,
    total_pages,
    file_size_bytes,
    LENGTH(page_image_png) as image_size_bytes,
    processing_timestamp
FROM {CATALOG}.{INPUT_SCHEMA}.{OUTPUT_TABLE}
ORDER BY source_filename, page_number
LIMIT 10
""")

display(sample_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation Queries
# MAGIC 
# MAGIC Run some validation queries to ensure data quality and completeness.

# COMMAND ----------

print("=== DATA VALIDATION ===")

# Check for any processing errors (empty results)
validation_df = spark.sql(f"""
SELECT 
    COUNT(DISTINCT doc_id) as unique_documents,
    COUNT(*) as total_pages,
    AVG(total_pages) as avg_pages_per_doc,
    MIN(page_number) as min_page_num,
    MAX(page_number) as max_page_num,
    AVG(LENGTH(page_image_png)) as avg_image_size_bytes
FROM {CATALOG}.{INPUT_SCHEMA}.{OUTPUT_TABLE}
""")

display(validation_df)

# COMMAND ----------

# Check for any documents with missing pages (gaps in page numbering)
page_gaps = spark.sql(f"""
WITH page_sequences AS (
    SELECT 
        doc_id,
        source_filename,
        total_pages,
        COUNT(*) as actual_pages,
        MIN(page_number) as min_page,
        MAX(page_number) as max_page
    FROM {CATALOG}.{INPUT_SCHEMA}.{OUTPUT_TABLE}
    GROUP BY doc_id, source_filename, total_pages
)
SELECT 
    source_filename,
    total_pages,
    actual_pages,
    min_page,
    max_page,
    CASE 
        WHEN actual_pages != total_pages THEN 'PAGE_COUNT_MISMATCH'
        WHEN min_page != 1 THEN 'MISSING_FIRST_PAGES'
        WHEN max_page != total_pages THEN 'MISSING_LAST_PAGES'
        ELSE 'OK'
    END as status
FROM page_sequences
WHERE actual_pages != total_pages OR min_page != 1 OR max_page != total_pages
ORDER BY source_filename
""")

print("Documents with page issues:")
display(page_gaps)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC 
# MAGIC ## Processing Complete! 
# MAGIC 
# MAGIC ### Key Improvements Made:
# MAGIC 1. **Used mapInPandas()**: Correctly handles variable output rows (1 PDF → multiple pages)
# MAGIC 2. **Direct Delta table source**: Read directly from structured Delta table instead of binary file format
# MAGIC 3. **Memory optimization**: Made high-resolution processing optional to reduce memory usage
# MAGIC 4. **Better error handling**: Individual PDF failures don't stop the entire process
# MAGIC 5. **Self-contained**: Removed external module dependencies
# MAGIC 6. **Performance monitoring**: Added detailed timing and throughput metrics
# MAGIC 7. **Proper partitioning**: Added repartitioning for better parallelism
# MAGIC 
# MAGIC ### Output Schema:
# MAGIC ```
# MAGIC |-- doc_id: string
# MAGIC |-- source_filename: string  
# MAGIC |-- page_number: integer
# MAGIC |-- page_image_png: binary
# MAGIC |-- total_pages: integer
# MAGIC |-- file_size_bytes: long
# MAGIC |-- processing_timestamp: string
# MAGIC |-- metadata_json: string
# MAGIC ```