# Databricks notebook source
# MAGIC %md
# MAGIC # Document Markdown Extraction with PyMuPDF4LLM
# MAGIC 
# MAGIC ## Overview
# MAGIC This notebook processes the document store table created previously and extracts markdown content from PDF documents using PyMuPDF4LLM in a distributed manner with Spark.
# MAGIC 
# MAGIC ## Benefits of This Approach
# MAGIC - 🚀 **Optimized Processing**: Leverage Spark's pandas UDF for vectorized operations with better performance than RDD map
# MAGIC - 📝 **Markdown Output**: Convert PDFs to clean, structured markdown suitable for LLMs
# MAGIC - 🔄 **Batch Processing**: Process hundreds or thousands of documents efficiently
# MAGIC - 💾 **Persistent Results**: Store extracted content in Delta Lake for reuse
# MAGIC - 🛡️ **Error Handling**: Robust processing with detailed error tracking
# MAGIC 
# MAGIC ## What You'll Get
# MAGIC A new Delta Lake table containing:
# MAGIC - Original document metadata
# MAGIC - Extracted markdown content
# MAGIC - Processing status and error information
# MAGIC - Performance metrics

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Setup and Configuration
# MAGIC 
# MAGIC Install required libraries and set up workspace configuration.

# COMMAND ----------

# Install pymupdf4llm if not already available
%pip install pymupdf4llm

# COMMAND ----------

# Restart Python to ensure the library is properly loaded
dbutils.library.restartPython()

# COMMAND ----------

import os
import io
import traceback
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
import pymupdf4llm
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType, BooleanType

# Configuration: Define source and target tables
CATALOG = "brian_gen_ai"
SCHEMA = "parsing_test"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.document_store"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.document_markdown"

print(f"Source table: {SOURCE_TABLE}")
print(f"Output table: {OUTPUT_TABLE}")
print("Note: This will process all PDF documents in the document store")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Document Processing Functions
# MAGIC 
# MAGIC Define functions for extracting markdown content from PDF binary data using pandas UDF.
# MAGIC 
# MAGIC ### Key Features:
# MAGIC - **Pandas UDF optimization** for vectorized processing with better performance than RDD operations
# MAGIC - **Memory-efficient processing** using BytesIO streams
# MAGIC - **Comprehensive error handling** with detailed error messages
# MAGIC - **Performance tracking** to monitor processing times
# MAGIC - **Flexible extraction options** for different document types

# COMMAND ----------

def extract_markdown_from_pdf_bytes(binary_data: bytes, file_name: str, volume_path: str = None) -> Dict[str, Any]:
    """
    Extract markdown content from PDF binary data using pymupdf4llm.
    
    Args:
        binary_data: The binary content of the PDF file
        file_name: Name of the file for error reporting
        volume_path: Optional volume path to use as fallback
    
    Returns:
        Dictionary containing extraction results and metadata
    """
    start_time = datetime.now()
    
    try:
        if binary_data is None:
            return {
                "success": False,
                "markdown_content": None,
                "error_message": "No binary data provided",
                "processing_time_seconds": 0.0,
                "character_count": 0,
                "page_count": 0
            }
        
        markdown_content = None
        
        # Try Method 1: BytesIO with name attribute
        try:
            pdf_stream = io.BytesIO(binary_data)
            pdf_stream.name = file_name
            markdown_content = pymupdf4llm.to_markdown(pdf_stream)
        except Exception as stream_error:
            print(f"BytesIO method failed for {file_name}: {stream_error}")
            
            # Try Method 2: Use volume_path directly if available
            if volume_path and os.path.exists(volume_path):
                try:
                    markdown_content = pymupdf4llm.to_markdown(volume_path)
                    print(f"Successfully used volume_path method for {file_name}")
                except Exception as path_error:
                    print(f"Volume path method also failed for {file_name}: {path_error}")
                    raise stream_error  # Raise original error
            else:
                raise stream_error  # Raise original error if no fallback available
        
        # Calculate processing metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        character_count = len(markdown_content) if markdown_content else 0
        
        # Estimate page count based on content length (rough approximation)
        # Average ~2000 characters per page is a reasonable estimate
        estimated_pages = max(1, character_count // 2000) if character_count > 0 else 0
        
        return {
            "success": True,
            "markdown_content": markdown_content,
            "error_message": None,
            "processing_time_seconds": round(processing_time, 3),
            "character_count": character_count,
            "page_count": estimated_pages
        }
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        error_details = f"{type(e).__name__}: {str(e)}"
        
        print(f"Error processing {file_name}: {error_details}")
        
        return {
            "success": False,
            "markdown_content": None,
            "error_message": error_details,
            "processing_time_seconds": round(processing_time, 3),
            "character_count": 0,
            "page_count": 0
        }

# Define the return schema for the pandas UDF
extraction_schema = StructType([
    StructField("extraction_success", BooleanType(), True),
    StructField("markdown_content", StringType(), True),
    StructField("error_message", StringType(), True),
    StructField("processing_time_seconds", StringType(), True),
    StructField("character_count", LongType(), True),
    StructField("page_count", LongType(), True),
    StructField("processing_timestamp", TimestampType(), True)
])

@pandas_udf(returnType=extraction_schema)
def extract_markdown_udf(binary_data_series: pd.Series, file_name_series: pd.Series, file_extension_series: pd.Series, volume_path_series: pd.Series) -> pd.DataFrame:
    """
    Pandas UDF to extract markdown content from PDF binary data in a vectorized manner.
    
    This function processes multiple documents in parallel using pandas operations,
    with fallback to volume_path if BytesIO stream method fails.
    
    Args:
        binary_data_series: Pandas Series containing binary PDF data
        file_name_series: Pandas Series containing file names
        file_extension_series: Pandas Series containing file extensions
        volume_path_series: Pandas Series containing volume paths as fallback
    
    Returns:
        Pandas DataFrame with extraction results
    """
    results = []
    processing_timestamp = datetime.now()
    
    for i in range(len(binary_data_series)):
        binary_data = binary_data_series.iloc[i]
        file_name = file_name_series.iloc[i]
        file_extension = file_extension_series.iloc[i]
        volume_path = volume_path_series.iloc[i]
        
        print(f"Processing: {file_name}")
        
        # Only process PDF files
        if file_extension.lower() != '.pdf':
            result = {
                "extraction_success": False,
                "markdown_content": None,
                "error_message": f"Unsupported file type: {file_extension}",
                "processing_time_seconds": "0.0",
                "character_count": 0,
                "page_count": 0,
                "processing_timestamp": processing_timestamp
            }
        else:
            # Extract markdown from PDF with volume_path fallback
            extraction_result = extract_markdown_from_pdf_bytes(binary_data, file_name, volume_path)
            result = {
                "extraction_success": extraction_result["success"],
                "markdown_content": extraction_result["markdown_content"],
                "error_message": extraction_result["error_message"],
                "processing_time_seconds": str(extraction_result["processing_time_seconds"]),
                "character_count": extraction_result["character_count"],
                "page_count": extraction_result["page_count"],
                "processing_timestamp": processing_timestamp
            }
        
        results.append(result)
    
    return pd.DataFrame(results)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Load Source Data and Prepare for Processing
# MAGIC 
# MAGIC Read the document store table and prepare it for distributed markdown extraction.
# MAGIC 
# MAGIC ### Processing Strategy:
# MAGIC - **Filter for PDFs**: Focus on documents that can be processed
# MAGIC - **Batch sizing**: Optimize for memory usage and performance
# MAGIC - **Progress tracking**: Monitor processing across partitions

# COMMAND ----------

# Load the document store table
print("Loading document store table...")
source_df = spark.table(SOURCE_TABLE)

# Get overview of the data
total_documents = source_df.count()
pdf_documents = source_df.filter(F.col("file_extension") == ".pdf").count()
documents_with_content = source_df.filter(F.col("binary_content").isNotNull()).count()

print(f"📊 Document Store Overview:")
print(f"  Total documents: {total_documents:,}")
print(f"  PDF documents: {pdf_documents:,}")
print(f"  Documents with binary content: {documents_with_content:,}")

# Filter to only PDF documents with binary content
pdf_df = source_df.filter(
    (F.col("file_extension") == ".pdf") & 
    (F.col("binary_content").isNotNull())
)

pdf_count = pdf_df.count()
print(f"\n🎯 Ready for processing: {pdf_count:,} PDF documents")

if pdf_count == 0:
    print("⚠️  No PDF documents found with binary content. Please check your document store.")
    dbutils.notebook.exit("No PDFs to process")

# Display sample of documents to be processed
print("\nSample of documents to be processed:")
pdf_df.select("file_name", "file_size_bytes", "modification_time").show(5, truncate=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Distributed Markdown Extraction
# MAGIC 
# MAGIC Execute the markdown extraction using Spark's distributed processing capabilities.
# MAGIC 
# MAGIC ### Processing Details:
# MAGIC - **Parallel execution** across Spark cluster nodes
# MAGIC - **Memory management** with proper cleanup
# MAGIC - **Progress monitoring** with real-time updates
# MAGIC - **Error isolation** to prevent single document failures from stopping the job

# COMMAND ----------

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Distributed Markdown Extraction
# MAGIC 
# MAGIC Execute the markdown extraction using Spark's pandas UDF for optimized distributed processing.
# MAGIC 
# MAGIC ### Processing Details:
# MAGIC - **Pandas UDF optimization** for vectorized processing with better performance than RDD operations
# MAGIC - **Automatic parallelization** across Spark cluster nodes
# MAGIC - **Memory management** with proper cleanup and efficient data handling
# MAGIC - **Progress monitoring** with real-time updates
# MAGIC - **Error isolation** to prevent single document failures from stopping the job

# COMMAND ----------

print(f"🚀 Starting distributed markdown extraction for {pdf_count:,} documents...")
print("Using pandas UDF for optimized vectorized processing.")
print("This may take several minutes depending on document size and cluster capacity.")

start_time = datetime.now()

# Apply the pandas UDF to extract markdown content
# The UDF will be distributed across Spark partitions automatically
processed_df = pdf_df.select(
    "*",  # Keep all original columns
    extract_markdown_udf(
        col("binary_content"), 
        col("file_name"), 
        col("file_extension"),
        col("volume_path")  # Pass volume_path for fallback method
    ).alias("extraction_results")
).select(
    # Original columns
    "file_name",
    "volume_path", 
    "file_extension",
    "file_size_bytes",
    "modification_time",
    "directory",
    # Extracted columns from the UDF result
    col("extraction_results.extraction_success"),
    col("extraction_results.markdown_content"),
    col("extraction_results.error_message"),
    col("extraction_results.processing_time_seconds"),
    col("extraction_results.character_count"),
    col("extraction_results.page_count"),
    col("extraction_results.processing_timestamp")
)

# Trigger the processing by counting results
total_processed = processed_df.count()
processing_time = datetime.now() - start_time

print("💾 Saving results to Delta Lake...")

# Write the processed results to our output table
processed_df.write.mode("overwrite").saveAsTable(OUTPUT_TABLE)

print(f"✅ Results saved to table: {OUTPUT_TABLE}")


print(f"✅ Processing complete!")
print(f"📈 Results: {total_processed:,} documents processed in {processing_time}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Analyse Saved Results to Delta Lake
# MAGIC 
# MAGIC Kets cgecj wgat we persisted in our new Delta Lake table.
# MAGIC 
# MAGIC ### Storage Benefits in Delta Lake:
# MAGIC - **Version control** with Delta Lake time travel
# MAGIC - **ACID transactions** for data consistency
# MAGIC - **Query optimization** for fast retrieval
# MAGIC - **Schema evolution** for future enhancements

# COMMAND ----------

processed_df = spark.table(OUTPUT_TABLE)

# Calculate success metrics
success_count = processed_df.filter(F.col("extraction_success") == True).count()
failure_count = processed_df.filter(F.col("extraction_success") == False).count()
success_rate = (success_count / total_processed * 100) if total_processed > 0 else 0

print(f"\n📊 Processing Summary:")
print(f"  ✅ Successful extractions: {success_count:,}")
print(f"  ❌ Failed extractions: {failure_count:,}")
print(f"  📈 Success rate: {success_rate:.1f}%")

# Calculate content statistics for successful extractions
if success_count > 0:
    content_stats = processed_df.filter(F.col("extraction_success") == True).agg(
        F.sum("character_count").alias("total_characters"),
        F.avg("character_count").alias("avg_characters"),
        F.sum("page_count").alias("total_pages"),
        F.avg("processing_time_seconds").alias("avg_processing_time")
    ).collect()[0]
    
    print(f"\n📝 Content Statistics:")
    print(f"  Total characters extracted: {content_stats.total_characters:,}")
    print(f"  Average characters per document: {content_stats.avg_characters:,.0f}")
    print(f"  Total pages processed: {content_stats.total_pages:,}")
    print(f"  Average processing time: {content_stats.avg_processing_time:.2f} seconds")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Verification and Quality Checks
# MAGIC 
# MAGIC Verify the extraction results and provide usage examples.
# MAGIC 
# MAGIC ### Quality Metrics:
# MAGIC - **Content validation**: Check for reasonable text extraction
# MAGIC - **Error analysis**: Understand failure patterns
# MAGIC - **Performance review**: Identify optimization opportunities

# COMMAND ----------

print("🔍 Running quality checks...")

# Show a sample of successful extractions
print("\n📄 Sample of successfully processed documents:")
successful_docs = processed_df.filter(F.col("extraction_success") == True)
successful_docs.select(
    "file_name", 
    "character_count", 
    "page_count", 
    "processing_time_seconds"
).show(5, truncate=False)

# Show failed extractions if any
failed_docs = processed_df.filter(F.col("extraction_success") == False)
failure_count = failed_docs.count()

if failure_count > 0:
    print(f"\n⚠️  Failed extractions ({failure_count}):")
    failed_docs.select("file_name", "error_message").show(5, truncate=False)
    
    # Analyze failure patterns
    print("\n🔍 Failure pattern analysis:")
    failed_docs.groupBy("error_message").count().orderBy(F.desc("count")).show(truncate=False)

# Show a preview of extracted markdown content
print("\n📝 Sample of extracted markdown content:")
sample_doc = successful_docs.select("file_name", "markdown_content").first()
if sample_doc and sample_doc.markdown_content:
    preview_length = min(500, len(sample_doc.markdown_content))
    print(f"File: {sample_doc.file_name}")
    print(f"Preview (first {preview_length} characters):")
    print("-" * 50)
    print(sample_doc.markdown_content[:preview_length])
    if len(sample_doc.markdown_content) > preview_length:
        print("... (truncated)")
    print("-" * 50)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Usage Examples and Next Steps
# MAGIC 
# MAGIC Learn how to use the extracted markdown content for various applications.
# MAGIC 
# MAGIC ### Common Use Cases:
# MAGIC - **LLM Processing**: Clean text for language model training or inference
# MAGIC - **Search Applications**: Full-text search across document collection
# MAGIC - **Content Analysis**: Text analytics and document classification
# MAGIC - **Knowledge Extraction**: Information retrieval and summarization

# COMMAND ----------

print("🎯 Usage Examples:")
print("\n1. Query extracted markdown content:")
print(f"   df = spark.table('{OUTPUT_TABLE}')")
print("   successful_docs = df.filter(df.extraction_success == True)")
print("   markdown_content = successful_docs.select('file_name', 'markdown_content')")

print("\n2. Search for specific content:")
print("   search_results = df.filter(df.markdown_content.contains('your_search_term'))")
print("   search_results.select('file_name', 'character_count').show()")

print("\n3. Get content for specific document:")
print("   doc_content = df.filter(df.file_name == 'specific_file.pdf').select('markdown_content').collect()[0][0]")

print("\n4. Analyze content by file size:")
print("   df.filter(df.extraction_success == True).groupBy('page_count').agg(avg('character_count')).show()")

print("\n📈 Performance Optimization Tips:")
print("- The pandas UDF provides better performance than RDD map operations for this use case")
print("- Cache the table if you'll be running multiple queries: df.cache()")
print("- Use column pruning to select only needed columns")
print("- Consider partitioning by directory or file_extension for better query performance")
print("- Use Delta Lake's Z-ordering for columns you frequently filter on")
print("- Pandas UDF automatically handles batching and vectorization for optimal performance")

print(f"\n🎉 Markdown extraction complete! Your processed documents are ready for LLM workflows.")
print(f"📊 Table location: {OUTPUT_TABLE}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Performance and Statistics Summary
# MAGIC 
# MAGIC Final summary of processing performance and document collection statistics.

# COMMAND ----------

# Create a comprehensive summary report
summary_query = f"""
SELECT 
    'Overall Processing' as category,
    COUNT(*) as document_count,
    SUM(CASE WHEN extraction_success THEN 1 ELSE 0 END) as successful_extractions,
    ROUND(SUM(CASE WHEN extraction_success THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as success_rate_percent,
    ROUND(SUM(character_count) / 1000000.0, 2) as total_content_mb_chars,
    ROUND(AVG(processing_time_seconds), 3) as avg_processing_time_sec,
    SUM(page_count) as total_pages_processed
FROM {OUTPUT_TABLE}

UNION ALL

SELECT 
    CONCAT('File Size: ', 
        CASE 
            WHEN file_size_bytes < 1048576 THEN 'Small (<1MB)'
            WHEN file_size_bytes < 10485760 THEN 'Medium (1-10MB)' 
            ELSE 'Large (>10MB)'
        END
    ) as category,
    COUNT(*) as document_count,
    SUM(CASE WHEN extraction_success THEN 1 ELSE 0 END) as successful_extractions,
    ROUND(SUM(CASE WHEN extraction_success THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as success_rate_percent,
    ROUND(SUM(character_count) / 1000000.0, 2) as total_content_mb_chars,
    ROUND(AVG(processing_time_seconds), 3) as avg_processing_time_sec,
    SUM(page_count) as total_pages_processed
FROM {OUTPUT_TABLE}
GROUP BY 
    CASE 
        WHEN file_size_bytes < 1048576 THEN 'Small (<1MB)'
        WHEN file_size_bytes < 10485760 THEN 'Medium (1-10MB)' 
        ELSE 'Large (>10MB)'
    END
ORDER BY category
"""

print("📈 Final Processing Report:")
summary_df = spark.sql(summary_query)
summary_df.show(10, truncate=False)

print("🏁 Document markdown extraction pipeline complete!")
print("Your documents are now ready for advanced LLM processing workflows! 🚀")