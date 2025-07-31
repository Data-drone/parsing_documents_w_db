# Databricks notebook source
# MAGIC %md
# MAGIC # PDF to Image Conversion for Vision Language Model (VLM) Processing
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook converts PDF documents into individual page images to enable processing with Vision Language Models (VLMs). Unlike traditional text extraction methods, VLMs require visual input to understand document structure, layouts, charts, diagrams, and complex formatting that would be lost in text-only extraction.
# MAGIC 
# MAGIC ## Why Convert PDFs to Images for VLM Processing?
# MAGIC 
# MAGIC ### The VLM Advantage
# MAGIC - **Visual Understanding**: VLMs can interpret charts, graphs, tables, and complex layouts
# MAGIC - **Spatial Awareness**: Understand relationships between visual elements 
# MAGIC - **Format Preservation**: Maintain document structure and formatting context
# MAGIC - **Multimodal Processing**: Combine visual and textual understanding
# MAGIC 
# MAGIC ### Use Cases
# MAGIC - **Financial Reports**: Extract data from complex tables and charts
# MAGIC - **Research Papers**: Understand figures, equations, and formatting
# MAGIC - **Forms and Documents**: Process structured layouts and handwritten content
# MAGIC - **Presentations**: Analyze slide layouts and visual content
# MAGIC 
# MAGIC ## Distributed Processing Benefits
# MAGIC 
# MAGIC This notebook leverages Apache Spark's distributed computing to:
# MAGIC - **Scale Processing**: Handle thousands of documents in parallel
# MAGIC - **Memory Management**: Process large files efficiently across cluster nodes
# MAGIC - **Fault Tolerance**: Recover from failures during processing
# MAGIC - **Cost Optimization**: Use cluster resources efficiently
# MAGIC 
# MAGIC ## Configuration
# MAGIC 
# MAGIC This notebook uses **Databricks widgets** for runtime configuration, following the same pattern as our foundation tutorials. The widgets default to environment variables from your `.env` file but can be overridden at runtime.
# MAGIC 
# MAGIC ### Widget Configuration Options:
# MAGIC - **Catalog Name**: Unity Catalog where tables are stored
# MAGIC - **Schema Name**: Schema within the catalog for document processing
# MAGIC - **Source Table Name**: Name of the table containing PDF documents (typically `document_store_blob`)
# MAGIC - **Output Table Name**: Name for the output table with page images (typically `document_page_docs`)
# MAGIC - **High Resolution Images**: Toggle for 2x resolution (larger files, better quality)
# MAGIC - **Max File Size (MB)**: Maximum PDF size to process (prevents memory issues)
# MAGIC - **Partition Count**: Number of Spark partitions for distributed processing
# MAGIC 
# MAGIC ### Environment File Setup (.env):
# MAGIC ```
# MAGIC CATALOG_NAME='your_catalog'
# MAGIC INPUT_SCHEMA='your_schema'
# MAGIC SOURCE_TABLE='document_store_blob'
# MAGIC OUTPUT_TABLE='document_page_docs'
# MAGIC USE_HIGH_RESOLUTION='false'
# MAGIC MAX_FILE_SIZE_MB='50'
# MAGIC REPARTITION_COUNT='8'
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install pymupdf python-dotenv
# MAGIC %restart_python

# COMMAND ----------

import os
import pandas as pd
import fitz  # PyMuPDF
import logging
import time
from datetime import datetime
import uuid
import gc
from dotenv import load_dotenv, find_dotenv

from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, col

# Load environment variables from .env file (if present) **before** we read them via os.getenv
load_dotenv(find_dotenv())

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Runtime Configuration with Widgets
# MAGIC 
# MAGIC Following the same pattern as our foundation tutorials, we use Databricks widgets to allow runtime configuration while defaulting to environment variables. This provides flexibility for different users and environments.

# COMMAND ----------

# -----------------------------------------------------------------------------
# Runtime configuration via widgets
# -----------------------------------------------------------------------------
# Pull defaults from environment variables (populated via `.env` or workspace)

# Create Databricks widgets so users can override at run-time
dbutils.widgets.text("catalog_name", os.getenv("CATALOG_NAME", "brian_gen_ai"), "Catalog Name")
dbutils.widgets.text("schema_name", os.getenv("INPUT_SCHEMA", "parsing_test"), "Schema Name") 
dbutils.widgets.text("source_table", os.getenv("SOURCE_TABLE", "document_store_blob"), "Source Table Name")
dbutils.widgets.text("output_table", os.getenv("OUTPUT_TABLE", "document_page_docs"), "Output Table Name")

# Image processing settings
dbutils.widgets.dropdown("use_high_resolution", os.getenv("USE_HIGH_RESOLUTION", "false"), ["true", "false"], "High Resolution Images")
dbutils.widgets.text("max_file_size_mb", os.getenv("MAX_FILE_SIZE_MB", "50"), "Max File Size (MB)")
dbutils.widgets.text("repartition_count", os.getenv("REPARTITION_COUNT", "8"), "Partition Count")

# Read values back from the widgets
CATALOG = dbutils.widgets.get("catalog_name")
INPUT_SCHEMA = dbutils.widgets.get("schema_name")
SOURCE_TABLE = dbutils.widgets.get("source_table")
OUTPUT_TABLE = dbutils.widgets.get("output_table")

# Image processing settings
USE_HIGH_RESOLUTION = dbutils.widgets.get("use_high_resolution").lower() == "true"
MAX_FILE_SIZE_MB = int(dbutils.widgets.get("max_file_size_mb"))
REPARTITION_COUNT = int(dbutils.widgets.get("repartition_count"))

# Construct full table names
SOURCE_TABLE_FULL = f"{CATALOG}.{INPUT_SCHEMA}.{SOURCE_TABLE}"
OUTPUT_TABLE_FULL = f"{CATALOG}.{INPUT_SCHEMA}.{OUTPUT_TABLE}"

print("=== PDF to Image Conversion Configuration ===")
print(f"Source Table: {SOURCE_TABLE_FULL}")
print(f"Output Table: {OUTPUT_TABLE_FULL}")
print(f"High Resolution: {USE_HIGH_RESOLUTION}")
print(f"Max File Size: {MAX_FILE_SIZE_MB} MB")
print(f"Partition Count: {REPARTITION_COUNT}")
print("=" * 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔍 Configuration Verification
# MAGIC 
# MAGIC Let's verify that our configuration is correct and the source table contains processable PDF documents.

# COMMAND ----------

# Verify source table exists and contains PDFs
try:
    # Check if source table exists and get basic stats
    table_info = spark.sql(f"DESCRIBE TABLE EXTENDED {SOURCE_TABLE_FULL}")
    print("✅ Source table exists")
    
    # Get PDF document counts
    pdf_stats = spark.sql(f"""
        SELECT 
            COUNT(*) as total_documents,
            COUNT(CASE WHEN file_extension = '.pdf' THEN 1 END) as pdf_documents,
            COUNT(CASE WHEN binary_content IS NOT NULL THEN 1 END) as documents_with_content,
            COUNT(CASE WHEN file_extension = '.pdf' AND binary_content IS NOT NULL 
                       AND file_size_bytes <= {MAX_FILE_SIZE_MB * 1024 * 1024} THEN 1 END) as processable_pdfs
        FROM {SOURCE_TABLE_FULL}
    """).collect()[0]
    
    print(f"📊 Document Statistics:")
    print(f"   Total documents: {pdf_stats['total_documents']:,}")
    print(f"   PDF documents: {pdf_stats['pdf_documents']:,}")
    print(f"   Documents with content: {pdf_stats['documents_with_content']:,}")
    print(f"   Processable PDFs: {pdf_stats['processable_pdfs']:,}")
    
    if pdf_stats['processable_pdfs'] == 0:
        print("⚠️  No processable PDF documents found. Check your source table and file size limits.")
    else:
        print(f"✅ Ready to process {pdf_stats['processable_pdfs']:,} PDF documents")
        
except Exception as e:
    print(f"❌ Error accessing source table: {e}")
    print("Please check your catalog, schema, and table names in the widgets above.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Core Processing Functions
# MAGIC 
# MAGIC The following functions handle document ID generation, metadata creation, and the distributed PDF-to-image conversion process.

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

# COMMAND ----------

# Define output schema
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

# COMMAND ----------

def extract_pdf_pages_batch(iterator):
    """Extract pages from PDF binary data as PNG images"""
    
    for pdf_batch in iterator:
        results = []
        
        try:
            # Ensure we have a proper DataFrame
            if hasattr(pdf_batch, 'to_dict'):
                records = pdf_batch.to_dict('records')
            else:
                logger.error(f"Unexpected data type: {type(pdf_batch)}")
                yield pd.DataFrame(columns=[field.name for field in pdf_pages_schema.fields])
                continue
            
            logger.info(f"Processing {len(records)} PDF files in batch")
            
            # Process each record
            for record in records:
                filename = record.get('file_name', 'unknown')
                filepath = record.get('volume_path', '') 
                binary_content = record.get('binary_content')
                file_size = record.get('file_size_bytes', 0)
                
                doc = None
                
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
                        page = None
                        pix = None
                        
                        try:
                            page = doc[page_num]
                            
                            # Create page image
                            if USE_HIGH_RESOLUTION:
                                matrix = fitz.Matrix(2, 2)
                                pix = page.get_pixmap(matrix=matrix, alpha=False)
                            else:
                                pix = page.get_pixmap(alpha=False)
                            
                            # Convert to PNG bytes
                            img_bytes = pix.pil_tobytes(format="PNG")
                            
                            results.append({
                                "doc_id": doc_id,
                                "source_filename": filename,
                                "page_number": page_num + 1,
                                "page_image_png": img_bytes,
                                "total_pages": total_pages,
                                "file_size_bytes": file_size,
                                "processing_timestamp": datetime.now().isoformat(),
                                "metadata_json": str(metadata)
                            })
                            
                        except Exception as page_error:
                            logger.error(f"Error processing page {page_num + 1} of {filename}: {str(page_error)}")
                            continue
                        finally:
                            if pix is not None:
                                pix = None
                            if page is not None:
                                page = None
                    
                    logger.info(f"Successfully processed {filename}: {total_pages} pages")
                    
                except Exception as e:
                    logger.error(f"Error processing PDF {filename}: {str(e)}")
                    continue
                finally:
                    if doc is not None:
                        doc.close()
                        doc = None
            
            if len(results) > 0:
                gc.collect()
            
            if not results:
                yield pd.DataFrame(columns=[field.name for field in pdf_pages_schema.fields])
            else:
                yield pd.DataFrame(results)
                
        except Exception as batch_error:
            logger.error(f"Error processing entire batch: {str(batch_error)}")
            yield pd.DataFrame(columns=[field.name for field in pdf_pages_schema.fields])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Processing Pipeline
# MAGIC 
# MAGIC Now we'll execute the PDF-to-image conversion using distributed Spark processing. The pipeline:
# MAGIC 1. Loads and filters PDF documents from the source table
# MAGIC 2. Processes documents in parallel across the cluster 
# MAGIC 3. Converts each PDF page to a PNG image
# MAGIC 4. Saves results to the output table with partitioning

# COMMAND ----------

# Load and filter source data
source_df = spark.table(SOURCE_TABLE_FULL)

processable_files = source_df.filter(
    (col("file_extension") == ".pdf") &
    (col("binary_content").isNotNull()) &
    (col("file_size_bytes") <= MAX_FILE_SIZE_MB * 1024 * 1024)
)

processable_count = processable_files.count()
print(f"Files eligible for processing: {processable_count:,}")

# COMMAND ----------

# Process PDFs
print("=== PROCESSING PDFs ===")
start_time = time.time()

pages_df = processable_files.select(
    col("file_name"),
    col("volume_path"),
    col("binary_content"),
    col("file_size_bytes")
).repartition(REPARTITION_COUNT).mapInPandas(
    extract_pdf_pages_batch,
    schema=pdf_pages_schema
)

# Save results
(pages_df
 .write
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .partitionBy("doc_id")
 .saveAsTable(OUTPUT_TABLE_FULL)
)

end_time = time.time()
processing_time = end_time - start_time
print(f"Processing completed in {processing_time:.1f} seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Processing Results and Quality Validation
# MAGIC 
# MAGIC Let's examine the results of our PDF-to-image conversion and validate the output quality.

# COMMAND ----------

# Verification queries
print("=== RESULTS ===")

# Get counts
final_pages_count = spark.table(OUTPUT_TABLE_FULL).count()
final_docs_count = spark.table(OUTPUT_TABLE_FULL).select("doc_id").distinct().count()

print(f"Documents processed: {final_docs_count:,}")
print(f"Total pages extracted: {final_pages_count:,}")
print(f"Average pages per document: {final_pages_count/final_docs_count:.1f}")

# COMMAND ----------

# Sample results
sample_results = spark.sql(f"""
SELECT 
    source_filename,
    page_number,
    total_pages,
    file_size_bytes,
    LENGTH(page_image_png) as image_size_bytes
FROM {OUTPUT_TABLE_FULL}
ORDER BY source_filename, page_number
LIMIT 10
""")

display(sample_results)

# COMMAND ----------

# Data validation
validation_df = spark.sql(f"""
SELECT 
    COUNT(DISTINCT doc_id) as unique_documents,
    COUNT(*) as total_pages,
    AVG(total_pages) as avg_pages_per_doc,
    MIN(page_number) as min_page_num,
    MAX(page_number) as max_page_num,
    AVG(LENGTH(page_image_png)) as avg_image_size_bytes
FROM {OUTPUT_TABLE_FULL}
""")

display(validation_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Processing Results and Next Steps
# MAGIC 
# MAGIC ### What We've Accomplished
# MAGIC 
# MAGIC This notebook has successfully:
# MAGIC 1. **Converted PDF documents to PNG images**: Each page is now a high-quality image ready for VLM processing
# MAGIC 2. **Preserved document metadata**: File information, processing timestamps, and document structure are maintained
# MAGIC 3. **Distributed processing**: Used Spark's parallel processing capabilities for efficient bulk conversion
# MAGIC 4. **Quality validation**: Verified the conversion results and data integrity
# MAGIC 
# MAGIC ### Output Data Structure
# MAGIC 
# MAGIC The resulting table contains:
# MAGIC - `doc_id`: Unique identifier for each document
# MAGIC - `source_filename`: Original PDF filename
# MAGIC - `page_number`: Page number within the document (1-indexed)
# MAGIC - `page_image_png`: Binary PNG image data for VLM processing
# MAGIC - `total_pages`: Total pages in the source document
# MAGIC - `file_size_bytes`: Original PDF file size
# MAGIC - `processing_timestamp`: When the conversion was performed
# MAGIC - `metadata_json`: Additional document metadata
# MAGIC 
# MAGIC ### Next Steps for VLM Processing
# MAGIC 
# MAGIC With your documents now converted to images, you can:
# MAGIC 
# MAGIC 1. **Vision Language Model Analysis**: Use models like GPT-4V, Claude Vision, or DALL-E to analyze document content
# MAGIC 2. **Structured Data Extraction**: Extract tables, charts, and form data with visual understanding
# MAGIC 3. **Document Classification**: Classify documents based on visual layout and content
# MAGIC 4. **Quality Assessment**: Identify page quality, orientation, or scanning issues
# MAGIC 5. **Multimodal RAG**: Combine visual and textual information for comprehensive document understanding
# MAGIC 
# MAGIC ### Performance Considerations
# MAGIC 
# MAGIC - **High Resolution Setting**: Enable `USE_HIGH_RESOLUTION=true` for detailed analysis but larger file sizes
# MAGIC - **Partition Count**: Adjust `REPARTITION_COUNT` based on cluster size and document volume
# MAGIC - **File Size Limits**: Modify `MAX_FILE_SIZE_MB` based on processing capacity and requirements
# MAGIC 
# MAGIC ### Error Handling and Monitoring
# MAGIC 
# MAGIC The pipeline includes robust error handling:
# MAGIC - Individual page processing failures don't stop the entire batch
# MAGIC - Large files are automatically skipped with warnings
# MAGIC - Processing statistics are logged for monitoring
# MAGIC - Memory management prevents cluster resource exhaustion
# MAGIC 
# MAGIC Continue to the next notebook in the series for VLM-based document analysis and extraction techniques.