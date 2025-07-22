# Databricks notebook source
# MAGIC %md
# MAGIC # PDF Page Splitting Pipeline

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
import gc

from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, col

# Configuration
CATALOG = "brian_gen_ai"
INPUT_SCHEMA = "parsing_test" 
SOURCE_TABLE = "document_store_blob"
OUTPUT_TABLE = "document_page_docs"

USE_HIGH_RESOLUTION = False
MAX_FILE_SIZE_MB = 50

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"Source: {CATALOG}.{INPUT_SCHEMA}.{SOURCE_TABLE}")
print(f"Output: {CATALOG}.{INPUT_SCHEMA}.{OUTPUT_TABLE}")

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

# Load and filter source data
source_df = spark.table(f"{CATALOG}.{INPUT_SCHEMA}.{SOURCE_TABLE}")

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
).repartition(8).mapInPandas(
    extract_pdf_pages_batch,
    schema=pdf_pages_schema
)

# Save results
(pages_df
 .write
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .partitionBy("doc_id")
 .saveAsTable(f"{CATALOG}.{INPUT_SCHEMA}.{OUTPUT_TABLE}")
)

end_time = time.time()
processing_time = end_time - start_time
print(f"Processing completed in {processing_time:.1f} seconds")

# COMMAND ----------

# Verification queries
print("=== RESULTS ===")

# Get counts
final_pages_count = spark.table(f"{CATALOG}.{INPUT_SCHEMA}.{OUTPUT_TABLE}").count()
final_docs_count = spark.table(f"{CATALOG}.{INPUT_SCHEMA}.{OUTPUT_TABLE}").select("doc_id").distinct().count()

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
FROM {CATALOG}.{INPUT_SCHEMA}.{OUTPUT_TABLE}
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
FROM {CATALOG}.{INPUT_SCHEMA}.{OUTPUT_TABLE}
""")

display(validation_df)