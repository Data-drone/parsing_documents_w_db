# Databricks notebook source
# MAGIC %md
# MAGIC # OCR Processing with Nanonets API
# MAGIC 
# MAGIC This notebook processes PDF page images extracted from the previous notebook and converts them to text using a Nanonets OCR model via OpenAI-compatible API.
# MAGIC 
# MAGIC ## Input Schema (from previous notebook):
# MAGIC ```
# MAGIC |-- doc_id: string
# MAGIC |-- source_filename: string  
# MAGIC |-- page_number: integer
# MAGIC |-- page_image_png: binary     ← PNG image data
# MAGIC |-- total_pages: integer
# MAGIC |-- file_size_bytes: long
# MAGIC |-- processing_timestamp: string
# MAGIC |-- metadata_json: string      ← Metadata as JSON string
# MAGIC ```
# MAGIC 
# MAGIC ## Output: 
# MAGIC Adds `ocr_text` column with extracted text/markdown content.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration and Imports

# COMMAND ----------

import os
import io
import time
import logging
import requests
import base64
import pandas as pd
import ast
from typing import List, Dict, Optional

from pyspark.sql.functions import pandas_udf, col, lit
from pyspark.sql.types import StringType
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

# Configuration - Updated to match previous notebook output
CATALOG = 'brian_gen_ai'
SCHEMA = 'parsing_test'
SOURCE_TABLE = 'document_page_docs'    # Output from previous notebook
OUTPUT_TABLE = 'document_store_ocr'     # Final table with OCR results

MODEL_NAME = 'nanonets/Nanonets-OCR-s'

print(f"Source: {CATALOG}.{SCHEMA}.{SOURCE_TABLE}")
print(f"Output: {CATALOG}.{SCHEMA}.{OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Logging and API Configuration

# COMMAND ----------

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy logging messages
logging.getLogger('py4j.clientserver').setLevel(logging.WARNING)

class OpenAIAPIConfig:
    """Configuration for OpenAI-spec API that hosts the Nanonets OCR model"""
    def __init__(self):
        # Get config from environment variables or set defaults
        self.api_url = os.getenv("OPENAI_API_URL", "http://localhost:8000/v1")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model_name = os.getenv("OPENAI_MODEL_NAME", MODEL_NAME)
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
        self.timeout = int(os.getenv("OPENAI_TIMEOUT", "300"))  # 5 minutes
        self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
        self.max_workers = int(os.getenv("OPENAI_MAX_WORKERS", "3"))  # Reduced for stability
        self.rate_limit_delay = float(os.getenv("OPENAI_RATE_LIMIT_DELAY", "0.5"))  # Increased delay

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Loading and Validation

# COMMAND ----------

# Load the PDF pages from previous notebook
print("=== LOADING PDF PAGES DATA ===")
pages_df = spark.table(f"{CATALOG}.{SCHEMA}.{SOURCE_TABLE}")

# Basic validation
total_pages = pages_df.count()
unique_docs = pages_df.select("doc_id").distinct().count()

print(f"Total pages to process: {total_pages:,}")
print(f"Unique documents: {unique_docs:,}")
print(f"Average pages per document: {total_pages/unique_docs:.1f}")

# Sample the data structure
print("\n=== SAMPLE DATA STRUCTURE ===")
sample_data = pages_df.select(
    "doc_id", 
    "source_filename", 
    "page_number", 
    "total_pages",
    "LENGTH(page_image_png) as image_size_bytes"
).limit(5)

display(sample_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## OCR Processing Functions
# MAGIC 
# MAGIC These functions handle the conversion of PNG page images to text using the Nanonets OCR model.

# COMMAND ----------

def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for API transmission"""
    buffer = io.BytesIO()
    # Convert to RGB if necessary (removes transparency)
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image.save(buffer, format='JPEG', quality=85, optimize=True)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def make_api_request(image: Image.Image, config: OpenAIAPIConfig, image_path: str = "unknown") -> Optional[str]:
    """
    Make a single OCR API request for one page image.
    
    Args:
        image: PIL Image object
        config: API configuration
        image_path: Identifier for logging
        
    Returns:
        Extracted text/markdown or None if failed
    """
    
    # Detailed OCR prompt for high-quality extraction
    prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
    
    try:
        # Encode image for API
        base64_image = encode_image_to_base64(image)
        
        # Prepare API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}"
        }
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that specializes in OCR and document analysis."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        payload = {
            "model": config.model_name,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature
        }
        
        # Retry logic with exponential backoff
        for attempt in range(config.max_retries + 1):
            try:
                response = requests.post(
                    f"{config.api_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=(30, config.timeout)
                )
                
                response.raise_for_status()
                result = response.json()
                
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    logger.warning(f"No valid response for {image_path}")
                    return None
                    
            except requests.exceptions.Timeout:
                if attempt < config.max_retries:
                    sleep_time = 2 ** attempt
                    logger.warning(f"Request timeout for {image_path}, retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Final timeout for {image_path}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed for {image_path}: {str(e)}")
                return None
                
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

def prepare_api_inputs(page_images: List[bytes], metadata_list: List[str]) -> List[Dict]:
    """
    Prepare image data and metadata for API processing.
    
    Args:
        page_images: List of PNG image bytes from page_image_png column
        metadata_list: List of JSON metadata strings from metadata_json column
        
    Returns:
        List of prepared input dictionaries
    """
    inputs = []
    
    for i, (image_bytes, metadata_str) in enumerate(zip(page_images, metadata_list)):
        try:
            # Parse metadata to get source info
            try:
                metadata = ast.literal_eval(metadata_str)
                image_path = f"{metadata.get('source_filename', 'unknown')}_page_{i+1}"
            except:
                image_path = f'image_{i}'
            
            # Convert PNG bytes to PIL Image
            if image_bytes and len(image_bytes) > 0:
                image = Image.open(io.BytesIO(image_bytes))
                inputs.append({
                    'image': image,
                    'image_path': image_path,
                    'index': i
                })
            else:
                logger.warning(f"Empty image data for {image_path}")
                inputs.append({
                    'image': None,
                    'image_path': image_path,
                    'index': i
                })
            
        except Exception as e:
            logger.error(f"Error preparing input {i}: {e}")
            inputs.append({
                'image': None,
                'image_path': f'failed_image_{i}',
                'index': i
            })
    
    return inputs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pandas UDF for Batch OCR Processing
# MAGIC 
# MAGIC This UDF processes batches of page images concurrently using the Nanonets OCR API.

# COMMAND ----------

@pandas_udf(returnType=StringType())
def nanonets_ocr_udf(page_images: pd.Series, metadata_series: pd.Series) -> pd.Series:
    """
    High-performance pandas UDF for batch OCR processing using Nanonets via OpenAI-spec API.
    
    Args:
        page_images: Series of PNG image bytes (from page_image_png column)
        metadata_series: Series of JSON metadata strings (from metadata_json column)
        
    Returns:
        Series of extracted text/markdown content
    """
    
    batch_size = len(page_images)
    logger.info(f"Processing OCR batch of {batch_size} images")
    
    # Initialize API config
    config = OpenAIAPIConfig()
    
    # Set API key (you may want to configure this differently)
    if not config.api_key:
        config.api_key = ""  # Set your API key here or via environment variable
        logger.warning("API key not configured - set OPENAI_API_KEY environment variable")
    
    # Prepare inputs
    inputs = prepare_api_inputs(
        page_images=page_images.tolist(),
        metadata_list=metadata_series.tolist()
    )
    
    # Filter valid inputs
    valid_inputs = [inp for inp in inputs if inp['image'] is not None]
    
    if not valid_inputs:
        logger.warning("No valid images to process in this batch")
        return pd.Series([None] * batch_size)
    
    # Initialize results array
    results = [None] * batch_size
    
    try:
        logger.info(f"Processing {len(valid_inputs)} valid images with {config.max_workers} workers")
        
        # Process images concurrently with rate limiting
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # Submit requests with rate limiting
            future_to_input = {}
            for inp in valid_inputs:
                time.sleep(config.rate_limit_delay)  # Rate limiting
                future = executor.submit(make_api_request, inp['image'], config, inp['image_path'])
                future_to_input[future] = inp
            
            # Collect results as they complete
            for future in as_completed(future_to_input):
                inp = future_to_input[future]
                try:
                    result = future.result()
                    results[inp['index']] = result
                except Exception as e:
                    logger.error(f"Error processing {inp['image_path']}: {e}")
                    results[inp['index']] = None
        
        successful_results = sum(1 for r in results if r is not None)
        logger.info(f"OCR batch completed: {successful_results}/{batch_size} successful")
        
        return pd.Series(results)
        
    except Exception as e:
        logger.error(f"Error in OCR batch processing: {e}")
        return pd.Series([None] * batch_size)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute OCR Processing
# MAGIC 
# MAGIC Process all page images to extract text content using the Nanonets OCR model.

# COMMAND ----------

print("=== STARTING OCR PROCESSING ===")
start_time = time.time()

# Apply OCR UDF to extract text from page images
# Updated column names to match actual schema: page_image_png and metadata_json
ocr_results_df = pages_df.withColumn(
    'ocr_text', 
    nanonets_ocr_udf(col('page_image_png'), col('metadata_json'))
)

# Add processing metadata
ocr_results_df = ocr_results_df.withColumn(
    'ocr_timestamp', 
    lit(time.strftime("%Y-%m-%d %H:%M:%S"))
).withColumn(
    'ocr_model',
    lit(MODEL_NAME)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Results
# MAGIC 
# MAGIC Save the OCR results to a new Delta table for further analysis.

# COMMAND ----------

print("=== SAVING OCR RESULTS ===")

# Save to Delta table
(ocr_results_df
 .write
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .partitionBy("doc_id")
 .saveAsTable(f"{CATALOG}.{SCHEMA}.{OUTPUT_TABLE}")
)

end_time = time.time()
processing_time = end_time - start_time

print(f"✅ OCR processing completed!")
print(f"Results saved to: {CATALOG}.{SCHEMA}.{OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Processing Summary and Results

# COMMAND ----------

print("=== OCR PROCESSING SUMMARY ===")

# Get final counts
total_pages_processed = spark.table(f"{CATALOG}.{SCHEMA}.{OUTPUT_TABLE}").count()
successful_ocr = spark.table(f"{CATALOG}.{SCHEMA}.{OUTPUT_TABLE}").filter(col("ocr_text").isNotNull()).count()
failed_ocr = total_pages_processed - successful_ocr

print(f"Processing Statistics:")
print(f"  - Total pages processed: {total_pages_processed:,}")
print(f"  - Successful OCR extractions: {successful_ocr:,}")
print(f"  - Failed OCR extractions: {failed_ocr:,}")
print(f"  - Success rate: {(successful_ocr/total_pages_processed)*100:.1f}%")
print(f"  - Total processing time: {processing_time:.1f} seconds")
print(f"  - Pages per second: {total_pages_processed/processing_time:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample OCR Results

# COMMAND ----------

# Show sample OCR results
sample_ocr = spark.sql(f"""
SELECT 
    doc_id,
    source_filename,
    page_number,
    total_pages,
    CASE 
        WHEN ocr_text IS NOT NULL THEN 'SUCCESS'
        ELSE 'FAILED'
    END as ocr_status,
    LENGTH(ocr_text) as text_length,
    LEFT(ocr_text, 200) as text_preview
FROM {CATALOG}.{SCHEMA}.{OUTPUT_TABLE}
ORDER BY source_filename, page_number
LIMIT 10
""")

print("=== SAMPLE OCR RESULTS ===")
display(sample_ocr)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality Analysis

# COMMAND ----------

# Analyze OCR quality by text length and success rates
quality_analysis = spark.sql(f"""
SELECT 
    CASE 
        WHEN LENGTH(ocr_text) = 0 OR ocr_text IS NULL THEN 'NO_TEXT'
        WHEN LENGTH(ocr_text) < 100 THEN 'SHORT_TEXT'
        WHEN LENGTH(ocr_text) < 500 THEN 'MEDIUM_TEXT'
        WHEN LENGTH(ocr_text) < 2000 THEN 'LONG_TEXT'
        ELSE 'VERY_LONG_TEXT'
    END as text_category,
    COUNT(*) as page_count,
    AVG(LENGTH(ocr_text)) as avg_text_length,
    MIN(LENGTH(ocr_text)) as min_text_length,
    MAX(LENGTH(ocr_text)) as max_text_length
FROM {CATALOG}.{SCHEMA}.{OUTPUT_TABLE}
GROUP BY 1
ORDER BY page_count DESC
""")

print("=== OCR QUALITY ANALYSIS ===")
display(quality_analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC 
# MAGIC ## Processing Complete!
# MAGIC 
# MAGIC ### Key Changes Made:
# MAGIC 1. **Updated table reference**: Changed from `split_pages` to `document_store_blob`
# MAGIC 2. **Fixed column names**: Updated `page_bytes` → `page_image_png` and `metadata` → `metadata_json`
# MAGIC 3. **Improved error handling**: Better logging and retry mechanisms
# MAGIC 4. **Enhanced documentation**: Clear markdown sections explaining each step
# MAGIC 5. **Quality analysis**: Added OCR quality assessment
# MAGIC 6. **Performance tuning**: Reduced concurrent workers and increased rate limiting for stability
# MAGIC 
# MAGIC ### Output Schema:
# MAGIC The final table includes all original columns plus:
# MAGIC - `ocr_text`: Extracted text/markdown content
# MAGIC - `ocr_timestamp`: When OCR was performed  
# MAGIC - `ocr_model`: Model used for OCR
# MAGIC 
# MAGIC ### Next Steps:
# MAGIC - Review OCR quality and adjust API parameters if needed
# MAGIC - Consider post-processing to clean up extracted text
# MAGIC - Implement document reconstruction by combining pages