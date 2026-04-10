# Databricks notebook source
# MAGIC %md
# MAGIC # Query Self-Hosted VLM Server
# MAGIC
# MAGIC Query a vLLM server (started by `01_deploy_vllm_server.py`) for document OCR.
# MAGIC The server exposes an OpenAI-compatible API that this notebook calls with page images.
# MAGIC
# MAGIC ## Features
# MAGIC - Concurrent requests with configurable worker count
# MAGIC - Exponential backoff for rate limiting (429 errors)
# MAGIC - Automatic retries with timeout handling
# MAGIC - Image optimization for network transmission
# MAGIC
# MAGIC ## Prerequisites
# MAGIC 1. Run `01_deploy_vllm_server.py` on a GPU cluster (server must be running)
# MAGIC 2. Have page images in a source table (from `00_setup/02_prepare_documents`)

# COMMAND ----------

%pip install Pillow
%restart_python

# COMMAND ----------

import os
import io
import time
import requests
import base64
import pandas as pd
import ast
from typing import Optional
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from pyspark.sql.functions import pandas_udf, col, lit
from pyspark.sql.types import StringType
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

# COMMAND ----------
# MAGIC %md
# MAGIC ## Runtime Configuration with Widgets

# COMMAND ----------

import re

current_user = spark.sql("SELECT current_user()").first()[0]
username = re.sub(r"[^a-z0-9_]", "_", current_user.split("@")[0].lower()).strip("_")
username = re.sub(r"^[0-9]+", "", username) or "user"

dbutils.widgets.text("catalog_name", f"{username}_document_parsing", "Catalog Name")
dbutils.widgets.text("schema_name", "tutorials", "Schema Name")
dbutils.widgets.text("source_table", "document_page_images", "Source Table Name")
dbutils.widgets.text("output_table", "parsed_self_hosted_vlm", "Output Table Name")

# OpenAI API Configuration (vLLM exposes OpenAI-compatible API)
dbutils.widgets.text("openai_api_url", "http://localhost:8000/v1", "VLM Server URL")
dbutils.widgets.text("openai_api_key", "dummy", "API Key (any value for local vLLM)")
dbutils.widgets.text("openai_model_name", "rednote-hilab/dots.ocr", "Model Name")
dbutils.widgets.text("openai_max_tokens", "4096", "Max Tokens")
dbutils.widgets.text("openai_temperature", "0.0", "Temperature")

dbutils.widgets.text("max_workers", "8", "Max Workers")

# Read values back from the widgets
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
SOURCE_TABLE = dbutils.widgets.get("source_table")
OUTPUT_TABLE = dbutils.widgets.get("output_table")

# OpenAI Configuration
OPENAI_API_URL = dbutils.widgets.get("openai_api_url")
OPENAI_API_KEY = dbutils.widgets.get("openai_api_key")
OPENAI_MODEL_NAME = dbutils.widgets.get("openai_model_name")
OPENAI_MAX_TOKENS = int(dbutils.widgets.get("openai_max_tokens"))
OPENAI_TEMPERATURE = float(dbutils.widgets.get("openai_temperature"))

# Simple Performance Configuration
MAX_WORKERS = int(dbutils.widgets.get("max_workers"))

# Construct full table names
SOURCE_TABLE_FULL = f"{CATALOG}.{SCHEMA}.{SOURCE_TABLE}"
OUTPUT_TABLE_FULL = f"{CATALOG}.{SCHEMA}.{OUTPUT_TABLE}"

print("=== Simple VLM Server Processing Configuration ===")
print(f"Source Table: {SOURCE_TABLE_FULL}")
print(f"Output Table: {OUTPUT_TABLE_FULL}")
print(f"API URL: {OPENAI_API_URL}")
print(f"Model: {OPENAI_MODEL_NAME}")
print(f"Max Tokens: {OPENAI_MAX_TOKENS}")
print(f"Temperature: {OPENAI_TEMPERATURE}")
print(f"Max Workers: {MAX_WORKERS}")
print("=" * 55)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Source Table

# COMMAND ----------

try:
    page_count = spark.table(SOURCE_TABLE_FULL).filter("page_image_png IS NOT NULL").count()
    print(f"Source table: {SOURCE_TABLE_FULL}")
    print(f"Pages with images: {page_count}")
    if page_count == 0:
        print("No page images found. Run 00_setup/02_prepare_documents first.")
except Exception as e:
    print(f"Error: {e}")
    print("Run 00_setup/02_prepare_documents first to create page images.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Simple Remote Server Configuration
# MAGIC 
# MAGIC Simplified configuration for handling remote VLM servers with basic rate limiting and retry logic.

# COMMAND ----------

class SimpleVLMConfig:
    """Simple configuration for remote VLM server processing"""
    def __init__(self, 
                 api_url=None, 
                 api_key=None, 
                 model_name=None, 
                 max_tokens=None, 
                 temperature=None,
                 max_workers=None):
        
        # API Configuration
        self.api_url = api_url or "http://localhost:8000/v1"
        self.api_key = api_key or "dummy"
        self.model_name = model_name or "rednote-hilab/dots.ocr"
        self.max_tokens = max_tokens or 4096
        self.temperature = temperature if temperature is not None else 0.0
        
        # Simple concurrency and retry settings
        self.max_workers = max_workers or 8
        self.base_delay = 0.1  # 100ms base delay between requests
        self.max_retries = 3
        self.timeout = (10, 60)  # (connect, read) timeouts
        
        # Image optimization (keep for efficiency)
        self.jpeg_quality = 70
        self.max_image_size = 2048
        
        print(f"Simple VLM Config: {self.max_workers} workers, {self.base_delay}s delay, {self.max_retries} retries")

def simple_backoff_delay(attempt: int, base_delay: float = 0.1) -> float:
    """Simple exponential backoff for rate limiting"""
    return min(base_delay * (2 ** attempt), 5.0)  # Cap at 5 seconds

# COMMAND ----------
# MAGIC %md
# MAGIC ## Simple Session and Request Functions

# COMMAND ----------

def create_simple_session(config: SimpleVLMConfig) -> requests.Session:
    """Create session for remote VLM server requests"""
    session = requests.Session()
    
    # Simple retry strategy for server errors (not 429)
    retry_strategy = Retry(
        total=2,
        status_forcelist=[500, 502, 503, 504],
        backoff_factor=0.5,
        allowed_methods=["POST"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    session.headers.update({
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
        "Connection": "keep-alive"
    })
    
    return session

def encode_image_optimized(image: Image.Image, config: SimpleVLMConfig) -> str:
    """Optimized image encoding"""
    # Convert and resize if needed
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    
    width, height = image.size
    max_size = config.max_image_size
    
    if width > max_size or height > max_size:
        ratio = min(max_size / width, max_size / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=config.jpeg_quality, optimize=True)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def make_simple_api_request(
    image: Image.Image,
    config: SimpleVLMConfig,
    session: requests.Session,
    image_path: str = "unknown"
) -> Optional[str]:
    """Simple API request with basic retry logic for remote servers"""
    
    prompt = """Extract all text from this document. Format tables as HTML, equations as LaTeX. Use ☐/☑ for checkboxes."""
    
    for attempt in range(config.max_retries):
        try:
            # Basic delay between requests
            if attempt > 0:
                delay = simple_backoff_delay(attempt, config.base_delay)
                time.sleep(delay)
            
            base64_image = encode_image_optimized(image, config)
            
            payload = {
                "model": config.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                "max_tokens": config.max_tokens,
                "temperature": config.temperature
            }
            
            response = session.post(
                f"{config.api_url.rstrip('/')}/chat/completions",
                json=payload,
                timeout=config.timeout
            )
            
            # Handle rate limiting (429) - wait and retry
            if response.status_code == 429:
                wait_time = 5 * (attempt + 1)  # Increase wait time with attempts
                print(f"Rate limited, waiting {wait_time}s before retry (attempt {attempt + 1})")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {image_path} (attempt {attempt + 1}): {str(e)}")
            if attempt == config.max_retries - 1:  # Last attempt
                return None
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    return None

# COMMAND ----------
# MAGIC %md
# MAGIC ## Simple VLM Processing Pandas UDF

# COMMAND ----------

@pandas_udf(returnType=StringType())
def simple_vlm_udf(page_images: pd.Series, metadata_series: pd.Series) -> pd.Series:
    """
    Simple VLM processing UDF for remote servers
    """
    
    batch_size = len(page_images)
    print(f"Processing VLM batch of {batch_size} images")
    
    config = SimpleVLMConfig(
        api_url=OPENAI_API_URL,
        api_key=OPENAI_API_KEY,
        model_name=OPENAI_MODEL_NAME,
        max_tokens=OPENAI_MAX_TOKENS,
        temperature=OPENAI_TEMPERATURE,
        max_workers=MAX_WORKERS
    )
    
    if not config.api_key:
        config.api_key = ""
    
    # Prepare inputs - simple approach
    inputs = []
    for i, (image_bytes, metadata_str) in enumerate(zip(page_images, metadata_series)):
        try:
            try:
                metadata = ast.literal_eval(metadata_str)
                image_path = f"{metadata.get('source_filename', 'unknown')}_p{i+1}"
            except:
                image_path = f'img_{i}'
            
            if image_bytes and len(image_bytes) > 0:
                image = Image.open(io.BytesIO(image_bytes))
                inputs.append({'image': image, 'image_path': image_path, 'index': i})
            else:
                inputs.append({'image': None, 'image_path': image_path, 'index': i})
                
        except Exception as e:
            inputs.append({'image': None, 'image_path': f'failed_{i}', 'index': i})
    
    valid_inputs = [inp for inp in inputs if inp['image'] is not None]
    
    if not valid_inputs:
        return pd.Series([None] * batch_size)
    
    results = [None] * batch_size
    session = create_simple_session(config)
    
    try:
        # Simple fixed concurrency processing
        print(f"Processing {len(valid_inputs)} images with {config.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # Submit all work at once - much simpler
            future_to_input = {
                executor.submit(
                    make_simple_api_request,
                    inp['image'],
                    config,
                    session,
                    inp['image_path']
                ): inp for inp in valid_inputs
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_input):
                inp = future_to_input[future]
                try:
                    result = future.result()
                    results[inp['index']] = result
                    completed += 1
                    
                    if completed % 10 == 0:
                        print(f"Completed {completed}/{len(valid_inputs)} images")
                        
                except Exception as e:
                    print(f"Error processing {inp['image_path']}: {e}")
                    results[inp['index']] = None
        
        successful = sum(1 for r in results if r is not None)
        print(f"Batch completed: {successful}/{batch_size} successful")
        
        return pd.Series(results)
        
    except Exception as e:
        print(f"Error in VLM processing: {e}")
        return pd.Series([None] * batch_size)
    finally:
        session.close()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Execute Simple VLM Processing

# COMMAND ----------

print("=== STARTING SIMPLE VLM PROCESSING ===")
start_time = time.time()

pages_df = spark.table(SOURCE_TABLE_FULL)
pages_df = pages_df.repartition(8)  # Simple partitioning

# Apply simple VLM UDF
ocr_results_df = pages_df.withColumn(
    'ocr_text', 
    simple_vlm_udf(col('page_image_png'), col('metadata_json'))
)

ocr_results_df = ocr_results_df.withColumn(
    'ocr_timestamp', 
    lit(time.strftime("%Y-%m-%d %H:%M:%S"))
).withColumn(
    'ocr_model',
    lit(OPENAI_MODEL_NAME)
).withColumn(
    'processing_mode',
    lit('simple_fixed_concurrency')
)

end_time = time.time()
processing_time = end_time - start_time

print(f"✅ SIMPLE VLM PROCESSING COMPLETED in {processing_time:.1f}s")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Save Results

# COMMAND ----------

(ocr_results_df
 .write
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .partitionBy("doc_id")
 .saveAsTable(OUTPUT_TABLE_FULL)
)

total_pages = spark.table(OUTPUT_TABLE_FULL).count()
successful_ocr = spark.table(OUTPUT_TABLE_FULL).filter(col("ocr_text").isNotNull()).count()

print(f"🎯 RESULTS:")
print(f"  - Total pages: {total_pages:,}")
print(f"  - Successful: {successful_ocr:,}")
print(f"  - Success rate: {(successful_ocr/total_pages)*100:.1f}%")
print(f"  - Throughput: {total_pages/processing_time:.1f} pages/second")