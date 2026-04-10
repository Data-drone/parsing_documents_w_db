# Databricks notebook source
# MAGIC %md
# MAGIC # Simple Remote VLM Server Processing
# MAGIC 
# MAGIC This notebook demonstrates a simplified approach to querying remote Vision Language Model (VLM) servers that are compatible with the OpenAI API specification.
# MAGIC 
# MAGIC ## Key Features for Remote Servers
# MAGIC - **Simple rate limiting**: Basic exponential backoff for 429 errors
# MAGIC - **Fixed concurrency**: Configurable worker count without complex adaptation
# MAGIC - **Retry logic**: Automatic retries for failed requests with backoff
# MAGIC - **Timeout handling**: Proper connection and read timeouts
# MAGIC - **Image optimization**: Efficient image encoding for network transmission
# MAGIC 
# MAGIC ## Simplified from Original
# MAGIC - Removed complex adaptive concurrency management
# MAGIC - Removed detailed metrics tracking and server performance analysis
# MAGIC - Removed dynamic worker adjustment based on response times
# MAGIC - Simplified configuration to essential parameters only
# MAGIC 
# MAGIC ## Setup
# MAGIC 
# MAGIC 1. Install required packages (including python-dotenv for environment config)
# MAGIC 2. Set up the OpenAI API key and configuration

# COMMAND ----------

%pip install python-dotenv Pillow
%restart_python

# COMMAND ----------

# Load environment variables from .env (if present) **before** we read them via os.getenv
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # returns True if a .env is found and parsed

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
# MAGIC 
# MAGIC Following the same pattern as our foundation tutorials, we use Databricks widgets to allow runtime configuration while defaulting to environment variables. This provides flexibility for different users and environments.

# COMMAND ----------

# -----------------------------------------------------------------------------
# Runtime configuration via widgets
# -----------------------------------------------------------------------------
# Pull defaults from environment variables (populated via `.env` or workspace)

# Create Databricks widgets for Unity Catalog configuration
dbutils.widgets.text("catalog_name", os.getenv("CATALOG_NAME", "brian_gen_ai"), "Catalog Name")
dbutils.widgets.text("schema_name", os.getenv("SCHEMA_NAME", "parsing_test"), "Schema Name")
dbutils.widgets.text("source_table", os.getenv("SOURCE_TABLE", "document_page_docs"), "Source Table Name")
dbutils.widgets.text("output_table", os.getenv("OUTPUT_TABLE", "document_store_ocr"), "Output Table Name")

# OpenAI API Configuration
dbutils.widgets.text("openai_api_url", os.getenv("OPENAI_API_URL", "http://localhost:8000/v1"), "OpenAI API URL")
dbutils.widgets.password("openai_api_key", "openai_api_key", os.getenv("OPENAI_API_KEY", ""), "OpenAI API Key")
dbutils.widgets.text("openai_model_name", os.getenv("OPENAI_MODEL_NAME", "nanonets/Nanonets-OCR-s"), "OpenAI Model Name")
dbutils.widgets.text("openai_max_tokens", os.getenv("OPENAI_MAX_TOKENS", "4096"), "Max Tokens")
dbutils.widgets.text("openai_temperature", os.getenv("OPENAI_TEMPERATURE", "0.0"), "Temperature")

# Simple Concurrency Configuration
dbutils.widgets.text("max_workers", os.getenv("MAX_WORKERS", "8"), "Max Workers")

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
# MAGIC ### 🔍 Configuration Verification
# MAGIC 
# MAGIC Let's verify that our configuration is correct and the source table contains processable page images.

# COMMAND ----------

# Verify source table exists and contains page images
try:
    # Check if source table exists and get basic stats
    table_info = spark.sql(f"DESCRIBE TABLE EXTENDED {SOURCE_TABLE_FULL}")
    print("✅ Source table exists")
    
    # Get page image counts and metadata
    page_stats = spark.sql(f"""
        SELECT 
            COUNT(*) as total_pages,
            COUNT(CASE WHEN page_image_png IS NOT NULL THEN 1 END) as pages_with_images,
            COUNT(DISTINCT doc_id) as unique_documents,
            ROUND(AVG(LENGTH(page_image_png))/1024/1024, 2) as avg_image_size_mb
        FROM {SOURCE_TABLE_FULL}
        WHERE page_image_png IS NOT NULL
    """).collect()[0]
    
    print(f"📊 Source Data Statistics:")
    print(f"   📄 Total pages: {page_stats.total_pages:,}")
    print(f"   🖼️ Pages with images: {page_stats.pages_with_images:,}")
    print(f"   📚 Unique documents: {page_stats.unique_documents:,}")
    print(f"   💾 Average image size: {page_stats.avg_image_size_mb:.2f} MB")
    
    if page_stats.pages_with_images == 0:
        print("⚠️ No pages with images found. Please ensure page images are available in the source table.")
        print("   This notebook requires the output from '01_split_to_images.py' tutorial.")
    else:
        # Show sample of first few documents for verification
        sample_docs = spark.sql(f"""
            SELECT doc_id, COUNT(*) as page_count
            FROM {SOURCE_TABLE_FULL}
            WHERE page_image_png IS NOT NULL
            GROUP BY doc_id
            ORDER BY doc_id
            LIMIT 3
        """).collect()
        
        print(f"\n🎯 Sample documents to be processed:")
        for doc in sample_docs:
            print(f"   📑 {doc.doc_id}: {doc.page_count} pages")
            
    # Check API configuration
    if not OPENAI_API_KEY:
        print("⚠️ OpenAI API key is not configured. Please set the OPENAI_API_KEY widget or environment variable.")
    else:
        print("✅ OpenAI API key is configured")
        
    print(f"🌐 API Endpoint: {OPENAI_API_URL}")
    print(f"🤖 Model: {OPENAI_MODEL_NAME}")
    
except Exception as e:
    print(f"❌ Error accessing source table: {e}")
    print("Please check your catalog, schema, and table configuration.")
    print("Make sure you've run the '01_split_to_images.py' tutorial first to create page images.")

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
        self.api_url = api_url or os.getenv("OPENAI_API_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "nanonets/Nanonets-OCR-s")
        self.max_tokens = max_tokens or int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
        self.temperature = temperature or float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
        
        # Simple concurrency and retry settings
        self.max_workers = max_workers or int(os.getenv("MAX_WORKERS", "8"))
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