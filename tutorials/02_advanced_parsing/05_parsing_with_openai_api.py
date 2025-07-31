# Databricks notebook source
# MAGIC %md
# MAGIC # Querying OpenAI Spec LLM Rest Services
# MAGIC 
# MAGIC This notebook demonstrates how to query OpenAI Spec LLM Rest Services.
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
import logging
import requests
import base64
import pandas as pd
import ast
import threading
from typing import List, Dict, Optional
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from queue import Queue
from dataclasses import dataclass

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

# Performance and Concurrency Configuration
dbutils.widgets.text("min_workers", os.getenv("MIN_WORKERS", "2"), "Min Workers")
dbutils.widgets.text("max_workers", os.getenv("MAX_WORKERS", "16"), "Max Workers")
dbutils.widgets.text("initial_workers", os.getenv("INITIAL_WORKERS", "4"), "Initial Workers")
dbutils.widgets.text("base_delay", os.getenv("BASE_DELAY", "0.05"), "Base Delay (seconds)")

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

# Performance Configuration
MIN_WORKERS = int(dbutils.widgets.get("min_workers"))
MAX_WORKERS = int(dbutils.widgets.get("max_workers"))
INITIAL_WORKERS = int(dbutils.widgets.get("initial_workers"))
BASE_DELAY = float(dbutils.widgets.get("base_delay"))

# Construct full table names
SOURCE_TABLE_FULL = f"{CATALOG}.{SCHEMA}.{SOURCE_TABLE}"
OUTPUT_TABLE_FULL = f"{CATALOG}.{SCHEMA}.{OUTPUT_TABLE}"

print("=== OpenAI API OCR Processing Configuration ===")
print(f"Source Table: {SOURCE_TABLE_FULL}")
print(f"Output Table: {OUTPUT_TABLE_FULL}")
print(f"OpenAI API URL: {OPENAI_API_URL}")
print(f"OpenAI Model: {OPENAI_MODEL_NAME}")
print(f"Max Tokens: {OPENAI_MAX_TOKENS}")
print(f"Temperature: {OPENAI_TEMPERATURE}")
print(f"Worker Range: {MIN_WORKERS}-{MAX_WORKERS} (starting: {INITIAL_WORKERS})")
print(f"Base Delay: {BASE_DELAY}s")
print("=" * 50)

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
# MAGIC ## Adaptive Rate Limiting and Backpressure Detection

# COMMAND ----------

@dataclass
class ServerMetrics:
    """Track server performance metrics for adaptive rate limiting"""
    response_times: List[float]
    error_count: int
    success_count: int
    last_429_time: float
    current_concurrency: int
    lock: threading.Lock

class AdaptiveConcurrencyConfig:
    """Configuration with adaptive concurrency based on server performance"""
    def __init__(self, 
                 api_url=None, 
                 api_key=None, 
                 model_name=None, 
                 max_tokens=None, 
                 temperature=None,
                 min_workers=None,
                 max_workers=None,
                 initial_workers=None,
                 base_delay=None):
        
        # API Configuration - use passed values or fallback to environment variables
        self.api_url = api_url or os.getenv("OPENAI_API_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "nanonets/Nanonets-OCR-s")
        self.max_tokens = max_tokens or int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
        self.temperature = temperature or float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
        
        # Adaptive concurrency settings - use passed values or fallback to environment variables
        self.min_workers = min_workers or int(os.getenv("MIN_WORKERS", "2"))
        self.max_workers = max_workers or int(os.getenv("MAX_WORKERS", "16"))  # More conservative
        self.initial_workers = initial_workers or int(os.getenv("INITIAL_WORKERS", "4"))
        
        # Rate limiting and backpressure
        self.base_delay = base_delay or float(os.getenv("BASE_DELAY", "0.05"))  # 50ms base delay
        self.max_delay = float(os.getenv("MAX_DELAY", "2.0"))  # Max 2s delay
        self.backpressure_threshold = float(os.getenv("BACKPRESSURE_THRESHOLD", "5.0"))  # 5s response time
        
        # Queue management
        self.max_queue_size = int(os.getenv("MAX_QUEUE_SIZE", "100"))
        self.queue_timeout = int(os.getenv("QUEUE_TIMEOUT", "30"))
        
        # Timeout settings
        self.connect_timeout = int(os.getenv("CONNECT_TIMEOUT", "10"))
        self.read_timeout = int(os.getenv("READ_TIMEOUT", "60"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "2"))
        
        # Image optimization
        self.jpeg_quality = int(os.getenv("JPEG_QUALITY", "70"))
        self.max_image_size = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
        
        print(f"Adaptive config: {self.initial_workers}-{self.max_workers} workers, {self.base_delay}s base delay")

class AdaptiveRateLimiter:
    """Adaptive rate limiter with backpressure detection"""
    
    def __init__(self, config: AdaptiveConcurrencyConfig):
        self.config = config
        self.metrics = ServerMetrics(
            response_times=[],
            error_count=0,
            success_count=0,
            last_429_time=0,
            current_concurrency=config.initial_workers,
            lock=threading.Lock()
        )
        self.request_queue = Queue(maxsize=config.max_queue_size)
        self.last_adjustment = time.time()
        
    def should_reduce_concurrency(self) -> bool:
        """Check if we should reduce concurrency based on server metrics"""
        with self.metrics.lock:
            # Recent 429 errors
            if time.time() - self.metrics.last_429_time < 10:
                return True
            
            # High error rate
            total_requests = self.metrics.success_count + self.metrics.error_count
            if total_requests > 10:
                error_rate = self.metrics.error_count / total_requests
                if error_rate > 0.2:  # 20% error rate
                    return True
            
            # Slow response times
            if len(self.metrics.response_times) > 5:
                avg_response_time = sum(self.metrics.response_times[-10:]) / min(10, len(self.metrics.response_times))
                if avg_response_time > self.config.backpressure_threshold:
                    return True
                    
            return False
    
    def should_increase_concurrency(self) -> bool:
        """Check if we can safely increase concurrency"""
        with self.metrics.lock:
            # Only increase if we haven't seen recent issues
            if time.time() - self.metrics.last_429_time < 30:
                return False
            
            # Good response times and low error rate
            if len(self.metrics.response_times) > 10:
                avg_response_time = sum(self.metrics.response_times[-10:]) / min(10, len(self.metrics.response_times))
                total_requests = self.metrics.success_count + self.metrics.error_count
                error_rate = self.metrics.error_count / max(1, total_requests)
                
                return (avg_response_time < 2.0 and error_rate < 0.05)
            
            return False
    
    def adjust_concurrency(self) -> int:
        """Dynamically adjust concurrency based on server performance"""
        now = time.time()
        
        # Only adjust every 10 seconds
        if now - self.last_adjustment < 10:
            return self.metrics.current_concurrency
        
        old_concurrency = self.metrics.current_concurrency
        
        if self.should_reduce_concurrency():
            self.metrics.current_concurrency = max(
                self.config.min_workers,
                int(self.metrics.current_concurrency * 0.7)  # Reduce by 30%
            )
            print(f"🔽 Reducing concurrency: {old_concurrency} → {self.metrics.current_concurrency}")
            
        elif self.should_increase_concurrency():
            self.metrics.current_concurrency = min(
                self.config.max_workers,
                self.metrics.current_concurrency + 2  # Gradual increase
            )
            print(f"🔼 Increasing concurrency: {old_concurrency} → {self.metrics.current_concurrency}")
        
        self.last_adjustment = now
        return self.metrics.current_concurrency
    
    def record_response(self, response_time: float, success: bool, is_429: bool = False):
        """Record response metrics for adaptive rate limiting"""
        with self.metrics.lock:
            self.metrics.response_times.append(response_time)
            if len(self.metrics.response_times) > 50:  # Keep only recent responses
                self.metrics.response_times = self.metrics.response_times[-50:]
            
            if success:
                self.metrics.success_count += 1
            else:
                self.metrics.error_count += 1
                
            if is_429:
                self.metrics.last_429_time = time.time()
    
    def get_delay(self) -> float:
        """Calculate adaptive delay based on current server load"""
        base_delay = self.config.base_delay
        
        # Increase delay if we're seeing issues
        if self.should_reduce_concurrency():
            multiplier = min(10, self.metrics.error_count + 1)
            return min(self.config.max_delay, base_delay * multiplier)
        
        return base_delay

# COMMAND ----------
# MAGIC %md
# MAGIC ## Optimized Session and Request Functions

# COMMAND ----------

def create_adaptive_session(config: AdaptiveConcurrencyConfig) -> requests.Session:
    """Create session optimized for adaptive rate limiting"""
    session = requests.Session()
    
    # Conservative retry strategy
    retry_strategy = Retry(
        total=config.max_retries,
        status_forcelist=[500, 502, 503, 504],  # Don't retry 429 - handle separately
        backoff_factor=0.5,
        allowed_methods=["POST"]
    )
    
    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        pool_block=False,
        max_retries=retry_strategy
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    session.headers.update({
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
        "Connection": "keep-alive"
    })
    
    return session

def encode_image_optimized(image: Image.Image, config: AdaptiveConcurrencyConfig) -> str:
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

def make_adaptive_api_request(
    image: Image.Image,
    config: AdaptiveConcurrencyConfig,
    session: requests.Session,
    rate_limiter: AdaptiveRateLimiter,
    image_path: str = "unknown"
) -> Optional[str]:
    """API request with adaptive rate limiting and backpressure handling"""
    
    prompt = """Extract all text from this document. Format tables as HTML, equations as LaTeX. Use ☐/☑ for checkboxes."""
    
    start_time = time.time()
    
    try:
        # Get adaptive delay
        delay = rate_limiter.get_delay()
        if delay > 0:
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
            timeout=(config.connect_timeout, config.read_timeout)
        )
        
        response_time = time.time() - start_time
        
        # Handle rate limiting
        if response.status_code == 429:
            rate_limiter.record_response(response_time, False, is_429=True)
            # logger.warning(f"Rate limited for {image_path}, waiting...")
            time.sleep(5)  # Wait before retry
            return None
        
        response.raise_for_status()
        result = response.json()
        
        # Record successful response
        rate_limiter.record_response(response_time, True)
        
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        response_time = time.time() - start_time
        rate_limiter.record_response(response_time, False)
        # logger.error(f"Request failed for {image_path}: {str(e)}")
        return None
    except Exception as e:
        response_time = time.time() - start_time
        rate_limiter.record_response(response_time, False)
        # logger.error(f"Error processing {image_path}: {str(e)}")
        return None

# COMMAND ----------
# MAGIC %md
# MAGIC ## Adaptive Concurrency Pandas UDF

# COMMAND ----------

@pandas_udf(returnType=StringType())
def adaptive_ocr_udf(page_images: pd.Series, metadata_series: pd.Series) -> pd.Series:
    """
    OCR UDF with adaptive concurrency and backpressure control
    """
    
    batch_size = len(page_images)
    # logger.info(f"Processing adaptive OCR batch of {batch_size} images")
    
    config = AdaptiveConcurrencyConfig(
        api_url=OPENAI_API_URL,
        api_key=OPENAI_API_KEY,
        model_name=OPENAI_MODEL_NAME,
        max_tokens=OPENAI_MAX_TOKENS,
        temperature=OPENAI_TEMPERATURE,
        min_workers=MIN_WORKERS,
        max_workers=MAX_WORKERS,
        initial_workers=INITIAL_WORKERS,
        base_delay=BASE_DELAY
    )
    rate_limiter = AdaptiveRateLimiter(config)
    
    if not config.api_key:
        config.api_key = ""
    
    # Prepare inputs
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
            # logger.error(f"Error preparing input {i}: {e}")
            inputs.append({'image': None, 'image_path': f'failed_{i}', 'index': i})
    
    valid_inputs = [inp for inp in inputs if inp['image'] is not None]
    
    if not valid_inputs:
        return pd.Series([None] * batch_size)
    
    results = [None] * batch_size
    session = create_adaptive_session(config)
    
    try:
        # Start with initial concurrency
        current_workers = config.initial_workers
        # logger.info(f"Starting with {current_workers} workers for {len(valid_inputs)} images")
        
        # Process with adaptive concurrency
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = []
            submitted = 0
            
            # Submit initial batch
            for inp in valid_inputs[:current_workers]:
                future = executor.submit(
                    make_adaptive_api_request,
                    inp['image'],
                    config,
                    session,
                    rate_limiter,
                    inp['image_path']
                )
                futures.append((future, inp))
                submitted += 1
            
            # Process remaining with adaptive submission
            remaining_inputs = valid_inputs[current_workers:]
            completed = 0
            
            while futures or remaining_inputs:
                # Check for completed futures
                completed_futures = []
                for future, inp in futures:
                    if future.done():
                        completed_futures.append((future, inp))
                
                # Process completed futures
                for future, inp in completed_futures:
                    futures.remove((future, inp))
                    try:
                        result = future.result()
                        results[inp['index']] = result
                        completed += 1
                        
                        if completed % 5 == 0:
                            # logger.info(f"Completed {completed}/{len(valid_inputs)}")
                            pass
                            
                    except Exception as e:
                        # logger.error(f"Error getting result for {inp['image_path']}: {e}")
                        results[inp['index']] = None
                
                # Adjust concurrency and submit more if needed
                if remaining_inputs:
                    current_workers = rate_limiter.adjust_concurrency()
                    slots_available = current_workers - len(futures)
                    
                    # Submit more work if we have slots
                    for _ in range(min(slots_available, len(remaining_inputs))):
                        if remaining_inputs:
                            inp = remaining_inputs.pop(0)
                            future = executor.submit(
                                make_adaptive_api_request,
                                inp['image'],
                                config,
                                session,
                                rate_limiter,
                                inp['image_path']
                            )
                            futures.append((future, inp))
                
                # Small delay to prevent busy waiting
                if futures:
                    time.sleep(0.1)
            
            # Wait for any remaining futures
            for future, inp in futures:
                try:
                    result = future.result()
                    results[inp['index']] = result
                except Exception as e:
                    # logger.error(f"Final error for {inp['image_path']}: {e}")
                    results[inp['index']] = None
        
        successful = sum(1 for r in results if r is not None)
        # logger.info(f"Adaptive batch completed: {successful}/{batch_size} successful")
        
        return pd.Series(results)
        
    except Exception as e:
        # logger.error(f"Error in adaptive OCR processing: {e}")
        return pd.Series([None] * batch_size)
    finally:
        session.close()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Execute Adaptive OCR Processing

# COMMAND ----------

print("=== STARTING ADAPTIVE OCR PROCESSING ===")
start_time = time.time()

pages_df = spark.table(SOURCE_TABLE_FULL)
pages_df = pages_df.repartition(10)  # Conservative partitioning

# Apply adaptive OCR UDF
ocr_results_df = pages_df.withColumn(
    'ocr_text', 
    adaptive_ocr_udf(col('page_image_png'), col('metadata_json'))
)

ocr_results_df = ocr_results_df.withColumn(
    'ocr_timestamp', 
    lit(time.strftime("%Y-%m-%d %H:%M:%S"))
).withColumn(
    'ocr_model',
    lit(OPENAI_MODEL_NAME)
).withColumn(
    'processing_mode',
    lit('adaptive_concurrency')
)

end_time = time.time()
processing_time = end_time - start_time

print(f"✅ ADAPTIVE OCR COMPLETED in {processing_time:.1f}s")

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