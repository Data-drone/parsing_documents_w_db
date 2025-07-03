# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Scaling up parsing with Nanonets on an API

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

from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import StringType
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

catalog = 'brian_gen_ai'
schema = 'parsing_test'
split_pages_table = 'split_pages'

MODEL_NAME = 'nanonets/Nanonets-OCR-s'

# COMMAND ----------

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# suppress logging messages
logging.getLogger('py4j.clientserver').setLevel(logging.WARNING)

# COMMAND ----------

# Check Data

split_pages_dataframe = spark.table(f"{catalog}.{schema}.{split_pages_table}")
display(split_pages_dataframe)

# COMMAND ----------

class OpenAIAPIConfig:
    """Configuration for OpenAI-spec API"""
    def __init__(self):
        # Get config from environment variables or set defaults
        self.api_url = os.getenv("OPENAI_API_URL", "http://localhost:8000/v1")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "nanonets/Nanonets-OCR-s")
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
        self.timeout = int(os.getenv("OPENAI_TIMEOUT", "300"))  # 5 minutes
        self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
        self.max_workers = int(os.getenv("OPENAI_MAX_WORKERS", "5"))  # Concurrent requests
        self.rate_limit_delay = float(os.getenv("OPENAI_RATE_LIMIT_DELAY", "0.1"))  # Delay between requests

# COMMAND ----------

def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    # Convert to RGB if necessary (for PNG with transparency)
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image.save(buffer, format='JPEG', quality=85, optimize=True)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def get_image_mime_type(image: Image.Image) -> str:
    """Get MIME type for image"""
    # Since we're converting to JPEG, always return JPEG mime type
    return 'image/jpeg'

def make_api_request(image: Image.Image, config: OpenAIAPIConfig, image_path: str = "unknown") -> Optional[str]:
    """Make a single API request for OCR"""
    
    prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
    
    try:
        # Encode image
        base64_image = encode_image_to_base64(image)
        mime_type = get_image_mime_type(image)
        
        # Prepare request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}"
        }
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
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
        
        # Retry logic
        last_exception = None
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
                    #logger.warning(f"No valid response for image {image_path}")
                    return None
                    
            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < config.max_retries:
                    #logger.warning(f"Request timeout for {image_path}, retrying... (attempt {attempt + 1})")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    #logger.error(f"Final timeout for {image_path} after {config.max_retries + 1} attempts")
                    return None
                    
            except requests.exceptions.RequestException as e:
                #logger.error(f"API request failed for {image_path}: {str(e)}")
                return None
                
    except Exception as e:
        #logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

def prepare_api_inputs(page_bytes_list: List[bytes], metadata_list: List[str]) -> List[Dict]:
    """Prepare inputs for API processing"""
    inputs = []
    
    for i, (page_bytes, metadata_str) in enumerate(zip(page_bytes_list, metadata_list)):
        try:
            # Parse metadata
            try:
                metadata = ast.literal_eval(metadata_str)
                image_path = metadata.get('source_filepath', f'image_{i}')
            except:
                image_path = f'image_{i}'
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(page_bytes))
            
            inputs.append({
                'image': image,
                'image_path': image_path,
                'index': i
            })
            
        except Exception as e:
            #logger.error(f"Error preparing input {i}: {e}")
            inputs.append({
                'image': None,
                'image_path': f'failed_image_{i}',
                'index': i
            })
    
    return inputs

@pandas_udf(returnType=StringType())
def openai_ocr_pandas_udf(page_bytes_series: pd.Series, metadata_series: pd.Series) -> pd.Series:
    """
    High-performance pandas UDF using OpenAI-spec REST API for batch OCR processing
    """

    logger = logging.getLogger(__name__)

    os.environ['OPENAI_API_KEY'] = ""

    batch_size = len(page_bytes_series)
    logger.info(f"Processing OpenAI API batch of {batch_size} images")
    
    # Initialize API config
    config = OpenAIAPIConfig()
    
    if not config.api_key:
        config.api_key = ""
    #    logger.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    #    return pd.Series([None] * batch_size)
    
    # Prepare inputs
    inputs = prepare_api_inputs(
        page_bytes_list=page_bytes_series.tolist(),
        metadata_list=metadata_series.tolist()
    )

    logger.error("Input Check.")
    
    # Filter valid inputs
    valid_inputs = [inp for inp in inputs if inp['image'] is not None]
    
    if not valid_inputs:
        #logger.warning("No valid images to process in this batch")
        return pd.Series([None] * batch_size)
    
    # Initialize results array
    results = [None] * batch_size
    
    try:
        #logger.info(f"Running OpenAI API inference on {len(valid_inputs)} valid images with {config.max_workers} workers")
        
        # Process images concurrently
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # Submit all requests
            future_to_input = {}
            for inp in valid_inputs:
                # Add small delay to respect rate limits
                time.sleep(config.rate_limit_delay)
                future = executor.submit(make_api_request, inp['image'], config, inp['image_path'])
                future_to_input[future] = inp
            
            # Collect results
            for future in as_completed(future_to_input):
                inp = future_to_input[future]
                try:
                    result = future.result()
                    results[inp['index']] = result
                except Exception as e:
                    logger.error(f"Error processing image {inp['image_path']}: {e}")
                    results[inp['index']] = None
        
        successful_results = sum(1 for r in results if r is not None)
        #logger.info(f"OpenAI API batch processing completed: {successful_results}/{batch_size} successful")
        
        return pd.Series(results)
        
    except Exception as e:
        #logger.error(f"Error in OpenAI API batch processing: {e}")
        return pd.Series([None] * batch_size)

# COMMAND ----------

result_df = split_pages_dataframe \
                .withColumn('markdown_data', 
                            openai_ocr_pandas_udf(col('page_bytes'), col('metadata')))
                
display(result_df)