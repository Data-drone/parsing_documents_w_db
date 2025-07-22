# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Scaling up parsing with Nanonets Model
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %pip install -U vllm==0.8.0 transformers==4.52.1 mlflow pillow rich==14.0.0 --quiet
# MAGIC %restart_python

# COMMAND ----------

# vllm init config
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Parsing Config
catalog = 'brian_gen_ai'
schema = 'parsing_test'
split_pages_table = 'split_pages'

MODEL_NAME = 'nanonets/Nanonets-OCR-s'
max_new_tokens = 1028*10

split_pages_dataframe = spark.table(f"{catalog}.{schema}.{split_pages_table}")
display(split_pages_dataframe)

# COMMAND ----------

import pandas as pd

import io
import ast
from typing import List, Optional
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from vllm import LLM, SamplingParams

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# suppress logging messages
logging.getLogger('py4j.clientserver').setLevel(logging.WARNING)

# COMMAND ----------

# Model Singleton for efficient spark loading
class vLLMOCRSingleton:
    """Singleton for vLLM model - initialized once per executor"""
    _instance = None
    _llm = None
    _sampling_params = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(vLLMOCRSingleton, cls).__new__(cls)
        return cls._instance

    def get_model(self):
        if self._llm is None:
            self._load_model()
        return self._llm, self._sampling_params

    def _load_model(self):
        """Load vLLM model with optimal configuration"""
        try:
            logger.info(f"Loading {MODEL_NAME} with vLLM...")
            
            # vLLM configuration optimized for vision-language models
            self._llm = LLM(
                model=MODEL_NAME,
                # Memory and performance optimizations
                gpu_memory_utilization=0.85,  # Use 85% of GPU memory
                trust_remote_code=True,       # Required for custom models
                # Batch processing optimizations
            )
            
            # Sampling parameters optimized for OCR
            self._sampling_params = SamplingParams(
                temperature=0.0,           # Deterministic for OCR
            )
            
            logger.info("✅ vLLM model loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error loading vLLM model: {e}")
            raise

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import StringType
import pyspark.sql.functions as F


def prepare_vllm_inputs(page_bytes_list: List[bytes], metadata_list: List[str]) -> List[dict]:
    """Prepare inputs for vLLM batch processing"""
    
    prompt_template = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
    
    inputs = []
    
    for page_bytes, metadata_str in zip(page_bytes_list, metadata_list):
        try:
            # Parse metadata
            try:
                metadata = ast.literal_eval(metadata_str)
                image_path = metadata.get('source_filepath', 'unknown_path')
            except:
                image_path = 'unknown_path'
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(page_bytes))
            
            # Prepare input for vLLM (multimodal format)
            input_data = {
                "prompt": f"<image>\n{prompt_template}",
                "multi_modal_data": {
                    "image": image
                }
            }
            
            inputs.append(input_data)
            
        except Exception as e:
            logger.error(f"Error preparing input: {e}")
            # Add placeholder for failed input
            inputs.append({
                "prompt": f"ERROR: Failed to process image - {str(e)}",
                "multi_modal_data": {"image": None}
            })
    
    return inputs

@pandas_udf(returnType=StringType())
def vllm_ocr_pandas_udf(page_bytes_series: pd.Series, metadata_series: pd.Series) -> pd.Series:
    """
    Ultra-high performance pandas UDF using vLLM for batch OCR processing
    """
    batch_size = len(page_bytes_series)
    logger.info(f"Processing vLLM batch of {batch_size} images")
    
    # Force vllm spawn
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    # Get vLLM model
    model_singleton = vLLMOCRSingleton()
    llm, sampling_params = model_singleton.get_model()
    
    # Prepare inputs for vLLM
    inputs = prepare_vllm_inputs(
        page_bytes_list=page_bytes_series.tolist(),
        metadata_list=metadata_series.tolist()
    )
    
    # Filter out failed inputs
    valid_inputs = [inp for inp in inputs if inp["multi_modal_data"]["image"] is not None]
    
    if not valid_inputs:
        logger.warning("No valid images to process in this batch")
        return pd.Series([None] * batch_size)
    
    try:
        logger.info(f"Running vLLM inference on {len(valid_inputs)} valid images")
        
        # THIS IS THE MAGIC: vLLM processes the entire batch efficiently
        # with continuous batching, PagedAttention, and optimized CUDA kernels
        outputs = llm.generate(valid_inputs, sampling_params)
        
        # Extract results
        results = []
        valid_idx = 0
        
        for i, original_input in enumerate(inputs):
            if original_input["multi_modal_data"]["image"] is not None:
                # Valid input - get result
                output_text = outputs[valid_idx].outputs # [0].text
                results.append(output_text)
                valid_idx += 1
            else:
                # Failed input - add None
                results.append(None)
        
        logger.info(f"vLLM batch processing completed successfully")
        return pd.Series(results)
        
    except Exception as e:
        logger.error(f"Error in vLLM processing: {e}")
        return pd.Series([None] * batch_size)

# COMMAND ----------

### Testing result
# pandas_frame = split_pages_dataframe.limit(5).toPandas()

# result_df = run_inference(pandas_frame)

# display(result_df)

# COMMAND ----------

# Testing PySpark pandas udf
result_df = split_pages_dataframe \
                .withColumn('markdown_data', 
                            vllm_ocr_pandas_udf(F.col('page_bytes'), F.col('metadata')))
                
display(result_df)

# COMMAND ----------

result_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.inference_table")