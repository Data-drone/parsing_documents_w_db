# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Scaling up Image-to-Markdown Parsing with Nanonets OCR Model using vLLM + Ray Data
# MAGIC
# MAGIC This notebook demonstrates how to:
# MAGIC - Read image data from a parameterized Spark Delta table
# MAGIC - Process base64 encoded images using vLLM + Ray Data with native Spark integration
# MAGIC - Generate markdown output using the Nanonets OCR model
# MAGIC - Scale efficiently across multiple GPUs using Ray Data's distributed processing
# MAGIC - Access Ray Dashboard for monitoring and debugging
# MAGIC - Handle preprocessing errors gracefully to prevent ChatTemplate failures
# MAGIC - **Fixed**: Binary PNG data serialization issues with Arrow
# MAGIC - **Fixed**: vLLM parameter conflicts with Ray
# MAGIC - **Fixed**: Multiprocessing signal handling conflicts

# COMMAND ----------

# MAGIC #%pip install -U vllm==0.8.0 transformers==4.52.1 ray[data]==2.47.1 pillow==10.0.0 --quiet
# MAGIC #%restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration and Environment Setup

# COMMAND ----------

# vLLM initialization config - FIXED for Ray compatibility
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "1"  # Enable vLLM v1 for better vision model support
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

# Configuration - Parameterized for easy adjustment
CATALOG = 'brian_gen_ai'
SCHEMA = 'parsing_test'
SOURCE_TABLE = 'document_page_docs'

TEMP_VOLUME = 'ray_temp'
OUTPUT_TABLE = 'parsed_markdown_pages'
MODEL_NAME = 'nanonets/Nanonets-OCR-s'

os.environ['RAY_UC_VOLUMES_FUSE_TEMP_DIR'] = f'/Volumes/{CATALOG}/{SCHEMA}/{TEMP_VOLUME}'

testing = False

# Full table paths
source_table_path = f"{CATALOG}.{SCHEMA}.{SOURCE_TABLE}"
output_table_path = f"{CATALOG}.{SCHEMA}.{OUTPUT_TABLE}"

print(f"Processing table: {source_table_path}")
print(f"Output table: {output_table_path}")
print(f"Using model: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Required Libraries

# COMMAND ----------

import ray
import base64
import io
from PIL import Image
import pandas as pd
from typing import Dict, Any, List
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, max as spark_max

# Ray Data and vLLM imports
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
import ray.data

# Initialize Ray with dashboard configuration
def get_dashboard_url(spark, dbutils, dashboard_port='9999'):
    base_url = 'https://' + spark.conf.get("spark.databricks.workspaceUrl")
    workspace_id = spark.conf.get("spark.databricks.clusterUsageTags.orgId")
    cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
    pathname_prefix = '/driver-proxy/o/' + workspace_id + '/' + cluster_id + '/' + dashboard_port + "/"
    apitoken = dbutils.notebook().entry_point.getDbutils().notebook().getContext().apiToken().get()
    dashboard_url = base_url + pathname_prefix
    return dashboard_url

# Initialize Ray if not already running
# if not ray.is_initialized():
#     context = ray.init(
#         include_dashboard=True,
#         dashboard_host="0.0.0.0",
#         dashboard_port=9999
#     )

# print("Ray initialized successfully")
# print(f"Ray cluster resources: {ray.cluster_resources()}")

# # Get and display dashboard URL
# dashboard_url = get_dashboard_url(spark, dbutils)
# print(f"Ray Dashboard URL: {dashboard_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Exploration and Loading

# COMMAND ----------

# Read the Delta table using Spark
print(f"Reading data from: {source_table_path}")
df = spark.read.table(source_table_path)

# Display basic information about the dataset
print("Schema:")
df.printSchema()

print("\nDataset statistics:")
df.select(
    count("*").alias("total_rows"),
    spark_max("page_number").alias("max_page_number"),
    count("doc_id").alias("total_docs")
).show()

# Sample a few rows to understand the data structure
print("\nSample data (excluding binary column):")
df.select("doc_id", "source_filename", "page_number", "total_pages", "file_size_bytes").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert Spark DataFrame to Ray Dataset with Binary PNG Fix

# COMMAND ----------

import ray.data

# Convert Spark DataFrame to Ray Dataset (basic method)
if testing:
    sparkset = df.limit(10)
else: 
    sparkset = df
    
ray_dataset = ray.data.from_spark(sparkset).map_batches(
        lambda pdf: pdf.assign(
            page_image_png=pdf["page_image_png"].apply(bytes)  # memoryview → bytes
        ),
        batch_format="pandas")


print("Ray Dataset created successfully")
print(f"Dataset schema: {ray_dataset.schema()}")
print(f"Dataset count: {ray_dataset.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure vLLM Engine for Nanonets OCR - FIXED

# COMMAND ----------


vllm_config = vLLMEngineProcessorConfig(
        model_source=MODEL_NAME,
        engine_kwargs={
            "max_model_len": 8192,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 24576,  # Optimized for better GPU utilization
            "max_num_seqs": 16,               # Increased for better batching
            "gpu_memory_utilization": 0.85,   # Higher GPU utilization
            "kv_cache_dtype": "auto",
            "limit_mm_per_prompt": {"image": 1},
        },
        runtime_env={
            "env_vars": {
                "VLLM_USE_V1": "1",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            }
        },
        concurrency=1,                       # CRITICAL: Keep at 1 to prevent Ray+vLLM conflicts
        batch_size=2,                       # Increased batch size for better GPU utilization
        #accelerator_type="GPU",
        has_image=True,
    )

print("✅ vLLM configuration created successfully with Ray compatibility fixes")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Image Processing and vLLM Preprocessing Functions - ENHANCED

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build vLLM Processor and Run Inference

# COMMAND ----------

# Build the vLLM processor with enhanced error handling
# Build and run vLLM processor

# TODO Adjust this here to read the right columns and output the right columns

def preprocess(row):
    from PIL import Image, ImageFile
    import io
    img = Image.open(io.BytesIO(row["page_image_png"]))
    return dict(
        # Ray Data expects OpenAI-style chat messages
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": "Extract all text from this document. Format tables as HTML, equations as LaTeX. Use ☐/☑ for checkboxes."}
            ],
        }],
        sampling_params=dict(max_tokens=128),
        id=row["doc_id"],              # keep primary key for join-back
    )
    
def postprocess(row):
    return {
        "doc_id": row["id"],
        "page_number": row["page_number"],
        "caption": row["generated_text"],       # or whatever you need
        "error": row.get("error")
    }   
    

vllm_processor = build_llm_processor(
    config=vllm_config,
    preprocess=preprocess,
    postprocess=postprocess
    )


processed_dataset = vllm_processor(ray_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Collect and Display Results

# COMMAND ----------

# Preview Results
if testing:
    results = processed_dataset.take_all()
    print(f"Processed {len(results)} images")

    for i, result in enumerate(results[:3]):
        print(f"\n--- Result {i+1} ---")
        for key, value in result.items():
            print(f"{key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert Results Back to Delta Table

# COMMAND ----------

out_ds = processed_dataset.materialize()
spark.sql(f"DROP TABLE IF EXISTS {output_table_path}_temp")
out_ds.write_databricks_table(name=f'{output_table_path}_temp')

# COMMAND ----------

# Merge datasets
results_temp = spark.table(f'{output_table_path}_temp')

merged_output = df.alias("orig") \
            .join(results_temp.alias("res"), on="doc_id", how="left")

merged_output.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(output_table_path)
