# Databricks notebook source
# MAGIC %md
# MAGIC # Custom VLM Server Setup
# MAGIC
# MAGIC This notebook sets up a custom Vision Language Model (VLM) server using vLLM to serve the Nanonets OCR model. This server can then be queried by other notebooks for distributed document processing.
# MAGIC
# MAGIC ## Overview
# MAGIC - **Purpose**: Start a local vLLM server for custom VLM processing
# MAGIC - **Model**: Nanonets OCR model optimized for document text extraction
# MAGIC - **Benefits**: Local control, custom parameters, optimized for your use case
# MAGIC
# MAGIC ## Configuration
# MAGIC This notebook uses **Databricks widgets** for runtime configuration, following the same pattern as our other tutorials.

# COMMAND ----------

# MAGIC %pip install -U vllm==0.8.0 transformers==4.52.1 python-dotenv --quiet
# MAGIC %restart_python

# COMMAND ----------

# Load environment variables from .env (if present) **before** we read them via os.getenv
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # returns True if a .env is found and parsed

import os

# COMMAND ----------
# MAGIC %md
# MAGIC ## Runtime Configuration with Widgets
# MAGIC 
# MAGIC Configure the VLM server parameters using widgets that default to environment variables.

# COMMAND ----------

# vLLM initialization config - FIXED for better compatibility
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Create Databricks widgets for VLM server configuration
dbutils.widgets.text("model_name", os.getenv("VLM_MODEL_NAME", "nanonets/Nanonets-OCR-s"), "Model Name")
dbutils.widgets.text("max_num_batched_tokens", os.getenv("VLM_MAX_BATCHED_TOKENS", "16384"), "Max Batched Tokens")
dbutils.widgets.text("max_num_seqs", os.getenv("VLM_MAX_NUM_SEQS", "16"), "Max Number of Sequences")
dbutils.widgets.text("max_model_len", os.getenv("VLM_MAX_MODEL_LEN", "8192"), "Max Model Length")
dbutils.widgets.text("gpu_memory_utilization", os.getenv("VLM_GPU_MEMORY_UTIL", "0.85"), "GPU Memory Utilization")
dbutils.widgets.text("swap_space", os.getenv("VLM_SWAP_SPACE", "8"), "Swap Space (GB)")
dbutils.widgets.text("server_port", os.getenv("VLM_SERVER_PORT", "8000"), "Server Port")

# Read values from widgets
MODEL_NAME = dbutils.widgets.get("model_name")
MAX_BATCHED_TOKENS = int(dbutils.widgets.get("max_num_batched_tokens"))
MAX_NUM_SEQS = int(dbutils.widgets.get("max_num_seqs"))
MAX_MODEL_LEN = int(dbutils.widgets.get("max_model_len"))
GPU_MEMORY_UTIL = float(dbutils.widgets.get("gpu_memory_utilization"))
SWAP_SPACE = int(dbutils.widgets.get("swap_space"))
SERVER_PORT = int(dbutils.widgets.get("server_port"))

print("=== Custom VLM Server Configuration ===")
print(f"Model: {MODEL_NAME}")
print(f"Max Batched Tokens: {MAX_BATCHED_TOKENS:,}")
print(f"Max Sequences: {MAX_NUM_SEQS}")
print(f"Max Model Length: {MAX_MODEL_LEN:,}")
print(f"GPU Memory Utilization: {GPU_MEMORY_UTIL:.1%}")
print(f"Swap Space: {SWAP_SPACE} GB")
print(f"Server Port: {SERVER_PORT}")
print("=" * 45)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Start VLM Server
# MAGIC 
# MAGIC Launch the vLLM server with the configured parameters. This server will be available for other notebooks to query.

# COMMAND ----------

# Construct the vLLM serve command with configured parameters
vllm_command = f"""vllm serve {MODEL_NAME} \\
  --max-num-batched-tokens {MAX_BATCHED_TOKENS} \\
  --max-num-seqs {MAX_NUM_SEQS} \\
  --max-model-len {MAX_MODEL_LEN} \\
  --limit-mm-per-prompt "image=1" \\
  --gpu-memory-utilization {GPU_MEMORY_UTIL} \\
  --enable-chunked-prefill \\
  --kv-cache-dtype auto \\
  --swap-space {SWAP_SPACE} \\
  --port {SERVER_PORT} \\
  --served-model-name "{MODEL_NAME}" """

print("🚀 Starting VLM Server with command:")
print(vllm_command)
print("\n⏳ Server will start in the next cell...")

# COMMAND ----------

!vllm serve {MODEL_NAME} \
  --max-num-batched-tokens {MAX_BATCHED_TOKENS} \
  --max-num-seqs {MAX_NUM_SEQS} \
  --max-model-len {MAX_MODEL_LEN} \
  --limit-mm-per-prompt "image=1" \
  --gpu-memory-utilization {GPU_MEMORY_UTIL} \
  --enable-chunked-prefill \
  --kv-cache-dtype auto \
  --swap-space {SWAP_SPACE} \
  --port {SERVER_PORT} \
  --served-model-name "{MODEL_NAME}"