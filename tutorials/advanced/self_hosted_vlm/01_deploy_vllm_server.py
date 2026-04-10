# Databricks notebook source
# MAGIC %md
# MAGIC # Custom VLM Server Setup
# MAGIC
# MAGIC This notebook sets up a VLM server using vLLM to serve dots.ocr for document parsing.
# MAGIC The server can then be queried by other notebooks via OpenAI-compatible API.
# MAGIC
# MAGIC ## Overview
# MAGIC - **Purpose**: Start a local vLLM server for self-hosted document OCR
# MAGIC - **Model**: dots.ocr (3B params, SOTA on OmniDocBench, 100+ languages, MIT license)
# MAGIC - **Benefits**: No per-call API fees, full control, GPU-local inference
# MAGIC
# MAGIC ## Requirements
# MAGIC - GPU cluster (A10 24GB+ recommended — model is ~6GB in bf16)
# MAGIC - DBR with CUDA 12.4+ (check `nvidia-smi` on your cluster)
# MAGIC
# MAGIC ## Alternative models (change via widget)
# MAGIC - `Qwen/Qwen3-VL-8B-Instruct` — general VLM, 256K context, 32 languages
# MAGIC - `tiiuae/Falcon-OCR` — 300M params, fast, great on tables
# MAGIC - `ibm-granite/granite-4.0-3b-vision` — enterprise KVP/table extraction
# MAGIC
# MAGIC ## Configuration
# MAGIC This notebook uses **Databricks widgets** for runtime configuration.

# COMMAND ----------

# MAGIC %pip install -U "vllm>=0.9.1" "transformers>=4.52.0" --quiet
# MAGIC %restart_python

# COMMAND ----------

import os

# COMMAND ----------
# MAGIC %md
# MAGIC ## Runtime Configuration with Widgets
# MAGIC
# MAGIC Configure the VLM server parameters using widgets.

# COMMAND ----------

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

dbutils.widgets.text("model_name", "rednote-hilab/dots.ocr", "Model Name")
dbutils.widgets.text("max_num_batched_tokens", "16384", "Max Batched Tokens")
dbutils.widgets.text("max_num_seqs", "16", "Max Number of Sequences")
dbutils.widgets.text("max_model_len", "16384", "Max Model Length")
dbutils.widgets.text("gpu_memory_utilization", "0.90", "GPU Memory Utilization")
dbutils.widgets.text("swap_space", "8", "Swap Space (GB)")
dbutils.widgets.text("server_port", "8000", "Server Port")

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
# MAGIC Launch the vLLM server with the configured parameters. The server exposes an
# MAGIC OpenAI-compatible API at `http://localhost:{port}/v1`.

# COMMAND ----------

vllm_command = f"""vllm serve {MODEL_NAME} \\
  --max-num-batched-tokens {MAX_BATCHED_TOKENS} \\
  --max-num-seqs {MAX_NUM_SEQS} \\
  --max-model-len {MAX_MODEL_LEN} \\
  --limit-mm-per-prompt "image=1,video=0" \\
  --gpu-memory-utilization {GPU_MEMORY_UTIL} \\
  --enable-chunked-prefill \\
  --kv-cache-dtype auto \\
  --swap-space {SWAP_SPACE} \\
  --port {SERVER_PORT} \\
  --served-model-name "{MODEL_NAME}" \\
  --async-scheduling"""

print("Starting VLM Server with command:")
print(vllm_command)
print("\nServer will start in the next cell...")

# COMMAND ----------

!vllm serve {MODEL_NAME} \
  --max-num-batched-tokens {MAX_BATCHED_TOKENS} \
  --max-num-seqs {MAX_NUM_SEQS} \
  --max-model-len {MAX_MODEL_LEN} \
  --limit-mm-per-prompt "image=1,video=0" \
  --gpu-memory-utilization {GPU_MEMORY_UTIL} \
  --enable-chunked-prefill \
  --kv-cache-dtype auto \
  --swap-space {SWAP_SPACE} \
  --port {SERVER_PORT} \
  --served-model-name "{MODEL_NAME}" \
  --async-scheduling