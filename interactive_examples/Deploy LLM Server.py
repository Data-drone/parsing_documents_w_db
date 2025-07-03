# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Scaling up parsing with Nanonets Model
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %pip install -U vllm==0.8.0 transformers==4.52.1 --quiet
# MAGIC %restart_python

# COMMAND ----------

# vllm init config
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Parsing Config
# catalog = 'brian_gen_ai'
# schema = 'parsing_test'
# split_pages_table = 'split_pages'

MODEL_NAME = 'nanonets/Nanonets-OCR-s'

# COMMAND ----------

!vllm serve nanonets/Nanonets-OCR-s 