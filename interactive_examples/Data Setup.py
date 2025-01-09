# Databricks notebook source
# MAGIC %md
# MAGIC # Setup the data
# MAGIC
# MAGIC This notebook is focused upon some set-up for the remaining worklow.
# MAGIC
# MAGIC TODO:
# MAGIC - Explain the volume creation / docs we're using

# COMMAND ----------

import shutil

#Some simple helper functions
from src.utils.databricks_utils import get_username_from_email

# COMMAND ----------

# DBTITLE 1,Set Up Configurations
import os

user_name = get_username_from_email(dbutils = dbutils)
catalog = f"{user_name}_parsing"
schema = 'parsing_tests'
volume = 'raw_data'

# COMMAND ----------

# DBTITLE 1,Create the folders if needed
spark.sql(f'CREATE CATALOG IF NOT EXISTS {catalog}')
spark.sql(f'CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}')
spark.sql(f'CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}')

# COMMAND ----------

# DBTITLE 1,Generate Volume Path for pasting to next cell
volume_path = os.path.join('/Volumes', catalog, schema, volume)
volume_path

# COMMAND ----------

# DBTITLE 1,Copy samples into volume - doesn't work on serverless
volume_path = os.path.join('/Volumes', catalog, schema, volume)

#otherPath = '/Volumes/brian_ml_dev/parsing_tests/raw_data'
shutil.copytree('./docs/', f'{volume_path}/', dirs_exist_ok=True)

# COMMAND ----------

# DBTITLE 1,Copy samples into volume if serverless
# MAGIC %sh
# MAGIC cp docs/* /Volumes/brian_ml_dev/parsing_tests/raw_data
