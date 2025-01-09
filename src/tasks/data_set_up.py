# Databricks notebook source
# MAGIC %md
# MAGIC # Setup the data
# MAGIC
# MAGIC This notebook is focused upon some set-up for the remaining worklow.
# MAGIC
# MAGIC TODO:
# MAGIC - Explain the volume creation / docs we're using

# COMMAND ----------
import os
import shutil
import sys

sys.path.append("..")

#Some simple helper functions
from utils.databricks_utils import get_username_from_email

# COMMAND ----------

# DBTITLE 1,Set Up Configurations

user_name = get_username_from_email(dbutils = dbutils)
catalog = f"{user_name}_parsing"
schema = dbutils.widgets.get("schema_name")
volume = dbutils.widgets.get("volume_name")
docs_file_path = dbutils.widgets.get("file_path")

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
shutil.copytree(f"/Workspace/{docs_file_path}", f'{volume_path}/', dirs_exist_ok=True)