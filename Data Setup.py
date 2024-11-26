# Databricks notebook source
# MAGIC %md
# MAGIC # Setup the data
# MAGIC
# MAGIC Lets look at how to parse and work with docs

# COMMAND ----------

# DBTITLE 1,Set Up Configurations
import os

catalog = 'brian_ml_dev'
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