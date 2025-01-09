# Databricks notebook source
# MAGIC %md
# MAGIC # Lets Parse Docs
# MAGIC
# MAGIC Lets look at how to parse and work with docs

# COMMAND ----------

# MAGIC %pip install -U pymupdf4llm langchain==0.2.17 langchain_databricks==0.1.1
# MAGIC %restart_python

# COMMAND ----------

#Some simple helper functions
from src.utils.databricks_utils import get_username_from_email

# COMMAND ----------

# DBTITLE 1,Set Up Configurations
import os

user_name = get_username_from_email(dbutils = dbutils)
catalog = f"{user_name}_parsing"
schema = 'parsing_tests'
volume = 'raw_data'

#llm_model_name = 'brian_serving_test' #TODO
llm_model_name = 'databricks-meta-llama-3-1-70b-instruct'

full_vol_path = f'/Volumes/{catalog}/{schema}/{volume}'

example_files = [f for f in os.listdir(f'{full_vol_path}')  if f.endswith('.pdf')]
example_files

# COMMAND ----------

legalese = f'{full_vol_path}/{example_files[0]}'
annual_report = f'{full_vol_path}/{example_files[1]}'
pds = f'{full_vol_path}/{example_files[2]}'

# COMMAND ----------

# DBTITLE 1,Setup the Langchain Testing
from langchain_databricks import ChatDatabricks
from langchain_core.messages import HumanMessage
import pymupdf4llm

model = ChatDatabricks(
  target_uri='databricks',
  endpoint=llm_model_name,
  temperature=0.1
)

model.invoke([HumanMessage(content='Harro')])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Doc 1 - Legalese

# COMMAND ----------

md_text = pymupdf4llm.to_markdown(legalese, pages=range(13,53))
md_text

# COMMAND ----------

len(md_text)

# COMMAND ----------

prompt = f"""You are a legal analysis bot. Your job is to colloquially explain corporate legalese texts

Use the following text snippet as context
{md_text}

What are current weekly earnings? 
"""



response = model.invoke([HumanMessage(content=prompt)])
print(response.content)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Doc 2 - Annual Report

# COMMAND ----------

annual_report_text = pymupdf4llm.to_markdown(annual_report, pages=range(0,15))
annual_report_text

# COMMAND ----------

len(annual_report_text)

# COMMAND ----------

prompt = f"""You are a financial and mining productivity bot. Your job is to colloquially explain business results based on annual report data

Use the following report as context
{annual_report_text}

What are the 5 ways that Rio's business model delivers value 
"""



response = model.invoke([HumanMessage(content=prompt)])
print(response.content)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Doc 3 - PDS

# COMMAND ----------

pds_text = pymupdf4llm.to_markdown(pds)
pds_text

# COMMAND ----------

prompt = f"""You are an insurance claims agent

Use the following report as context
{pds_text}

Answer the following question, make sure to recheck any numbers that you extract

Under what conditions would I not be covered for fire damange?
"""



response = model.invoke([HumanMessage(content=prompt)])
print(response.content)

# COMMAND ----------


