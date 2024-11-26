# Databricks notebook source
# MAGIC %md
# MAGIC # Lets Parse Docs w VLM
# MAGIC
# MAGIC Lets look at how to parse and work with docs

# COMMAND ----------

# MAGIC %pip install -U pillow langchain==0.2.17 langchain_databricks==0.1.1
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Set Up Configurations
import os

catalog = 'brian_ml_dev'
schema = 'parsing_tests'
volume = 'raw_data'

llm_model_name = 'brian_serving_test'
llm_model_name = 'databricks-meta-llama-3-1-70b-instruct'

full_vol_path = f'/Volumes/{catalog}/{schema}/{volume}'

example_files = os.listdir(full_vol_path)
example_files

# COMMAND ----------

legalese = f'{full_vol_path}/{example_files[0]}'
annual_report = f'{full_vol_path}/{example_files[1]}'
pds = f'{full_vol_path}/{example_files[2]}'

# COMMAND ----------

# DBTITLE 1,Setup the Langchain Testing
from langchain_databricks import ChatDatabricks
from langchain_core.messages import HumanMessage
from openai import OpenAI

chat_model = ChatDatabricks(
  target_uri='databricks',
  endpoint=llm_model_name,
  temperature=0.1
)

chat_model.invoke([HumanMessage(content='Harro')])

vlm_model = ChatDatabricks(
  target_uri='databricks',
  endpoint='brian_serving_test',
  temperature=0.1
)

vlm_model.invoke([HumanMessage(content='can you digest imges?')])

# COMMAND ----------

# DBTITLE 1,Helper Functions
import base64
from io import BytesIO
from PIL import Image

def encode_ppm_image(ppm_file_path):
    # Open PPM file using PIL and convert to PNG format in memory
    with Image.open(ppm_file_path) as img:
        # Convert image to PNG format in memory
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        # Encode to base64
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Doc 1 - Legalese

# COMMAND ----------

# DBTITLE 1,Image Conversion
subfolder = f'/Volumes/{catalog}/{schema}/{volume}/legislation'
os.makedirs(subfolder, exist_ok=True)

#pages = convert_from_path(legalese, output_folder=subfolder)
pages_in_doc = os.listdir(subfolder)
len(pages_in_doc)

# COMMAND ----------

sample_file = os.path.join(subfolder, pages_in_doc[24])
chosen_page = encode_ppm_image(sample_file)

# COMMAND ----------

prompt = f"""The following is a page from a pdf article.

Do not explain what the image is just give us a markdown representation markdown
"""

message = HumanMessage(
    content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{chosen_page}"}},
    ],
)

vlm_model.invoke([message])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Doc 2 - Annual Report

# COMMAND ----------

ar_subfolder = f'/Volumes/{catalog}/{schema}/{volume}/annual_report'
os.makedirs(ar_subfolder, exist_ok=True)

#ar_pages = convert_from_path(annual_report, output_folder=ar_subfolder)
ar_pages = os.listdir(ar_subfolder)
len(ar_pages)

# COMMAND ----------

sample_file = os.path.join(ar_subfolder, ar_pages[24])
chosen_page = encode_ppm_image(sample_file)

# COMMAND ----------

prompt = f"""The following is a page from a pdf article.

Do not explain what the image is just give us a markdown representation markdown
"""

message = HumanMessage(
    content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{chosen_page}"}},
    ],
)

vlm_model.invoke([message])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Doc 3 - PDS

# COMMAND ----------

pds_subfolder = f'/Volumes/{catalog}/{schema}/{volume}/annual_report'
os.makedirs(pds_subfolder, exist_ok=True)

#pds_pages = convert_from_path(pds, output_folder=pds_subfolder)
pds_pages = os.listdir(pds_subfolder)
len(pds_pages)

# COMMAND ----------

sample_file = os.path.join(pds_subfolder, pds_pages[10])
chosen_page = encode_ppm_image(sample_file)

# COMMAND ----------

prompt = f"""The following is a page from a pdf article.

Do not explain what the image is just give us a markdown representation markdown
"""

message = HumanMessage(
    content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{chosen_page}"}},
    ],
)

vlm_model.invoke([message])

# COMMAND ----------


