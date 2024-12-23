# Databricks notebook source
# MAGIC %md
# MAGIC # Lets Parse Docs w OCR
# MAGIC
# MAGIC Lets look at how to parse and work with docs

# COMMAND ----------

!apt-get update
!apt install -y tesseract-ocr
!apt install -y libtesseract-dev
!apt install -y poppler-utils

# COMMAND ----------

# MAGIC %pip install -U pytesseract pillow langchain==0.2.17 langchain_databricks==0.1.1 pdf2image
# MAGIC %restart_python

# COMMAND ----------

#Some simple helper functions
from helper_functions.utils import get_username_from_email

# COMMAND ----------

# DBTITLE 1,Set Up Configurations
import os

user_name = get_username_from_email(dbutils = dbutils)
catalog = f"{user_name}_parsing"
schema = 'parsing_tests'
volume = 'raw_data'

llm_model_name = 'brian_serving_test'
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
import pytesseract
from pdf2image import convert_from_path

model = ChatDatabricks(
  target_uri='databricks',
  endpoint=llm_model_name,
  temperature=0.1
)

model.invoke([HumanMessage(content='Harro')])

# COMMAND ----------

# DBTITLE 1,Helper Functions
import pytesseract

def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text


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
pages_in_doc

# COMMAND ----------

extracted_text = []

for page in pages_in_doc[0:10]:
    # Step 2: Preprocess the image (deskew)
    
    # Step 3: Extract text using OCR
    print(os.path.join(subfolder, page))
    text = extract_text_from_image(os.path.join(subfolder, page))
    extracted_text.append(text)

extracted_text

# COMMAND ----------

len(extracted_text)

# COMMAND ----------

prompt = f"""You are a legal analysis bot. Your job is to colloquially explain corporate legalese texts

Use the following text snippet as context
{extracted_text}

What are current weekly earnings? 
"""



response = model.invoke([HumanMessage(content=prompt)])
print(response.content)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Doc 2 - Annual Report

# COMMAND ----------

ar_subfolder = f'/Volumes/{catalog}/{schema}/{volume}/annual_report'
os.makedirs(ar_subfolder, exist_ok=True)

#ar_pages = convert_from_path(annual_report, output_folder=ar_subfolder)
ar_pages = os.listdir(ar_subfolder)
ar_pages

# COMMAND ----------

ar_extracted_text = []

for page in ar_pages[0:10]:
    # Step 2: Preprocess the image (deskew)
    
    # Step 3: Extract text using OCR
    text = extract_text_from_image(os.path.join(ar_subfolder, page))
    ar_extracted_text.append(text)

ar_extracted_text

# COMMAND ----------

len(ar_extracted_text)

# COMMAND ----------

prompt = f"""You are a financial and mining productivity bot. Your job is to colloquially explain business results based on annual report data

Use the following report as context
{ar_extracted_text}

What are the 5 ways that Rio's business model delivers value 
"""



response = model.invoke([HumanMessage(content=prompt)])
print(response.content)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Doc 3 - PDS

# COMMAND ----------

pds_subfolder = f'/Volumes/{catalog}/{schema}/{volume}/pds'
os.makedirs(pds_subfolder, exist_ok=True)

#pds_pages = convert_from_path(pds, output_folder=pds_subfolder)
pds_pages = os.listdir(pds_subfolder)
pds_pages

# COMMAND ----------

extracted_pds_text = []

for page in pds_pages[0:10]:
    # Step 2: Preprocess the image (deskew)
    
    # Step 3: Extract text using OCR
    text = extract_text_from_image(os.path.join(pds_subfolder, page))
    extracted_pds_text.append(text)

extracted_pds_text

# COMMAND ----------

prompt = f"""You are an insurance claims agent

Use the following report as context
{extracted_pds_text}

Answer the following question, make sure to recheck any numbers that you extract

Under what conditions would I not be covered for fire damange?
"""



response = model.invoke([HumanMessage(content=prompt)])
print(response.content)
