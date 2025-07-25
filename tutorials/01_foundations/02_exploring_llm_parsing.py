# Databricks notebook source
# MAGIC %md
# MAGIC # Lets Parse Docs w LLM
# MAGIC
# MAGIC Lets look at how to parse and work with docs

# COMMAND ----------

#TODO: use a vision focused model instead.
#TODO: Convert into workflow instead.

# COMMAND ----------

# MAGIC %pip install -U pillow langchain==0.2.17 langchain_databricks==0.1.1
# MAGIC %restart_python

# COMMAND ----------

params = {"catalog_name":"parsing","file_path":"/Users/nicholas.anile@mantelgroup.com.au/powering_knowledge_driven_applications/files/src/utils/common_functions.py","input_schema_name":"knowledge_extraction","output_schema_name":"knowledge_extraction","volume_name":"raw_data"}

# COMMAND ----------

from pyspark.dbutils import DBUtils

def get_username_from_email(dbutils: DBUtils):
    """
    Extract username from email address in Databricks notebook context.
    """
    try:
        user_email = (dbutils.notebook.entry_point
                                .getDbutils()
                                .notebook()
                                .getContext()
                                .tags().apply("user")
            )
        
        if not user_email or '@' not in user_email:
            raise ValueError("Invalid email address")
            
        username = user_email.split('@')[0].replace('.', '_')
        
        return username
        
    except Exception as e:
        raise ValueError(f"Failed to get username: {str(e)}")

# COMMAND ----------

# DBTITLE 1,Set Up Configurations
#num_partitions = spark.sparkContext.defaultParallelism * 2  #adjust the partition count based on cluster size
num_partitions = 16
user_name = get_username_from_email(dbutils = dbutils)
catalog = f"{user_name}_parsing"
input_schema = params.get("input_schema_name")
output_schema = params.get("output_schema_name")
volume = params.get("volume_name")
file_path = params.get("file_path")

full_vol_path = f'/Volumes/{catalog}/{input_schema}/{volume}'

# COMMAND ----------

# DBTITLE 1,Setup the Langchain Testing
from langchain_databricks import ChatDatabricks
from langchain_core.messages import HumanMessage
from openai import OpenAI

llm_model = ChatDatabricks(
  target_uri='databricks',
  endpoint='databricks-meta-llama-3-3-70b-instruct',
  temperature=0.1
)

llm_model.invoke([HumanMessage(content='can you digest imges?')])

# COMMAND ----------

df_images = spark.table(f"nicholas_anile_parsing.knowledge_extraction.parsed_pages")

# COMMAND ----------

df_images.printSchema()

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import DataFrame

def filter_and_select_random_row(
    df: DataFrame, 
    filter_column: str, 
    filter_value, 
    seed: int = 42
) -> DataFrame:
    """
    Filters a PySpark DataFrame based on a given column and value, then selects a random row from the filtered data using a specified seed for reproducibility.
    """

    df_filtered = df.filter(F.col(filter_column) == filter_value)
    df_random_row = df_filtered.orderBy(F.rand(seed)).limit(1)

    return df_random_row


# COMMAND ----------

import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import io
import PIL.Image

# Function to render an image from byte data
def render_png(byte_data):
    try:
        image = PIL.Image.open(io.BytesIO(byte_data))
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    except Exception as e:
        print("Error rendering image:", e)


# COMMAND ----------

display(df_images)

# COMMAND ----------

doc_id = "65e6da6e3592032eedf37e413ffbf9d1bfc381e1bbf82268b74745d3ca201e72"
split_page_doc_id = "6513dd1370fa453664cb0ad3a6d6245607800c0c723b67c5779409ecb8c829fd"

# COMMAND ----------

#Select a single image for testing purposes
df_random_image = filter_and_select_random_row(
    df_images,
    filter_column="doc_id",
    filter_value=f"{doc_id}",
    seed=42
)

# COMMAND ----------

display(df_random_image)

# COMMAND ----------

df_random_image_page = df_random_image.select("page_num").collect()[0][0]
print(df_random_image_page)

# COMMAND ----------

df_image_content = (spark.table("nicholas_anile_parsing.knowledge_extraction.split_pages")
                    .filter(F.col("doc_id") == split_page_doc_id)
                    .filter(F.col("page_num") == df_random_image_page)
                    )

# COMMAND ----------

display(df_image_content)

# COMMAND ----------

bytes_array = df_image_content.select("page_bytes").first()[0]

# COMMAND ----------

render_png(bytes_array)

# COMMAND ----------

#need to resize the image to fit within token counts
from PIL import Image
import io, base64

image = Image.open(io.BytesIO(bytes_array))


new_width = image.width // 2
new_height = image.height // 2
resized_image = image.resize((new_width, new_height))

buffer = io.BytesIO()
resized_image.save(buffer, format="PNG", quality=85)
compressed_data = buffer.getvalue()
compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')


# COMMAND ----------

prompt = f"""The following is a page from a pdf article.

Do not explain what the image is but instead give us a markdown representation markdown.
- Extract all the text content in the image
- If a table exists in the image, extract the content but maintain the existing structure

Here's the image in base64 format: {compressed_base64}
"""

message = HumanMessage(
    content=prompt
)

llm_model.invoke([message])