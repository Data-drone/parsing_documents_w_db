# Databricks notebook source
# MAGIC %md
# MAGIC %md
# MAGIC
# MAGIC # Scaling up parsing routines
# MAGIC
# MAGIC Lets look at how to scale up parsing leveraging a spark cluster
# MAGIC
# MAGIC As a general guidance if you have less than ?couple of dozen? files then it may not be worth distributing across a cluster 

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC
# MAGIC # Write some notes about the logic of this file
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC TODOs:
# MAGIC Add logging to the functions
# MAGIC Update logger to filer out standard py4j command recieves
# MAGIC Convert functions into a class. Tweak the read of the pdf's to account for different conversion (i.e get markdown vs convert to image bytes)
# MAGIC Currently using an import_module to import common functions while the pdf functions exist within this notebook. Probably unnecessary if we split them out too

# COMMAND ----------

#standard imports
import os
import pandas as pd
import fitz
import logging
import sys

sys.path.append("../..")

#databricks specific
from pyspark.sql.types import *
from pyspark.sql import functions as F

#additional helper functions
from utils.databricks_utils import get_username_from_email
from utils.import_module import safe_import_module

# COMMAND ----------

# DBTITLE 1,Constants
num_partitions = spark.sparkContext.defaultParallelism * 2  #adjust the partition count based on cluster size
user_name = get_username_from_email(dbutils = dbutils)
catalog = f"{user_name}_parsing"
schema = dbutils.widgets.get("schema_name")
volume = dbutils.widgets.get("volume_name")
file_path = dbutils.widgets.get("file_path")
full_vol_path = f'/Volumes/{catalog}/{schema}/{volume}'

output_table_schema = StructType([
    StructField("doc_id", StringType()),
    StructField("page_num", IntegerType()),
    StructField("content", StringType()),
    StructField("metadata", StringType())
])

example_files = [f for f in os.listdir(f'{full_vol_path}')  if f.endswith('.pdf')]
example_files

common_functions = safe_import_module("common_functions", f"/Workspace/{file_path}")

# COMMAND ----------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

#Note we're going to retain all the metadata so that it can be regrouped / chunked in other methods

def parse_pdf_pages(pdf_group: pd.DataFrame) -> pd.DataFrame:
    """
    Parse PDF content into pages
    """
    results = []
    
    for _, row in pdf_group.iterrows():
        try:
            filepath = row.get("path", "")
            #print('here')
            #logging.info(f"{str(filepath)}")
            
            filename = common_functions.extract_filename(filepath)
            timestamp = common_functions.extract_datetime()
            
            doc_id = common_functions.generate_doc_id(filename, timestamp)
            
            doc = fitz.open(stream = row['content'], filetype = "pdf")
            
            pdf_metadata = doc.metadata if doc.metadata else {}
            enhanced_metadata = {
                "source_filename" : filename,
                "source_filepath" : filepath,
                "processing_timestamp" : timestamp,
                "pdf_metadata" : pdf_metadata,
                "total_pages" : len(doc),
                "file_size" : len(row["content"]) if "content" in row else 0
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                content = page.get_text("markdown")
                
                results.append({
                    "doc_id" : doc_id,
                    "page_num" : page_num + 1,
                    "content" : content,
                    "metadata" : str(enhanced_metadata)
                })
            
            doc.close()
        except Exception as e:
            print(f"Error processing document {filepath}: {str(e)}")
            continue
    
    if not results:
        return pd.DataFrame(columns=["doc_id", "page_num", "content", "metadata"])
    
    return pd.DataFrame(results)

# COMMAND ----------

def process_pdfs(raw_files_df, output_struct_schema, num_partitions=None):
    """Process PDFs using applyInPandas"""
    processed_df = raw_files_df.withColumn("batch_id", F.expr("uuid()"))
    
    if num_partitions:
        processed_df = processed_df.repartition(num_partitions)
    
    parsed_pages = processed_df.groupby("batch_id").applyInPandas(
        parse_pdf_pages,
        schema = output_struct_schema
    )
    
    return parsed_pages

# COMMAND ----------

#TODO:
#create a class instead and have incremental tracking so that it's easier to track the progress
def monitor_processing(df):
    """df monitor of the PDF processing"""
    metrics = df.agg(
        F.countDistinct("doc_id").alias("total_docs"),
        F.count("*").alias("total_pages"),
        F.countDistinct(F.when(F.col("content").isNull(), F.col("doc_id"))).alias("failed_docs")
    ).collect()[0]
    
    print(f"""
    Processing Summary:
    ------------------
    Total Documents: {metrics.total_docs}
    Total Pages: {metrics.total_pages}
    Failed Documents: {metrics.failed_docs}
    Average Pages per Document: {metrics.total_pages / metrics.total_docs if metrics.total_docs > 0 else 0:.2f}
    """)
    
    return df

# COMMAND ----------

def parse_pipeline(catalog, schema, full_vol_path, output_schema_structure, num_partitions=None):
    """Main processing pipeline"""
    try:
        raw_files_df = (
            spark.read.format("binaryFile")
            .option("recursiveFileLookup", "true")
            .option("pathGlobFilter", "*.pdf")
            .load(full_vol_path)
        )
        
        processed_df = process_pdfs(raw_files_df, output_schema_structure, num_partitions)
        monitored_df = monitor_processing(processed_df)
        
        (monitored_df.write
            .format("delta")
            .mode("overwrite")
            .partitionBy("doc_id")
            .option("optimizeWrite", "true")
            .option("mergeSchema", "true")
            .saveAsTable(f"{catalog}.{schema}.parsed_pages")
        )
        
        display(monitored_df.select("doc_id", "page_num", "content").limit(5))
        
        return monitored_df
        
    except Exception as e:
        print(f"Error in main processing: {str(e)}")
        raise

# COMMAND ----------
raw_files_df = (
    spark.read.format("binaryFile")
    .option("recursiveFileLookup", "true")
    .option("pathGlobFilter", "*.pdf")
    .load(full_vol_path)
)

display(raw_files_df)

# COMMAND ----------

result_df = parse_pipeline(
    catalog = catalog,
    schema = schema,
    full_vol_path = full_vol_path,
    output_schema_structure = output_table_schema,
    num_partitions = num_partitions
)

# COMMAND ----------

display(result_df)
