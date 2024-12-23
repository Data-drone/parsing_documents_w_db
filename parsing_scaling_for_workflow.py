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

# MAGIC %pip install -U pymupdf4llm
# MAGIC %restart_python

# COMMAND ----------

#standard imports
import os
import pandas as pd
import datetime
import hashlib
import fitz

#databricks specific
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import functions as F

#additional helper functions
from helper_functions.utils import get_username_from_email

# COMMAND ----------

# DBTITLE 1,Constants
num_partitions = spark.sparkContext.defaultParallelism * 2  #adjust the partition count based on cluster size
user_name = get_username_from_email(dbutils = dbutils)
catalog = f"{user_name}_parsing"
schema = 'parsing_tests'
volume = 'raw_data'
full_vol_path = f'/Volumes/{catalog}/{schema}/{volume}'

example_files = [f for f in os.listdir(f'{full_vol_path}')  if f.endswith('.pdf')]
example_files

# COMMAND ----------

def generate_doc_id(filename: str, timestamp: str) -> str:
    """
    Generate a deterministic ID combining the filename and timestamp of processing
    
    """
    content = f"{filename}{timestamp}"
    return hashlib.sha256(content.encode()).hexdigest()

# COMMAND ----------

from pathlib import Path

def extract_filename(dbfs_path: str) -> str:
    """
    Extracts the filename from the volume file path
    
    Example: "dbfs:/Volumes/catalog/schema/volume_name/file.pdf" become "file.pdf"
            
    """
    return Path(dbfs_path).name

# COMMAND ----------

def extract_datetime() -> str:
    """
    Get current datetime in ISO format
    """
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

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
            filename = extract_filename(filepath)
            timestamp = extract_datetime()
            
            doc_id = generate_doc_id(filename, timestamp)
            

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
            print(f"Error processing document {filename}: {str(e)}")
            continue
    
    if not results:
        return pd.DataFrame(columns=["doc_id", "page_num", "content", "metadata"])
    
    return pd.DataFrame(results)


output_schema = StructType([
    StructField("doc_id", StringType()),
    StructField("page_num", IntegerType()),
    StructField("content", StringType()),
    StructField("metadata", StringType())
])

# COMMAND ----------

def process_pdfs(raw_files_df, num_partitions=None):
    """Process PDFs using applyInPandas"""
    processed_df = raw_files_df.withColumn("batch_id", F.expr("uuid()"))
    
    if num_partitions:
        processed_df = processed_df.repartition(num_partitions)
    
    parsed_pages = processed_df.groupby("batch_id").applyInPandas(
        parse_pdf_pages,
        schema=output_schema
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

def parse_pipeline(catalog, schema, full_vol_path, num_partitions=None):
    """Main processing pipeline"""
    try:
        raw_files_df = (
            spark.read.format("binaryFile")
            .option("recursiveFileLookup", "true")
            .option("pathGlobFilter", "*.pdf")
            .load(full_vol_path)
        )
        
        processed_df = process_pdfs(raw_files_df, num_partitions)
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

result_df = parse_pipeline(
    catalog = catalog,
    schema = schema,
    full_vol_path = full_vol_path,
    num_partitions = num_partitions
)

# COMMAND ----------

display(result_df)
