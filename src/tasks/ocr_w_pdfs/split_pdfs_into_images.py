# Databricks notebook source
# MAGIC %md
# MAGIC %md
# MAGIC
# MAGIC # Scaling up parsing routines
# MAGIC
# MAGIC Add explanatory notes here
# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC TODOs:
# MAGIC Add logging to the functions
# MAGIC Update logger to filer out standard py4j command receives
# MAGIC XXX

# COMMAND ----------

#standard imports
import os
import pandas as pd
import fitz
import logging
import sys
import time

sys.path.append("../..")

#databricks specific
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.window import Window

#additional helper functions
from utils.databricks_utils import get_username_from_email
from utils.import_module import safe_import_module

# COMMAND ----------

# DBTITLE 1,Constants
#num_partitions = spark.sparkContext.defaultParallelism * 2  #adjust the partition count based on cluster size
num_partitions = 16
user_name = get_username_from_email(dbutils = dbutils)
catalog = f"{user_name}_parsing"
input_schema = dbutils.widgets.get("input_schema_name")
output_schema = dbutils.widgets.get("output_schema_name")
volume = dbutils.widgets.get("volume_name")
file_path = dbutils.widgets.get("file_path")
full_vol_path = f'/Volumes/{catalog}/{input_schema}/{volume}'

output_table_schema = StructType([
    StructField("doc_id", StringType()),
    StructField("page_num", IntegerType()),
    StructField("page_bytes", BinaryType()),
    StructField("metadata", StringType())
])

common_functions = safe_import_module("common_functions", f"/Workspace/{file_path}")

# COMMAND ----------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------
#TODO: Make this a more modular function and share between different workflows
def parse_pdf_pages(pdf_group: pd.DataFrame) -> pd.DataFrame:
    """
    Parse PDF content into pages but converts them into pages to demo OCR
    """
    results = []
    
    for _, row in pdf_group.iterrows():
        try:
            filepath = row.get("path", "")
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
                
                #WEre going to create higher resolution images to improve the quality of the ocr. Note this requires 4x the memory of the original processing
                matrix = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img_bytes = pix.pil_tobytes(format="png")
                
                results.append({
                    "doc_id" : doc_id,
                    "page_num" : page_num + 1,
                    "page_bytes" : img_bytes,
                    "metadata" : str(enhanced_metadata)
                })
            
            doc.close()
        except Exception as e:
            logging.error(f"Error processing document {filepath}: {str(e)}")
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

def parse_pipeline(catalog, schema, full_vol_path, output_schema_structure, num_partitions=None):
    """Main processing pipeline"""
    try:
        raw_files_df = (
            spark.read.format("binaryFile")
            .option("recursiveFileLookup", "true")
            .option("pathGlobFilter", "*.pdf")
            .load(full_vol_path)
        )

        if raw_files_df.count() == 0:
            raise ValueError(f"No PDF files found in {full_vol_path}")
        
        start_time = time.time()
        input_count = raw_files_df.count()
        
        processed_df = process_pdfs(raw_files_df, output_schema_structure, num_partitions)
        
        (processed_df.write
            .mode("overwrite")
            .partitionBy("doc_id")
            .saveAsTable(f"{catalog}.{schema}.split_pages")
        )

        end_time = time.time()
        output_count = processed_df.count()
        
        logging.info(f"""
        Processing Summary:
        - Input Files: {input_count}
        - Output Pages: {output_count}
        - Processing Time: {end_time - start_time:.2f} seconds
        - Pages/Second: {output_count/(end_time - start_time):.2f}
        """)
        
        return processed_df
        
    except Exception as e:
        logging.error(f"Error in main processing: {str(e)}")
        raise

# COMMAND ----------

result_df = parse_pipeline(
    catalog = catalog,
    schema = input_schema,
    full_vol_path = full_vol_path,
    output_schema_structure = output_table_schema,
    num_partitions = num_partitions
)

# COMMAND ----------

display(result_df)
