# Databricks notebook source
# COMMAND ----------
# Setup
import os
from datetime import datetime
from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType

# Use env vars if available, else hardcoded defaults
CATALOG = "brian_gen_ai"
SCHEMA = "parsing_test"
VOLUME = "raw_data"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.document_store"

print(f"Using volume: {VOLUME_PATH}")
print(f"Output table: {OUTPUT_TABLE}")

# COMMAND ----------
# List files recursively and collect metadata
def list_files_with_metadata(volume_path, exts=(".pdf", ".doc", ".docx")):
    all_files = []
    def _recurse(path):
        try:
            for entry in dbutils.fs.ls(path):
                if entry.isDir():
                    _recurse(entry.path)
                else:
                    if entry.name.lower().endswith(exts):
                        all_files.append({
                            "file_name": entry.name,
                            "volume_path": entry.path,
                            "file_extension": os.path.splitext(entry.name)[1].lower(),
                            "file_size_bytes": entry.size,
                            "modification_time": datetime.fromtimestamp(entry.modificationTime / 1000.0),
                            "directory": os.path.dirname(entry.path.rstrip("/"))
                        })
        except Exception as e:
            print(f"Error accessing {path}: {e}")
    _recurse(volume_path)
    return all_files

files = list_files_with_metadata(VOLUME_PATH)
print(f"Found {len(files)} document files.")

# COMMAND ----------
# Create Spark DataFrame
def make_spark_df(files):
    schema = StructType([
        StructField("file_name", StringType(), True),
        StructField("volume_path", StringType(), True),
        StructField("file_extension", StringType(), True),
        StructField("file_size_bytes", LongType(), True),
        StructField("modification_time", TimestampType(), True),
        StructField("directory", StringType(), True)
    ])
    
    return spark.createDataFrame(files, schema=schema)

df = make_spark_df(files)
display(df)


# COMMAND ----------
# Save as a table
df.write.mode("overwrite").saveAsTable(OUTPUT_TABLE)
print(f"Table {OUTPUT_TABLE} created with {df.count()} files.")

