# Databricks notebook source
# MAGIC %md
# MAGIC # Document Store Creation
# MAGIC 
# MAGIC **Time**: 5 minutes  
# MAGIC **Prerequisites**: Completed `01_environment_setup` notebook
# MAGIC 
# MAGIC This notebook creates a document store table containing both file metadata and binary content for all documents in your volume.
# MAGIC 
# MAGIC ## What This Notebook Does
# MAGIC 1. Scans your volume for PDF documents recursively
# MAGIC 2. Extracts file metadata (name, size, modification time, etc.)
# MAGIC 3. Reads binary content of each document
# MAGIC 4. Stores everything in a Delta Lake table for downstream processing
# MAGIC 
# MAGIC ## What You'll Get
# MAGIC A Delta Lake table containing both metadata and binary content for all your documents, ready for text extraction and analysis in subsequent tutorials.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Setup and Configuration
# MAGIC 
# MAGIC Import libraries and configure your workspace settings.

# COMMAND ----------

import os
from datetime import datetime

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Get current user for naming defaults
current_user = spark.sql("SELECT current_user()").first()[0]
username = current_user.split('@')[0].replace('.', '_')

# Configuration with defaults from environment or sensible fallbacks
CATALOG_NAME = os.getenv("CATALOG_NAME", f"{username}_document_parsing")
SCHEMA_NAME  = os.getenv("SCHEMA_NAME", "tutorials")
VOLUME_NAME  = os.getenv("VOLUME_NAME", "sample_docs")

# Create widgets for easy configuration
dbutils.widgets.text("CATALOG_NAME", CATALOG_NAME, "Catalog")
dbutils.widgets.text("SCHEMA_NAME",  SCHEMA_NAME,  "Schema")
dbutils.widgets.text("VOLUME_NAME",  VOLUME_NAME,  "Volume")

# Get final values (can be overridden via widgets)
CATALOG = dbutils.widgets.get("CATALOG_NAME")
SCHEMA  = dbutils.widgets.get("SCHEMA_NAME")
VOLUME  = dbutils.widgets.get("VOLUME_NAME")

# Derived paths and table names
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.document_store"

print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")
print(f"Volume: {VOLUME}")
print(f"Volume path: {VOLUME_PATH}")
print(f"Output table: {OUTPUT_TABLE}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Load Documents
# MAGIC 
# MAGIC Scan the volume for documents and read their content using Spark's built-in `binaryFile` format.

# COMMAND ----------

from pyspark.sql.functions import col, regexp_extract, split, element_at, when, lit, concat
import time

print("Loading documents from volume...")

# Load documents using Spark's binaryFile format
# Note: You can modify the glob pattern to include other formats: "*.{pdf,doc,docx}"
df_raw = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.pdf") \
    .option("recursiveFileLookup", "true") \
    .load(VOLUME_PATH)

print("Files loaded successfully")
print("Raw binaryFile schema:")
df_raw.printSchema()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Process File Metadata
# MAGIC 
# MAGIC Transform the raw file data to extract useful metadata fields.

# COMMAND ----------

print("Processing file metadata...")

# Extract useful metadata fields from the file paths
df = df_raw.withColumn(
    "file_name", element_at(split(col("path"), "/"), -1)
).withColumn(
    "file_extension_raw", regexp_extract(col("path"), r"\.([^.]+)$", 1)
).withColumn(
    "file_extension", 
    when(col("file_extension_raw") != "", concat(lit("."), col("file_extension_raw")))
    .otherwise(lit(""))
).withColumn(
    "directory", regexp_extract(col("path"), r"^(.+)/[^/]+$", 1)
).select(
    col("file_name"),
    col("path").alias("volume_path"),
    col("file_extension"),
    col("length").alias("file_size_bytes"),
    col("modificationTime").alias("modification_time"),
    col("directory"),
    col("content").alias("binary_content")
)

# Count files found
file_count = df.count()
print(f"Found {file_count} documents")

print("Final schema:")
df.printSchema()

# Preview document metadata (excluding binary content)
print("Document metadata preview:")
df_display = df.select(
    "file_name", 
    "volume_path", 
    "file_extension", 
    "file_size_bytes", 
    "modification_time", 
    "directory"
)
display(df_display)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Save to Delta Lake
# MAGIC 
# MAGIC Save the document store as a managed Delta Lake table.

# COMMAND ----------

print("Saving to Delta Lake...")

# Write to Delta Lake table
df.write.mode("overwrite").saveAsTable(OUTPUT_TABLE)

print(f"Table {OUTPUT_TABLE} created successfully!")
print(f"Stored {file_count} documents with binary content")

# Calculate total storage
total_bytes = df.agg({"file_size_bytes": "sum"}).collect()[0][0] or 0
total_mb = total_bytes / (1024 * 1024)
print(f"Total document storage: {total_mb:.2f} MB")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Verify Results
# MAGIC 
# MAGIC Check that the documents were processed correctly and view some summary statistics.

# COMMAND ----------

print("Verifying document storage...")

# Check that binary content was stored properly
verification_query = f"""
SELECT 
    file_name,
    file_extension,
    file_size_bytes,
    CASE 
        WHEN binary_content IS NOT NULL THEN 'Yes' 
        ELSE 'No' 
    END as has_binary_content,
    LENGTH(binary_content) as binary_size_bytes,
    CASE 
        WHEN LENGTH(binary_content) = file_size_bytes THEN 'Match'
        WHEN binary_content IS NULL THEN 'Missing'
        ELSE 'Size Mismatch'
    END as size_validation
FROM {OUTPUT_TABLE}
ORDER BY file_name
"""

result_df = spark.sql(verification_query)
display(result_df)

print("Verification complete!")
print(f"\nTo access your documents:")
print(f"df = spark.table('{OUTPUT_TABLE}')")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Summary Statistics
# MAGIC 
# MAGIC View summary statistics about your document collection.

# COMMAND ----------

# Generate summary statistics
summary_query = f"""
SELECT 
    file_extension,
    COUNT(*) as file_count,
    ROUND(SUM(file_size_bytes) / 1024 / 1024, 2) as total_mb,
    ROUND(AVG(file_size_bytes) / 1024, 2) as avg_size_kb,
    MIN(modification_time) as oldest_file,
    MAX(modification_time) as newest_file
FROM {OUTPUT_TABLE}
WHERE binary_content IS NOT NULL
GROUP BY file_extension
ORDER BY file_count DESC
"""

summary_df = spark.sql(summary_query)
display(summary_df)

print("Document store creation complete!")
print(f"\nYour document store table '{OUTPUT_TABLE}' is ready for use in the next tutorials.")