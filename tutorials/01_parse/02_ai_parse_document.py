# Databricks notebook source
# MAGIC %md
# MAGIC # Parse with ai_parse_document
# MAGIC
# MAGIC **Type**: Native Databricks SQL function, zero Python dependencies
# MAGIC **Best for**: General purpose — tables, figures, layout extraction out of the box
# MAGIC **Prerequisites**: Run `00_setup/01_environment_setup` first (needs files in volume), DBR 17.1+
# MAGIC
# MAGIC `ai_parse_document()` is Databricks' built-in document parsing function.
# MAGIC One SQL call on a file path returns structured VARIANT output with tables,
# MAGIC figures, headers, and text — no libraries or model servers required.
# MAGIC
# MAGIC ## How it works
# MAGIC - Reads files directly from Unity Catalog Volumes
# MAGIC - Returns VARIANT with structured elements (tables, figures, text blocks)
# MAGIC - Pricing is per-page (check Databricks pricing for current rates)
# MAGIC
# MAGIC ## Limitations
# MAGIC - Databricks-managed model — you can't swap it
# MAGIC - Less prompt control than ai_query approach

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import time
from datetime import datetime
from pyspark.sql.functions import col, lit, current_timestamp, explode, length, expr

current_user = spark.sql("SELECT current_user()").first()[0]
username = current_user.split("@")[0].replace(".", "_")

dbutils.widgets.text("catalog_name", f"{username}_document_parsing", "Catalog")
dbutils.widgets.text("schema_name", "tutorials", "Schema")
dbutils.widgets.text("volume_name", "sample_docs", "Volume")
dbutils.widgets.dropdown("demo_mode", "true", ["true", "false"], "Demo Mode (2 files)")

CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
VOLUME = dbutils.widgets.get("volume_name")
DEMO_MODE = dbutils.widgets.get("demo_mode") == "true"

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.parsed_ai_parse_document"

print(f"Volume:  {VOLUME_PATH}")
print(f"Output:  {OUTPUT_TABLE}")
print(f"Demo:    {DEMO_MODE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## List Files in Volume

# COMMAND ----------

files_df = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.pdf")
    .load(VOLUME_PATH)
    .select(
        col("path"),
        col("length").alias("file_size_bytes"),
    )
    .withColumn("file_name", expr("element_at(split(path, '/'), -1)"))
)

if DEMO_MODE:
    files_df = files_df.limit(2)

file_count = files_df.count()
print(f"Files to parse: {file_count}")
display(files_df.select("file_name", "file_size_bytes"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parse with ai_parse_document
# MAGIC
# MAGIC The function returns a VARIANT column with an array of parsed elements.
# MAGIC Each element has a `type` (e.g., "table", "figure", "text") and content.

# COMMAND ----------

# Register files as temp view for SQL access
files_df.createOrReplaceTempView("files_to_parse")

print("Running ai_parse_document on each file...")
start = time.time()

# ai_parse_document takes a file path and returns VARIANT
parsed_df = spark.sql(f"""
    SELECT
        path AS source_file,
        file_name,
        file_size_bytes,
        ai_parse_document(path) AS parsed_result
    FROM files_to_parse
""")

# Force evaluation
parsed_df.cache()
parsed_count = parsed_df.count()
elapsed = time.time() - start

print(f"Parsed {parsed_count} file(s) in {elapsed:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract to Standard Output Schema
# MAGIC
# MAGIC Convert the VARIANT output into our standard schema with one row per document.

# COMMAND ----------

# Extract text from VARIANT — concatenate all text elements
# The VARIANT structure varies, so we cast the whole thing to string as parsed_text
# and check for table elements

result_df = parsed_df.selectExpr(
    "source_file",
    "file_name",
    "CAST(NULL AS INT) AS page_number",
    "CAST(parsed_result AS STRING) AS parsed_text",
    """CASE
        WHEN CAST(parsed_result AS STRING) LIKE '%table%' THEN true
        ELSE false
    END AS contains_tables""",
    "'ai_parse_document' AS parse_method",
    f"CAST({elapsed / max(parsed_count, 1):.3f} AS FLOAT) AS parse_duration_seconds",
    "CAST(0.01 * file_size_bytes / 100000 AS FLOAT) AS estimated_cost_usd",  # rough estimate
    "current_timestamp() AS parsed_at",
)

result_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUTPUT_TABLE)
row_count = spark.table(OUTPUT_TABLE).count()
print(f"Wrote {row_count} row(s) to {OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explore VARIANT Output
# MAGIC
# MAGIC The raw VARIANT output contains rich structured data. Let's look at what
# MAGIC `ai_parse_document` found in our documents.

# COMMAND ----------

# Show the raw VARIANT for the first document
display(parsed_df.select("file_name", "parsed_result").limit(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

display(
    spark.table(OUTPUT_TABLE).select(
        "file_name",
        "contains_tables",
        "parse_duration_seconds",
        "estimated_cost_usd",
    )
)

# Preview parsed text
sample = spark.table(OUTPUT_TABLE).select("file_name", "parsed_text").first()
if sample:
    print(f"\n--- {sample.file_name} (first 500 chars) ---")
    print(sample.parsed_text[:500])
