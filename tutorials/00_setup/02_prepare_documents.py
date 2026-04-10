# Databricks notebook source
# MAGIC %md
# MAGIC # Prepare Documents
# MAGIC
# MAGIC **Time**: 5–15 minutes
# MAGIC **Prerequisites**: Run `01_environment_setup` first
# MAGIC
# MAGIC This notebook creates the input tables that all parser notebooks read from:
# MAGIC 1. `document_store` — binary PDF content + file metadata
# MAGIC 2. `document_page_images` — one row per page with a PNG image (for VLM parsers)

# COMMAND ----------

# MAGIC %pip install pymupdf
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import os, time, gc, re
from datetime import datetime

current_user = spark.sql("SELECT current_user()").first()[0]
username = re.sub(r"[^a-z0-9_]", "_", current_user.split("@")[0].lower()).strip("_")
username = re.sub(r"^[0-9]+", "", username) or "user"

dbutils.widgets.text("catalog_name", f"{username}_document_parsing", "Catalog")
dbutils.widgets.text("schema_name", "tutorials", "Schema")
dbutils.widgets.text("volume_name", "sample_docs", "Volume")
dbutils.widgets.dropdown("demo_mode", "true", ["true", "false"], "Demo Mode (2 pages)")

CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
VOLUME = dbutils.widgets.get("volume_name")
DEMO_MODE = dbutils.widgets.get("demo_mode") == "true"

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
DOC_STORE_TABLE = f"{CATALOG}.{SCHEMA}.document_store"
PAGE_IMAGES_TABLE = f"{CATALOG}.{SCHEMA}.document_page_images"

print(f"Volume path:        {VOLUME_PATH}")
print(f"Document store:     {DOC_STORE_TABLE}")
print(f"Page images table:  {PAGE_IMAGES_TABLE}")
print(f"Demo mode:          {DEMO_MODE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Load PDFs into Document Store
# MAGIC
# MAGIC Uses Spark's `binaryFile` reader to load PDF content and metadata.

# COMMAND ----------

from pyspark.sql.functions import col, element_at, split, regexp_extract, when, lit, concat

df_raw = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.pdf")
    .option("recursiveFileLookup", "true")
    .load(VOLUME_PATH)
)

df = (
    df_raw
    .withColumn("file_name", element_at(split(col("path"), "/"), -1))
    .withColumn("file_extension_raw", regexp_extract(col("path"), r"\.([^.]+)$", 1))
    .withColumn(
        "file_extension",
        when(col("file_extension_raw") != "", concat(lit("."), col("file_extension_raw")))
        .otherwise(lit("")),
    )
    .withColumn("directory", regexp_extract(col("path"), r"^(.+)/[^/]+$", 1))
    .select(
        col("file_name"),
        col("path").alias("volume_path"),
        col("file_extension"),
        col("length").alias("file_size_bytes"),
        col("modificationTime").alias("modification_time"),
        col("directory"),
        col("content").alias("binary_content"),
    )
)

if DEMO_MODE:
    df = df.limit(2)

file_count = df.count()
print(f"Found {file_count} document(s)")

df.write.mode("overwrite").saveAsTable(DOC_STORE_TABLE)
print(f"Document store saved to {DOC_STORE_TABLE}")

display(df.select("file_name", "file_extension", "file_size_bytes"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Split PDFs to Page Images
# MAGIC
# MAGIC Converts each page of each PDF into a PNG image. These are used by VLM-based
# MAGIC parsers (`ai_query`, self-hosted VLM) that process images instead of text.

# COMMAND ----------

import fitz  # PyMuPDF
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BinaryType, LongType
from pyspark.sql.functions import pandas_udf

page_schema = StructType([
    StructField("source_file", StringType(), True),
    StructField("file_name", StringType(), True),
    StructField("page_number", IntegerType(), True),
    StructField("page_image_png", BinaryType(), True),
    StructField("total_pages", IntegerType(), True),
    StructField("file_size_bytes", LongType(), True),
])


def _split_pdf_batch(iterator):
    """mapInPandas function: split each PDF into page images."""
    for batch in iterator:
        results = []
        for _, row in batch.iterrows():
            binary = row["binary_content"]
            fname = row["file_name"]
            fsize = row["file_size_bytes"]
            vol_path = row["volume_path"]

            if binary is None or len(binary) == 0:
                continue

            doc = None
            try:
                doc = fitz.open(stream=bytes(binary), filetype="pdf")
                total = doc.page_count

                max_pages = 2 if DEMO_MODE else total
                for p in range(min(max_pages, total)):
                    pix = doc[p].get_pixmap(alpha=False)
                    img_bytes = pix.pil_tobytes(format="PNG")
                    results.append({
                        "source_file": vol_path,
                        "file_name": fname,
                        "page_number": p + 1,
                        "page_image_png": img_bytes,
                        "total_pages": total,
                        "file_size_bytes": fsize,
                    })
                    pix = None
            except Exception as e:
                print(f"Error splitting {fname}: {e}")
            finally:
                if doc:
                    doc.close()

        if results:
            yield pd.DataFrame(results)
        else:
            yield pd.DataFrame(columns=[f.name for f in page_schema.fields])

# COMMAND ----------

print("Splitting PDFs to page images...")
start = time.time()

source_df = spark.table(DOC_STORE_TABLE).filter(col("file_extension") == ".pdf")

pages_df = source_df.mapInPandas(_split_pdf_batch, schema=page_schema)
pages_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(PAGE_IMAGES_TABLE)

elapsed = time.time() - start
page_count = spark.table(PAGE_IMAGES_TABLE).count()
doc_count = spark.table(PAGE_IMAGES_TABLE).select("file_name").distinct().count()

print(f"Done in {elapsed:.1f}s — {doc_count} document(s), {page_count} page(s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print(f"Document store:  {DOC_STORE_TABLE}  ({file_count} files)")
print(f"Page images:     {PAGE_IMAGES_TABLE}  ({page_count} pages from {doc_count} docs)")
print(f"\nReady for parsing! Run any notebook in 01_parse/:")
print(f"  01_pymupdf.py          — reads document_store (binary PDFs)")
print(f"  02_ai_parse_document.py — reads files from volume directly")
print(f"  03_ai_query_vlm.py     — reads document_page_images (PNG images)")
print(f"  04_docling.py          — reads files from volume directly")
