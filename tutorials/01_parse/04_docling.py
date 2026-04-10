# Databricks notebook source
# MAGIC %md
# MAGIC # Parse with Docling
# MAGIC
# MAGIC **Type**: Open source with built-in OCR + table structure detection
# MAGIC **Best for**: Documents with tables, scanned PDFs, mixed content
# MAGIC **Prerequisites**: Run `00_setup/01_environment_setup` first (needs files in volume)
# MAGIC
# MAGIC Docling is a newer open-source document processing library that combines:
# MAGIC - Text extraction from digital PDFs
# MAGIC - OCR for scanned/image content
# MAGIC - Table structure detection and preservation
# MAGIC
# MAGIC ## Compared to PyMuPDF
# MAGIC - Slower (does more work per page)
# MAGIC - Handles scanned docs (OCR)
# MAGIC - Detects and preserves table structure
# MAGIC - Still free / open source
# MAGIC
# MAGIC ## Limitations
# MAGIC - Slower than PyMuPDF on simple digital PDFs
# MAGIC - May need more memory for large documents
# MAGIC - Newer library — less battle-tested

# COMMAND ----------

# MAGIC %pip install -U docling
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import os, time, re
from datetime import datetime
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

current_user = spark.sql("SELECT current_user()").first()[0]
username = re.sub(r"[^a-z0-9_]", "_", current_user.split("@")[0].lower()).strip("_")
username = re.sub(r"^[0-9]+", "", username) or "user"

dbutils.widgets.text("catalog_name", f"{username}_document_parsing", "Catalog")
dbutils.widgets.text("schema_name", "tutorials", "Schema")
dbutils.widgets.text("volume_name", "sample_docs", "Volume")
dbutils.widgets.dropdown("demo_mode", "true", ["true", "false"], "Demo Mode (2 files)")

CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
VOLUME = dbutils.widgets.get("volume_name")
DEMO_MODE = dbutils.widgets.get("demo_mode") == "true"

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.parsed_docling"

print(f"Volume:  {VOLUME_PATH}")
print(f"Output:  {OUTPUT_TABLE}")
print(f"Demo:    {DEMO_MODE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Docling Converter

# COMMAND ----------

# File size threshold: files above this use a lighter pipeline to avoid OOM
LARGE_FILE_MB = 5

# Full pipeline: OCR + table structure detection
full_pipeline = PdfPipelineOptions(
    do_ocr=True,
    do_table_structure=True,
)
full_converter = DocumentConverter(
    format_options={"pdf": PdfFormatOption(pipeline_options=full_pipeline)}
)

# Lightweight pipeline: no OCR, no table structure (fast, low memory)
light_pipeline = PdfPipelineOptions(
    do_ocr=False,
    do_table_structure=False,
)
light_converter = DocumentConverter(
    format_options={"pdf": PdfFormatOption(pipeline_options=light_pipeline)}
)

print(f"Docling converters ready (full pipeline for files < {LARGE_FILE_MB}MB, light for larger)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find PDF Files

# COMMAND ----------

pdf_files = []
try:
    files = dbutils.fs.ls(f"dbfs:{VOLUME_PATH}")
    pdf_files = [f for f in files if f.name.lower().endswith(".pdf")]
except Exception as e:
    print(f"Error listing volume: {e}")

if DEMO_MODE:
    pdf_files = pdf_files[:2]

print(f"Files to process: {len(pdf_files)}")
for f in pdf_files:
    print(f"  {f.name} ({f.size / 1024:.1f} KB)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process Documents with Docling

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, FloatType, TimestampType

results = []
now = datetime.now()

for pdf_file in pdf_files:
    fname = pdf_file.name
    fpath = f"{VOLUME_PATH}/{fname}"
    size_mb = pdf_file.size / (1024 * 1024)

    # Pick converter based on file size
    if size_mb > LARGE_FILE_MB:
        active_converter = light_converter
        mode = "light"
    else:
        active_converter = full_converter
        mode = "full"

    print(f"Processing: {fname} ({size_mb:.1f}MB, {mode})...", end=" ")
    start = time.time()

    try:
        result = active_converter.convert(fpath)
        md = result.document.export_to_markdown()
        elapsed = time.time() - start

        # Check for table markers in markdown output
        has_tables = "|" in md and "---" in md

        results.append({
            "source_file": fpath,
            "file_name": fname,
            "page_number": None,
            "parsed_text": md,
            "contains_tables": has_tables,
            "parse_method": "docling",
            "parse_duration_seconds": round(elapsed, 3),
            "parsed_at": now,
        })
        print(f"{len(md)} chars in {elapsed:.1f}s")

    except Exception as e:
        print(f"FAILED — {e}")

print(f"\nProcessed {len(results)}/{len(pdf_files)} file(s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Standard Output Schema

# COMMAND ----------

import pandas as pd

if results:
    schema = StructType([
        StructField("source_file", StringType()),
        StructField("file_name", StringType()),
        StructField("page_number", IntegerType()),
        StructField("parsed_text", StringType()),
        StructField("contains_tables", BooleanType()),
        StructField("parse_method", StringType()),
        StructField("parse_duration_seconds", FloatType()),
        StructField("parsed_at", TimestampType()),
    ])

    result_df = spark.createDataFrame(pd.DataFrame(results), schema=schema)
    result_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUTPUT_TABLE)
    print(f"Wrote {len(results)} row(s) to {OUTPUT_TABLE}")
else:
    print("No results to write")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

if results:
    display(
        spark.table(OUTPUT_TABLE).select(
            "file_name",
            "contains_tables",
            "parse_duration_seconds",
        )
    )

    sample = spark.table(OUTPUT_TABLE).select("file_name", "parsed_text").first()
    if sample:
        print(f"\n--- {sample.file_name} (first 500 chars) ---")
        print(sample.parsed_text[:500])
