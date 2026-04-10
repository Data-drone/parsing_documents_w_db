# Databricks notebook source
# MAGIC %md
# MAGIC # Parse with PyMuPDF4LLM
# MAGIC
# MAGIC **Type**: Open source, CPU-only, zero API cost
# MAGIC **Best for**: Clean digital PDFs with text layers
# MAGIC **Prerequisites**: Run `00_setup/02_prepare_documents` first
# MAGIC
# MAGIC PyMuPDF4LLM extracts text from PDFs using the embedded text layer and
# MAGIC converts it to clean markdown. It's fast, free, and works well when PDFs
# MAGIC have proper text (not scanned images).
# MAGIC
# MAGIC ## Limitations
# MAGIC - Cannot read scanned/image-only PDFs
# MAGIC - No table structure detection
# MAGIC - Quality depends on how the PDF was created

# COMMAND ----------

# MAGIC %pip install pymupdf4llm pymupdf
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import io, time, re
from datetime import datetime

import pymupdf
import pymupdf4llm
import pandas as pd
from pyspark.sql.functions import col, lit, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, FloatType, TimestampType

current_user = spark.sql("SELECT current_user()").first()[0]
username = re.sub(r"[^a-z0-9_]", "_", current_user.split("@")[0].lower()).strip("_")
username = re.sub(r"^[0-9]+", "", username) or "user"

dbutils.widgets.text("catalog_name", f"{username}_document_parsing", "Catalog")
dbutils.widgets.text("schema_name", "tutorials", "Schema")
dbutils.widgets.dropdown("demo_mode", "true", ["true", "false"], "Demo Mode (2 docs)")

CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
DEMO_MODE = dbutils.widgets.get("demo_mode") == "true"

SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.document_store"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.parsed_pymupdf"

print(f"Source: {SOURCE_TABLE}")
print(f"Output: {OUTPUT_TABLE}")
print(f"Demo:   {DEMO_MODE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Documents

# COMMAND ----------

source_df = spark.table(SOURCE_TABLE).filter(col("file_extension") == ".pdf")
if DEMO_MODE:
    source_df = source_df.limit(2)

doc_count = source_df.count()
print(f"Documents to process: {doc_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract Markdown with Pandas UDF

# COMMAND ----------

output_schema = StructType([
    StructField("source_file", StringType()),
    StructField("file_name", StringType()),
    StructField("page_number", IntegerType()),
    StructField("parsed_text", StringType()),
    StructField("contains_tables", BooleanType()),
    StructField("parse_method", StringType()),
    StructField("parse_duration_seconds", FloatType()),
    StructField("estimated_cost_usd", FloatType()),
    StructField("parsed_at", TimestampType()),
])


def _pymupdf_batch(iterator):
    """Extract markdown from PDFs using PyMuPDF4LLM."""
    for batch in iterator:
        results = []
        now = datetime.now()

        for _, row in batch.iterrows():
            binary = row["binary_content"]
            fname = row["file_name"]
            vol_path = row["volume_path"]

            if binary is None or len(binary) == 0:
                continue

            start = time.time()
            try:
                doc = pymupdf.open(stream=bytes(binary), filetype="pdf")
                md = pymupdf4llm.to_markdown(doc)
                elapsed = time.time() - start
                page_count = doc.page_count
                doc.close()

                # One row per document (whole-doc extraction)
                results.append({
                    "source_file": vol_path,
                    "file_name": fname,
                    "page_number": None,
                    "parsed_text": md,
                    "contains_tables": False,
                    "parse_method": "pymupdf",
                    "parse_duration_seconds": round(elapsed, 3),
                    "estimated_cost_usd": 0.0,
                    "parsed_at": now,
                })
            except Exception as e:
                print(f"Error processing {fname}: {e}")

        if results:
            yield pd.DataFrame(results)
        else:
            yield pd.DataFrame(columns=[f.name for f in output_schema.fields])


# COMMAND ----------

# MAGIC %md
# MAGIC ## Process Documents

# COMMAND ----------

print(f"Processing {doc_count} document(s) with PyMuPDF4LLM...")
start = time.time()

result_df = source_df.mapInPandas(_pymupdf_batch, schema=output_schema)
result_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUTPUT_TABLE)

elapsed = time.time() - start
result_df = spark.table(OUTPUT_TABLE)
row_count = result_df.count()

print(f"Done in {elapsed:.1f}s — {row_count} row(s) written to {OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

display(
    result_df.select(
        "file_name",
        "parse_duration_seconds",
        "estimated_cost_usd",
    )
)

# Preview first document's markdown
sample = result_df.select("file_name", "parsed_text").first()
if sample:
    print(f"\n--- {sample.file_name} (first 500 chars) ---")
    print(sample.parsed_text[:500])
