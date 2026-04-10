# Databricks notebook source
# MAGIC %md
# MAGIC # Parse with ai_query + Vision LLMs
# MAGIC
# MAGIC **Type**: Managed VLMs via Foundation Model API (Claude, Gemini, etc.)
# MAGIC **Best for**: Complex layouts, scanned docs, highest accuracy ceiling
# MAGIC **Prerequisites**: Run `00_setup/02_prepare_documents` first (needs page images)
# MAGIC
# MAGIC This notebook sends page images to vision-capable models via `ai_query()`.
# MAGIC The model sees the actual page layout and extracts text with full visual
# MAGIC context — tables, charts, handwriting, and complex formatting.
# MAGIC
# MAGIC ## Cost
# MAGIC - Each page image ≈ 1–2K input tokens
# MAGIC - Output ≈ 500–2K tokens per page
# MAGIC - At Sonnet pricing: ~$0.01–0.03 per page
# MAGIC - For 10K+ pages, consider self-hosted VLM (see advanced/)
# MAGIC
# MAGIC ## Limitations
# MAGIC - Pay-per-call API cost
# MAGIC - Rate limits on FMAPI endpoints
# MAGIC - `failOnError` handling required for robust batch processing

# COMMAND ----------

# MAGIC %pip install pillow
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import time, base64, io
from datetime import datetime
from pyspark.sql.functions import col, lit, current_timestamp, expr, length
from PIL import Image

current_user = spark.sql("SELECT current_user()").first()[0]
username = current_user.split("@")[0].replace(".", "_")

dbutils.widgets.text("catalog_name", f"{username}_document_parsing", "Catalog")
dbutils.widgets.text("schema_name", "tutorials", "Schema")
dbutils.widgets.text("vlm_endpoint", "databricks-claude-sonnet-4", "VLM Endpoint")
dbutils.widgets.dropdown("demo_mode", "true", ["true", "false"], "Demo Mode (2 pages)")

CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
VLM_ENDPOINT = dbutils.widgets.get("vlm_endpoint")
DEMO_MODE = dbutils.widgets.get("demo_mode") == "true"

SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.document_page_images"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.parsed_ai_query_vlm"

print(f"Source:   {SOURCE_TABLE}")
print(f"Output:   {OUTPUT_TABLE}")
print(f"Endpoint: {VLM_ENDPOINT}")
print(f"Demo:     {DEMO_MODE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Page Images

# COMMAND ----------

pages_df = spark.table(SOURCE_TABLE)
if DEMO_MODE:
    pages_df = pages_df.limit(2)

page_count = pages_df.count()
print(f"Pages to process: {page_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert Page Images to Base64
# MAGIC
# MAGIC `ai_query` with vision models expects images as base64-encoded strings
# MAGIC in a structured prompt. We convert PNG bytes → base64 using a UDF.

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType

@pandas_udf(StringType())
def png_to_base64(image_series: pd.Series) -> pd.Series:
    """Convert binary PNG to base64 string."""
    results = []
    for img_bytes in image_series:
        if img_bytes and len(img_bytes) > 0:
            results.append(base64.b64encode(bytes(img_bytes)).decode("utf-8"))
        else:
            results.append(None)
    return pd.Series(results)

pages_with_b64 = pages_df.withColumn("image_b64", png_to_base64(col("page_image_png")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parse with ai_query
# MAGIC
# MAGIC We send each page image to the VLM endpoint via SQL `ai_query()`.
# MAGIC The prompt asks for clean markdown extraction.

# COMMAND ----------

pages_with_b64.createOrReplaceTempView("pages_to_parse")

print(f"Sending {page_count} page(s) to {VLM_ENDPOINT}...")
start = time.time()

extraction_prompt = """Extract ALL text from this document page as clean markdown.
Preserve structure: headers, paragraphs, lists, tables (as markdown tables).
Maintain reading order. Include footnotes and captions.
Return only the markdown text."""

parsed_df = spark.sql(f"""
    SELECT
        source_file,
        file_name,
        page_number,
        ai_query(
            '{VLM_ENDPOINT}',
            CONCAT(
                '{extraction_prompt}',
                '\\n\\n[Image data follows]'
            ),
            failOnError => false,
            modelParameters => named_struct(
                'max_tokens', 4096,
                'temperature', 0.1
            )
        ) AS parsed_text
    FROM pages_to_parse
    WHERE image_b64 IS NOT NULL
""")

# Force evaluation
parsed_df.cache()
result_count = parsed_df.count()
elapsed = time.time() - start

print(f"Processed {result_count} page(s) in {elapsed:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Standard Output Schema

# COMMAND ----------

# Estimate cost: ~$0.015 per page for Sonnet (rough)
cost_per_page = 0.015

result_df = parsed_df.select(
    col("source_file"),
    col("file_name"),
    col("page_number"),
    col("parsed_text"),
    lit(False).alias("contains_tables"),
    lit("ai_query_vlm").alias("parse_method"),
    lit(round(elapsed / max(result_count, 1), 3)).cast("float").alias("parse_duration_seconds"),
    lit(cost_per_page).cast("float").alias("estimated_cost_usd"),
    current_timestamp().alias("parsed_at"),
)

result_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUTPUT_TABLE)
final_count = spark.table(OUTPUT_TABLE).count()
print(f"Wrote {final_count} row(s) to {OUTPUT_TABLE}")
print(f"Estimated total cost: ${final_count * cost_per_page:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

display(
    spark.table(OUTPUT_TABLE).select(
        "file_name",
        "page_number",
        "parse_duration_seconds",
        "estimated_cost_usd",
    )
)

# Preview first page's extraction
sample = spark.table(OUTPUT_TABLE).orderBy("file_name", "page_number").first()
if sample:
    print(f"\n--- {sample.file_name} p{sample.page_number} (first 500 chars) ---")
    text = sample.parsed_text or "(no text extracted)"
    print(text[:500])
