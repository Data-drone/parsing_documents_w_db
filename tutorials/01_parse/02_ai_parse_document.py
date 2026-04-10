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

import time, re
from datetime import datetime
from pyspark.sql.functions import col, lit, current_timestamp, explode, length, expr

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
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.parsed_ai_parse_document"

print(f"Volume:  {VOLUME_PATH}")
print(f"Output:  {OUTPUT_TABLE}")
print(f"Demo:    {DEMO_MODE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## List Files in Volume

# COMMAND ----------

# List files in volume first
all_files = [f.name for f in dbutils.fs.ls(f"dbfs:{VOLUME_PATH}") if f.name.lower().endswith(".pdf")]
files = all_files[:2] if DEMO_MODE else all_files
print(f"Files to parse: {len(files)}")
for f in files:
    print(f"  {f}")

# Build a glob filter for the selected files (used by READ_FILES)
if DEMO_MODE and len(all_files) > 2:
    # Create a path list for filtering in SQL later
    file_filter_sql = ", ".join([f"'{VOLUME_PATH}/{f}'" for f in files])
else:
    file_filter_sql = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parse with ai_parse_document
# MAGIC
# MAGIC `ai_parse_document(content, options)` takes BINARY content (not a path string).
# MAGIC We use `READ_FILES(path, format => 'binaryFile')` to read files as raw bytes,
# MAGIC then pass the binary `content` column to the parsing function.
# MAGIC
# MAGIC Returns a VARIANT column with structured elements (text, tables, figures).

# COMMAND ----------

print("Running ai_parse_document...")
start = time.time()

# READ_FILES with binaryFile format gives us a 'content' BINARY column
# Filter files first (before the expensive ai_parse_document call)
filter_clause = f"WHERE path IN ({file_filter_sql})" if file_filter_sql else ""

parsed_df = spark.sql(f"""
    SELECT
        path AS source_file,
        ai_parse_document(
            content,
            map('version', '2.0')
        ) AS parsed_result
    FROM (
        SELECT path, content
        FROM READ_FILES(
            '{VOLUME_PATH}',
            format => 'binaryFile',
            pathGlobFilter => '*.pdf',
            recursiveFileLookup => 'true'
        )
        {filter_clause}
    )
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

# Register parsed results for SQL access
parsed_df.createOrReplaceTempView("parsed_results_temp")

# Extract text from VARIANT: explode elements, concatenate text+table content per doc
cost_per_doc = round(elapsed / max(parsed_count, 1), 3)
result_df = spark.sql(f"""
    WITH elements AS (
        SELECT
            source_file,
            REGEXP_EXTRACT(source_file, '([^/]+)$', 1) AS file_name,
            explode(CAST(parsed_result:document:elements AS ARRAY<VARIANT>)) AS elem
        FROM parsed_results_temp
    )
    SELECT
        source_file,
        file_name,
        CAST(NULL AS INT) AS page_number,
        CONCAT_WS('\n\n',
            COLLECT_LIST(CASE
                WHEN elem:type::STRING IN ('text','title','section_header','caption')
                THEN elem:content::STRING
                WHEN elem:type::STRING = 'table'
                THEN elem:content::STRING
                ELSE NULL
            END)
        ) AS parsed_text,
        MAX(CASE WHEN elem:type::STRING = 'table' THEN true ELSE false END) AS contains_tables,
        'ai_parse_document' AS parse_method,
        CAST({cost_per_doc} AS FLOAT) AS parse_duration_seconds,
        CAST(0.01 AS FLOAT) AS estimated_cost_usd,
        current_timestamp() AS parsed_at
    FROM elements
    GROUP BY source_file, file_name
""")

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

# Show element type counts per document (raw VARIANT is too large to display directly)
display(spark.sql("""
    SELECT
        source_file,
        elem:type::STRING AS element_type,
        COUNT(*) AS count
    FROM parsed_results_temp
    LATERAL VIEW explode(CAST(parsed_result:document:elements AS ARRAY<VARIANT>)) t AS elem
    GROUP BY source_file, elem:type::STRING
    ORDER BY source_file, count DESC
"""))

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
