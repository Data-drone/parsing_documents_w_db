# Databricks notebook source
# MAGIC %md
# MAGIC # Parser Quality Comparison
# MAGIC
# MAGIC **Time**: 5–20 minutes (depends on LLM judge usage)
# MAGIC **Prerequisites**: Run at least 2 parser notebooks from `01_parse/`
# MAGIC
# MAGIC This is the centrepiece notebook. It compares all parsers you've run using
# MAGIC two tiers of evaluation:
# MAGIC
# MAGIC 1. **Deterministic metrics** (always run) — text length, word count, timing, cost
# MAGIC 2. **LLM-as-judge** (optional, selective) — send page image + parsed text to a
# MAGIC    vision model for quality scoring
# MAGIC
# MAGIC ## What you'll get
# MAGIC A comparison table showing every parser's performance across all metrics,
# MAGIC helping you decide which parser to use for your documents.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

from pyspark.sql.functions import (
    col, lit, length, size, split, avg, sum as spark_sum,
    count, when, round as spark_round, desc, current_timestamp,
    expr, concat, first,
)
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType

current_user = spark.sql("SELECT current_user()").first()[0]
username = current_user.split("@")[0].replace(".", "_")

dbutils.widgets.text("catalog_name", f"{username}_document_parsing", "Catalog")
dbutils.widgets.text("schema_name", "tutorials", "Schema")
dbutils.widgets.text("judge_endpoint", "databricks-claude-sonnet-4", "LLM Judge Endpoint")
dbutils.widgets.dropdown("run_llm_judge", "false", ["true", "false"], "Run LLM Judge")
dbutils.widgets.dropdown("demo_mode", "true", ["true", "false"], "Demo Mode")

CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
JUDGE_ENDPOINT = dbutils.widgets.get("judge_endpoint")
RUN_LLM_JUDGE = dbutils.widgets.get("run_llm_judge") == "true"
DEMO_MODE = dbutils.widgets.get("demo_mode") == "true"

OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.parser_comparison"

# Tables to look for (one per parser)
PARSER_TABLES = {
    "pymupdf": f"{CATALOG}.{SCHEMA}.parsed_pymupdf",
    "ai_parse_document": f"{CATALOG}.{SCHEMA}.parsed_ai_parse_document",
    "ai_query_vlm": f"{CATALOG}.{SCHEMA}.parsed_ai_query_vlm",
    "docling": f"{CATALOG}.{SCHEMA}.parsed_docling",
}

print(f"Catalog:     {CATALOG}")
print(f"Schema:      {SCHEMA}")
print(f"LLM Judge:   {'ON' if RUN_LLM_JUDGE else 'OFF'} ({JUDGE_ENDPOINT})")
print(f"Output:      {OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Parser Results
# MAGIC
# MAGIC Loads whichever parser output tables exist and unions them.

# COMMAND ----------

loaded = {}
union_df = None

for method, table_name in PARSER_TABLES.items():
    try:
        df = spark.table(table_name)
        row_count = df.count()
        if row_count > 0:
            loaded[method] = row_count
            if union_df is None:
                union_df = df
            else:
                union_df = union_df.unionByName(df, allowMissingColumns=True)
            print(f"  {method}: {row_count} row(s)")
        else:
            print(f"  {method}: table exists but is empty")
    except Exception:
        print(f"  {method}: table not found (skipped)")

if not loaded:
    print("\nNo parser results found. Run at least one notebook from 01_parse/ first.")
    dbutils.notebook.exit("No parser results to compare")

print(f"\nLoaded {len(loaded)} parser(s), {union_df.count()} total row(s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tier 1: Deterministic Metrics
# MAGIC
# MAGIC These metrics are always computed — no API calls needed.

# COMMAND ----------

# Add deterministic metrics columns
metrics_df = (
    union_df
    .withColumn("text_length", length(col("parsed_text")))
    .withColumn(
        "word_count",
        when(col("parsed_text").isNotNull(), size(split(col("parsed_text"), r"\s+")))
        .otherwise(lit(0)),
    )
)

# Per-method summary
summary_df = (
    metrics_df
    .groupBy("parse_method")
    .agg(
        count("*").alias("documents"),
        spark_round(avg("text_length"), 0).alias("avg_text_length"),
        spark_round(avg("word_count"), 0).alias("avg_word_count"),
        spark_round(avg("parse_duration_seconds"), 2).alias("avg_duration_s"),
        spark_round(spark_sum("estimated_cost_usd"), 4).alias("total_cost_usd"),
        spark_round(avg("estimated_cost_usd"), 4).alias("avg_cost_per_doc"),
        spark_sum(when(col("contains_tables"), 1).otherwise(0)).alias("docs_with_tables"),
    )
    .orderBy(desc("avg_text_length"))
)

print("Deterministic Metrics Summary")
print("=" * 60)
display(summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-Document Comparison
# MAGIC
# MAGIC Side-by-side metrics for each document across parsers.

# COMMAND ----------

per_doc_df = (
    metrics_df
    .select(
        "file_name", "parse_method", "text_length", "word_count",
        "parse_duration_seconds", "estimated_cost_usd", "contains_tables",
    )
    .orderBy("file_name", "parse_method")
)

display(per_doc_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tier 2: LLM-as-Judge (Optional)
# MAGIC
# MAGIC If enabled, sends original page images + parsed text to a vision model
# MAGIC for quality scoring. Only runs on a small sample to keep costs down.
# MAGIC
# MAGIC Set the `Run LLM Judge` widget to `true` to enable.

# COMMAND ----------

judge_results = None

if RUN_LLM_JUDGE:
    PAGE_IMAGES_TABLE = f"{CATALOG}.{SCHEMA}.document_page_images"

    try:
        pages_df = spark.table(PAGE_IMAGES_TABLE)
        print(f"Page images table found: {pages_df.count()} pages")
    except Exception:
        print(f"Page images table not found ({PAGE_IMAGES_TABLE})")
        print("LLM judge requires page images. Run 00_setup/02_prepare_documents first.")
        RUN_LLM_JUDGE = False

if RUN_LLM_JUDGE:
    import base64, pandas as pd
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import StringType

    # Get page-level parser results (only ai_query_vlm has page_number set)
    # For whole-doc parsers, we take page 1 image as representative
    sample_pages = (
        pages_df
        .filter(col("page_number") == 1)
        .select("file_name", "page_image_png")
        .limit(3 if DEMO_MODE else 10)
    )

    # For each sampled page, get parsed text from each parser
    judge_input_rows = []
    sample_files = [r.file_name for r in sample_pages.select("file_name").collect()]

    for method, table_name in PARSER_TABLES.items():
        if method not in loaded:
            continue
        try:
            parser_df = spark.table(table_name).filter(col("file_name").isin(sample_files))
            for row in parser_df.collect():
                judge_input_rows.append({
                    "file_name": row.file_name,
                    "parse_method": method,
                    "parsed_text": (row.parsed_text or "")[:2000],  # truncate for judge
                })
        except Exception:
            pass

    print(f"Prepared {len(judge_input_rows)} judge evaluations across {len(sample_files)} file(s)")

# COMMAND ----------

if RUN_LLM_JUDGE and judge_input_rows:
    # Create temp view with parsed text samples
    judge_df = spark.createDataFrame(pd.DataFrame(judge_input_rows))
    judge_df.createOrReplaceTempView("judge_inputs")

    # Join with page images for vision scoring
    judge_with_images = spark.sql(f"""
        SELECT
            j.file_name,
            j.parse_method,
            j.parsed_text,
            p.page_image_png
        FROM judge_inputs j
        JOIN {PAGE_IMAGES_TABLE} p
            ON j.file_name = p.file_name AND p.page_number = 1
    """)
    judge_with_images.createOrReplaceTempView("judge_with_images")

    judge_prompt = """You are evaluating the quality of text extracted from a document page.
Compare the parsed text against what you can see in the original page image.

Score each dimension from 1 (poor) to 5 (excellent):
- completeness: Is all visible text captured?
- accuracy: Is the extracted text correct (no hallucinations)?
- formatting: Is the structure preserved (headers, tables, lists)?

Return ONLY a JSON object like: {"completeness": 4, "accuracy": 5, "formatting": 3}"""

    print(f"Running LLM judge on {judge_with_images.count()} samples...")
    judge_scored = spark.sql(f"""
        SELECT
            file_name,
            parse_method,
            ai_query(
                '{JUDGE_ENDPOINT}',
                CONCAT('{judge_prompt}', '\\n\\nParsed text:\\n', parsed_text),
                failOnError => false,
                modelParameters => named_struct('max_tokens', 100, 'temperature', 0.0)
            ) AS judge_score_raw
        FROM judge_with_images
    """)

    judge_results = judge_scored.cache()
    display(judge_results)
    print("LLM judge scoring complete")

elif RUN_LLM_JUDGE:
    print("No judge input rows prepared — skipping")
else:
    print("LLM judge is OFF. Set 'Run LLM Judge' widget to 'true' to enable.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Comparison Results

# COMMAND ----------

# Save the deterministic metrics
metrics_df.select(
    "source_file", "file_name", "page_number", "parse_method",
    "text_length", "word_count", "contains_tables",
    "parse_duration_seconds", "estimated_cost_usd", "parsed_at",
).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUTPUT_TABLE)

saved_count = spark.table(OUTPUT_TABLE).count()
print(f"Saved {saved_count} comparison row(s) to {OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recommendation
# MAGIC
# MAGIC Quick decision guide based on the metrics above:

# COMMAND ----------

if len(loaded) >= 2:
    print("Parser Recommendation Guide:")
    print("=" * 50)
    print()

    # Find parser with most text (rough quality proxy)
    best_text = summary_df.orderBy(desc("avg_text_length")).first()
    print(f"  Most text extracted:  {best_text.parse_method} ({int(best_text.avg_text_length)} avg chars)")

    # Find fastest parser
    best_speed = summary_df.orderBy("avg_duration_s").first()
    print(f"  Fastest:              {best_speed.parse_method} ({best_speed.avg_duration_s}s avg)")

    # Find cheapest
    best_cost = summary_df.orderBy("total_cost_usd").first()
    print(f"  Cheapest:             {best_cost.parse_method} (${best_cost.total_cost_usd:.4f} total)")

    print()
    print("Next step: Run 02_compare/02_choose_and_export.py to export your preferred parser's output.")
else:
    print(f"Only 1 parser loaded ({list(loaded.keys())[0]}). Run more parsers for comparison.")
