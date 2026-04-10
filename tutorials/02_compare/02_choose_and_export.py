# Databricks notebook source
# MAGIC %md
# MAGIC # Choose Parser & Export
# MAGIC
# MAGIC **Time**: 2 minutes
# MAGIC **Prerequisites**: Run `02_compare/01_quality_comparison` first
# MAGIC
# MAGIC This notebook lets you pick a parser and export its output to a clean
# MAGIC `final_parsed_documents` table — the handoff point for downstream use
# MAGIC (chunking, indexing, RAG, etc.).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import re
from pyspark.sql.functions import col, count, avg, desc, round as spark_round

current_user = spark.sql("SELECT current_user()").first()[0]
username = re.sub(r"[^a-z0-9_]", "_", current_user.split("@")[0].lower()).strip("_")
username = re.sub(r"^[0-9]+", "", username) or "user"

dbutils.widgets.text("catalog_name", f"{username}_document_parsing", "Catalog")
dbutils.widgets.text("schema_name", "tutorials", "Schema")
dbutils.widgets.dropdown(
    "preferred_parser",
    "pymupdf",
    ["pymupdf", "ai_parse_document", "ai_query_vlm", "docling"],
    "Preferred Parser",
)

CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
PREFERRED = dbutils.widgets.get("preferred_parser")

COMPARISON_TABLE = f"{CATALOG}.{SCHEMA}.parser_comparison"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.final_parsed_documents"

print(f"Preferred parser: {PREFERRED}")
print(f"Output table:     {OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review Comparison Results

# COMMAND ----------

try:
    comp_df = spark.table(COMPARISON_TABLE)
    summary = (
        comp_df
        .groupBy("parse_method")
        .agg(
            count("*").alias("docs"),
            spark_round(avg("text_length"), 0).alias("avg_chars"),
            spark_round(avg("word_count"), 0).alias("avg_words"),
            spark_round(avg("parse_duration_seconds"), 2).alias("avg_time_s"),
        )
        .orderBy(desc("avg_chars"))
    )
    display(summary)
except Exception:
    print(f"Comparison table not found ({COMPARISON_TABLE})")
    print("Run 02_compare/01_quality_comparison first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Selected Parser Output

# COMMAND ----------

# Map parser name to source table
source_table_map = {
    "pymupdf": f"{CATALOG}.{SCHEMA}.parsed_pymupdf",
    "ai_parse_document": f"{CATALOG}.{SCHEMA}.parsed_ai_parse_document",
    "ai_query_vlm": f"{CATALOG}.{SCHEMA}.parsed_ai_query_vlm",
    "docling": f"{CATALOG}.{SCHEMA}.parsed_docling",
}

source_table = source_table_map.get(PREFERRED)
if not source_table:
    print(f"Unknown parser: {PREFERRED}")
    dbutils.notebook.exit(f"Unknown parser: {PREFERRED}")

try:
    parser_df = spark.table(source_table)
    row_count = parser_df.count()
    if row_count == 0:
        print(f"Table {source_table} is empty. Run the parser notebook first.")
        dbutils.notebook.exit("Empty source table")
except Exception:
    print(f"Table {source_table} not found. Run 01_parse/{PREFERRED} notebook first.")
    dbutils.notebook.exit("Source table not found")

print(f"Exporting {row_count} row(s) from {PREFERRED} to {OUTPUT_TABLE}")

# COMMAND ----------

# Write to final output table
parser_df.select(
    "source_file",
    "file_name",
    "page_number",
    "parsed_text",
    "contains_tables",
    "parse_method",
    "parse_duration_seconds",
    "estimated_cost_usd",
    "parsed_at",
).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUTPUT_TABLE)

final_count = spark.table(OUTPUT_TABLE).count()
print(f"Exported {final_count} row(s) to {OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print(f"Parser:       {PREFERRED}")
print(f"Documents:    {final_count}")
print(f"Output table: {OUTPUT_TABLE}")
print()
print("This table is ready for downstream use:")
print(f"  df = spark.table('{OUTPUT_TABLE}')")
print()
print("Common next steps:")
print("  - Chunk text for vector indexing")
print("  - Build a RAG pipeline")
print("  - Run further analysis or summarization")
