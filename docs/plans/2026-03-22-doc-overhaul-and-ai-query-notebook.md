# Repo Documentation Overhaul, ai_parse_document & ai_query Notebooks

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the repo's documentation and tutorial flow, add two new parsing notebooks (`ai_parse_document` for native Databricks parsing and `ai_query + VLM` for Sonnet/Opus-based OCR), and clean up existing notebooks so the repo reads as a coherent developer guide rather than a personal notebook collection.

**Architecture:** The repo becomes a four-option tutorial (CPU fast → Databricks-native → managed VLM via ai_query → self-hosted VLM via vLLM) with a shared ingestion step and shared downstream indexing/agent steps. A new README provides the narrative backbone. Each notebook gets standardized config handling and consistent documentation style.

**Tech Stack:** Databricks notebooks (Python), PyMuPDF/PyMuPDF4LLM, ai_parse_document (Databricks SQL), ai_query (Databricks SQL), vLLM, Ray Data, Databricks Vector Search, LangGraph, MLflow

---

## Overview of Changes

**New files:**
- `interactive_examples/Parsing w ai_parse_document.py` — new notebook: native Databricks document parsing (SQL one-liner)
- `interactive_examples/Parsing w ai_query.py` — new notebook: VLM-OCR via Sonnet/Opus using ai_query()
- `README.md` — complete rewrite

**Modified files (cleanup):**
- `interactive_examples/Create Document Database.py` — parameterize config
- `interactive_examples/Parsing Documents.py` — parameterize config
- `interactive_examples/Split Documents.py` — parameterize config
- `interactive_examples/Deploy LLM Server.py` — update model reference, parameterize
- `interactive_examples/Parsing w OpenAI API.py` — parameterize config
- `interactive_examples/Parsing_w_ray.py` — fix max_tokens bug, parameterize config
- `interactive_examples/Create Document Summaries.py` — parameterize config
- `interactive_examples/Create Document Summary Index.py` — parameterize config, fix ID generation
- `interactive_examples/Create Page Index.py` — parameterize config
- `interactive_examples/Building A Compound Chain.py` — fix system prompt domain mismatch, parameterize config
- `interactive_examples/Deploy A Compound Chain.py` — parameterize config

**Files to consider removing or marking legacy:**
- `interactive_examples/Parsing Documents - OCR.py` — uses outdated Tesseract + langchain 0.2.17
- `interactive_examples/Parsing Documents - VLM.py` — incomplete scratch notebook
- `interactive_examples/Exploring LLM Parsing in DBX.py` — sends base64 to non-vision LLM (broken approach)

---

## Task 1: Standardize Configuration Across All Notebooks

All notebooks currently hardcode `CATALOG = "brian_gen_ai"` and `SCHEMA = "parsing_test"`. They should use widgets so they work for any user and integrate with the DAB config in `databricks.yml`.

**Files to modify:**
- All notebooks in `interactive_examples/`

**Step 1: Define the standard config block**

Every notebook should start (after pip installs / restart) with this pattern:

```python
# COMMAND ----------
# DBTITLE 1,Configuration

# Use widgets for parameterized execution (DAB workflows set these automatically)
dbutils.widgets.text("catalog_name", "")
dbutils.widgets.text("schema_name", "parsing_test")
dbutils.widgets.text("volume_name", "raw_data")

# Resolve catalog: widget value if set, otherwise user-scoped default
_cat = dbutils.widgets.get("catalog_name")
if not _cat:
    import re
    _user = spark.sql("SELECT current_user()").first()[0].split("@")[0]
    _user = re.sub(r"[^a-z0-9]", "_", _user.lower()).strip("_")
    _cat = f"{_user}_parsing"

CATALOG = _cat
SCHEMA = dbutils.widgets.get("schema_name")
VOLUME = dbutils.widgets.get("volume_name")

print(f"Using: {CATALOG}.{SCHEMA}")
```

> **Note on import path:** The existing repo pattern uses `from src.utils.databricks_utils import get_username_from_email` in some notebooks, but this depends on `sys.path` manipulation and doesn't work consistently across interactive vs workflow contexts. The `spark.sql("SELECT current_user()")` approach above is self-contained — no imports, works everywhere (interactive, workflows, serverless). This replaces the need for the `get_username_from_email` helper entirely in interactive notebooks.
>
> **Note on username sanitization:** The `re.sub` strips everything except `[a-z0-9_]` from the username, lowercases, and strips leading/trailing underscores. This handles dots, hyphens, plus signs, uppercase, and any other characters that are invalid in Unity Catalog names. Service principals may return non-email identifiers — if `catalog_name` widget is empty and the derived name looks wrong, set the widget explicitly.

**Step 2: Apply this pattern to each notebook**

Go through each notebook listed above and replace the hardcoded config block with the standard one. Remove any `brian_gen_ai` / `brian_serving_test` / `nicholas_anile` references.

For notebooks that have extra config (e.g., `MODEL_NAME`, `LLM_ENDPOINT_NAME`), keep those as additional variables below the standard block.

**Step 3: Commit**

```bash
git add interactive_examples/*.py
git commit -m "refactor: standardize config with widgets across all notebooks"
```

---

## Task 2: Write the New `Parsing w ai_parse_document` Notebook

This is the simplest possible parsing path — a single Databricks SQL function that handles PDF parsing natively. No UDFs, no model servers, no image splitting. Requires DBR 17.1+.

**File:** `interactive_examples/Parsing w ai_parse_document.py`

**Step 1: Create the notebook**

The notebook should follow this structure:

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # Document Parsing with ai_parse_document
# MAGIC
# MAGIC This notebook demonstrates Databricks' native document parsing function
# MAGIC `ai_parse_document`. This is the simplest path — one SQL call replaces
# MAGIC the entire PyMuPDF / split / VLM pipeline.
# MAGIC
# MAGIC ## When to Use This Approach
# MAGIC - You want zero-setup document parsing (no libraries, no GPU, no model server)
# MAGIC - You're on DBR 17.1+ with Unity Catalog
# MAGIC - You need table, figure, and layout extraction out of the box
# MAGIC - You want Databricks-governed lineage and cost tracking
# MAGIC
# MAGIC ## What It Handles
# MAGIC - Tables (including merged/nested cells)
# MAGIC - Figures with AI-generated captions
# MAGIC - Bounding boxes and spatial metadata
# MAGIC - Multi-format: PDF, images, Office documents
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - DBR 17.1+
# MAGIC - Unity Catalog enabled workspace
# MAGIC - Files in a Unity Catalog Volume
# MAGIC
# MAGIC ## Limitations
# MAGIC - Databricks-managed — you can't swap in a different model
# MAGIC - Pricing is per-page (check current Databricks pricing)
# MAGIC - Less control over parsing prompts vs ai_query approach

# COMMAND ----------

# DBTITLE 1,Configuration
dbutils.widgets.text("catalog_name", "")
dbutils.widgets.text("schema_name", "parsing_test")
dbutils.widgets.text("volume_name", "raw_data")

_cat = dbutils.widgets.get("catalog_name")
if not _cat:
    import re
    _user = spark.sql("SELECT current_user()").first()[0].split("@")[0]
    _user = re.sub(r"[^a-z0-9]", "_", _user.lower()).strip("_")
    _cat = f"{_user}_parsing"

CATALOG = _cat
SCHEMA = dbutils.widgets.get("schema_name")
VOLUME = dbutils.widgets.get("volume_name")

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.document_ai_parsed"

print(f"Volume: {VOLUME_PATH}")
print(f"Output: {OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Check Prerequisites
# MAGIC
# MAGIC Verify we're on a compatible runtime and can access the volume.

# COMMAND ----------

# Check DBR version — requires 17.1+
dbr_version = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion", "unknown")
print(f"DBR version: {dbr_version}")

# List files in volume using dbutils (works across all compute modes)
all_files = dbutils.fs.ls(VOLUME_PATH)
files = [f.name for f in all_files if f.name.lower().endswith('.pdf')]
print(f"PDF files found: {len(files)}")
for f in files[:10]:
    print(f"  - {f}")

if not files:
    raise ValueError(f"No PDF files found in {VOLUME_PATH}. Upload PDFs to the volume first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Parse a Single Document
# MAGIC
# MAGIC Test `ai_parse_document` on one file. The function takes a `BINARY`
# MAGIC content column (from `READ_FILES` with `format => 'binaryFile'`) and
# MAGIC returns a `VARIANT` with structured document elements.

# COMMAND ----------

sample_file = files[0]
result = spark.sql(f"""
    SELECT
        path,
        ai_parse_document(
            content,
            map('version', '2.0')
        ) AS parsed
    FROM READ_FILES(
        '{VOLUME_PATH}/{sample_file}',
        format => 'binaryFile'
    )
""")
display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Explore the Output Schema
# MAGIC
# MAGIC `ai_parse_document` returns a VARIANT with this structure:
# MAGIC ```
# MAGIC {
# MAGIC   "document": {
# MAGIC     "pages": [{"id": INT, "image_uri": STRING}],
# MAGIC     "elements": [{
# MAGIC       "id": INT,
# MAGIC       "type": STRING,       -- text, table, figure, title, caption, section_header, etc.
# MAGIC       "content": STRING,    -- text content; HTML for tables; NULL for figures
# MAGIC       "bbox": [...],        -- bounding box coordinates
# MAGIC       "description": STRING -- AI-generated description (figures)
# MAGIC     }]
# MAGIC   },
# MAGIC   "error_status": [{"error_message": STRING, "page_id": INT}],
# MAGIC   "metadata": {"id": STRING, "version": STRING, "file_metadata": {...}}
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC Element types: `text`, `table` (HTML), `figure`, `title`, `caption`,
# MAGIC `section_header`, `page_header`, `page_footer`, `page_number`, `footnote`
# MAGIC
# MAGIC > **VARIANT handling note:** `ai_parse_document` returns a `VARIANT`.
# MAGIC > Use `:` path notation to navigate fields (e.g. `parsed:document:elements`).
# MAGIC > If `FILTER`/`TRANSFORM`/`explode` fail on VARIANT arrays, cast first:
# MAGIC > `CAST(parsed:document:elements AS ARRAY<STRUCT<id:INT, type:STRING, content:STRING, description:STRING>>)`
# MAGIC > Test on your DBR version — VARIANT higher-order function support may vary.

# COMMAND ----------

# Inspect the parsed output — VARIANT path notation
display(result.selectExpr(
        "parsed:metadata:file_metadata:file_name::STRING AS file_name",
        "parsed:metadata:version::STRING AS schema_version",
        "size(parsed:document:elements) AS num_elements",
        "size(parsed:document:pages) AS num_pages",
        "size(parsed:error_status) AS num_errors"
))

# COMMAND ----------

# Show individual elements (first 20)
# If explode on VARIANT fails, use the CAST fallback in the comment below
elements_df = spark.sql(f"""
        SELECT
            elem:id::INT AS element_id,
            elem:type::STRING AS element_type,
            SUBSTRING(elem:content::STRING, 1, 200) AS content_preview,
            elem:description::STRING AS description
        FROM (
            SELECT explode(parsed:document:elements) AS elem
            FROM {OUTPUT_TABLE}
            WHERE source_file LIKE '%{sample_file}%'
        )
        -- Fallback if explode on VARIANT fails:
        -- SELECT explode(CAST(parsed:document:elements AS ARRAY<STRUCT<id:INT, type:STRING, content:STRING, description:STRING>>)) AS elem
        LIMIT 20
    """)
    display(elements_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Batch Parse All Documents
# MAGIC
# MAGIC Parse every PDF in the volume. `ai_parse_document` handles
# MAGIC parallelization and retries automatically — submit the full dataset
# MAGIC in one query.

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE TABLE {OUTPUT_TABLE} AS
    SELECT
        path AS source_file,
        ai_parse_document(
            content,
            map('version', '2.0')
        ) AS parsed,
        current_timestamp() AS parsed_at
    FROM READ_FILES(
        '{VOLUME_PATH}',
        format => 'binaryFile',
        pathGlobFilter => '*.pdf',
        recursiveFileLookup => 'true'
    )
""")

total = spark.table(OUTPUT_TABLE).count()
print(f"Parsed {total} documents -> saved to {OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Extract Text Content for Downstream Use
# MAGIC
# MAGIC Flatten the parsed elements into a format compatible with the rest
# MAGIC of the pipeline (summaries, vector indexing, agent). We concatenate
# MAGIC all text-type elements per document into a single markdown string.

# COMMAND ----------

# Approach: explode elements, then re-aggregate by document.
# This avoids FILTER/TRANSFORM on VARIANT arrays (which may not work on all DBR versions).
display(spark.sql(f"""
    WITH elements AS (
        SELECT
            source_file,
            parsed:metadata:file_metadata:file_name::STRING AS file_name,
            explode(parsed:document:elements) AS elem,
            size(parsed:error_status) AS errors,
            parsed_at
        FROM {OUTPUT_TABLE}
    )
    SELECT
        source_file,
        file_name,
        -- Concatenate text-type elements
        CONCAT_WS('\n\n',
            COLLECT_LIST(
                CASE WHEN elem:type::STRING IN ('text', 'title', 'section_header', 'caption')
                     THEN elem:content::STRING END
            )
        ) AS text_content,
        -- Tables (HTML)
        COLLECT_LIST(
            CASE WHEN elem:type::STRING = 'table' THEN elem:content::STRING END
        ) AS tables_html,
        -- Figure descriptions
        COLLECT_LIST(
            CASE WHEN elem:type::STRING = 'figure' THEN elem:description::STRING END
        ) AS figure_descriptions,
        COUNT(*) AS total_elements,
        MAX(errors) AS errors,
        MAX(parsed_at) AS parsed_at
    FROM elements
    GROUP BY source_file, file_name
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Compare with PyMuPDF4LLM Output
# MAGIC
# MAGIC If you've already run the PyMuPDF4LLM notebook, compare outputs.

# COMMAND ----------

# Compare: extract filename from the full path and join on that
comparison = spark.sql(f"""
    WITH ai_elements AS (
        SELECT
            source_file,
            -- Extract filename from full Volume path for joining
            REGEXP_EXTRACT(source_file, '[^/]+$') AS file_name,
            explode(parsed:document:elements) AS elem
        FROM {OUTPUT_TABLE}
    ),
    ai_text AS (
        SELECT
            source_file,
            file_name,
            CONCAT_WS('\n\n',
                COLLECT_LIST(
                    CASE WHEN elem:type::STRING IN ('text', 'title', 'section_header')
                         THEN elem:content::STRING END
                )
            ) AS ai_parse_text
        FROM ai_elements
        GROUP BY source_file, file_name
    )
    SELECT
        a.file_name,
        SUBSTRING(a.ai_parse_text, 1, 500) AS ai_parse_extract,
        SUBSTRING(m.markdown_content, 1, 500) AS pymupdf_extract,
        LENGTH(a.ai_parse_text) AS ai_parse_chars,
        LENGTH(m.markdown_content) AS pymupdf_chars
    FROM ai_text a
    LEFT JOIN {CATALOG}.{SCHEMA}.document_markdown m
        ON a.file_name = m.file_name
    LIMIT 5
""")
display(comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC The parsed output feeds directly into the same downstream pipeline:
# MAGIC - `Create Document Summaries` → AI summarization
# MAGIC - `Create Document Summary Index` / `Create Page Index` → Vector search
# MAGIC - `Building A Compound Chain` → RAG agent
# MAGIC
# MAGIC For more control over the parsing prompt (custom instructions,
# MAGIC specific output formats), see `Parsing w ai_query` which uses
# MAGIC Claude Sonnet/Opus directly.
```

**Step 2: Commit**

```bash
git add "interactive_examples/Parsing w ai_parse_document.py"
git commit -m "feat: add ai_parse_document notebook — native Databricks parsing"
```

---

## Task 3: Write the New `Parsing w ai_query` Notebook (VLM)

This is the new notebook that shows using Sonnet/Opus (or any FMAPI-hosted VLM) to OCR document pages via `ai_query()`.

**File:** `interactive_examples/Parsing w ai_query.py`

**Step 1: Create the notebook**

The notebook should follow this structure:

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # Document Parsing with ai_query + Vision LLMs
# MAGIC
# MAGIC This notebook demonstrates using Databricks Foundation Model API endpoints
# MAGIC (Claude Sonnet, Claude Opus, or similar vision-capable models) to extract
# MAGIC text from document page images via `ai_query()`.
# MAGIC
# MAGIC ## When to Use This Approach
# MAGIC - You need higher quality than PyMuPDF4LLM (scanned docs, complex tables, figures)
# MAGIC - You don't want to manage GPU infrastructure (no vLLM server needed)
# MAGIC - Your batch size is moderate (hundreds to low thousands of pages)
# MAGIC - You want to leverage managed model serving endpoints
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Run `Create Document Database` first to ingest your PDFs
# MAGIC - Run `Split Documents` to create page images
# MAGIC - A Foundation Model API endpoint with a vision-capable model
# MAGIC
# MAGIC ## Cost Considerations
# MAGIC - Each page image ≈ 1-2K input tokens (depending on resolution)
# MAGIC - Output ≈ 500-2000 tokens per page
# MAGIC - At Sonnet pricing, expect ~$0.01-0.03 per page
# MAGIC - For 10K+ pages, consider the self-hosted VLM path instead

# COMMAND ----------

# MAGIC %pip install pillow
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Configuration
dbutils.widgets.text("catalog_name", "")
dbutils.widgets.text("schema_name", "parsing_test")
dbutils.widgets.text("volume_name", "raw_data")
dbutils.widgets.text("vlm_endpoint", "databricks-claude-sonnet-4-5")

_cat = dbutils.widgets.get("catalog_name")
if not _cat:
    import re
    _user = spark.sql("SELECT current_user()").first()[0].split("@")[0]
    _user = re.sub(r"[^a-z0-9]", "_", _user.lower()).strip("_")
    _cat = f"{_user}_parsing"

CATALOG = _cat
SCHEMA = dbutils.widgets.get("schema_name")

# Source: page images from Split Documents notebook
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.document_page_docs"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.document_vlm_parsed"

# Model endpoint — use any vision-capable FMAPI endpoint
# Vision-confirmed endpoints: databricks-claude-opus-4-5, databricks-claude-opus-4-1,
#   databricks-llama-4-maverick, databricks-gemini-2-5-pro, databricks-gpt-5-4
VLM_ENDPOINT = dbutils.widgets.get("vlm_endpoint")

print(f"Source: {SOURCE_TABLE}")
print(f"Output: {OUTPUT_TABLE}")
print(f"Model: {VLM_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Explore the Page Images
# MAGIC
# MAGIC Let's look at what we're working with from the Split Documents step.

# COMMAND ----------

pages_df = spark.table(SOURCE_TABLE)

print(f"Total pages: {pages_df.count():,}")
print(f"Documents: {pages_df.select('doc_id').distinct().count():,}")

pages_df.select(
    "doc_id", "source_filename", "page_number", "total_pages", "file_size_bytes"
).show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Test on a Single Page
# MAGIC
# MAGIC Before running at scale, let's test the prompt and model on one page
# MAGIC to verify quality and tune the extraction prompt.

# COMMAND ----------

import base64
import io
from PIL import Image

# Grab one sample page
sample = pages_df.limit(1).collect()[0]
img_bytes = sample["page_image_png"]
filename = sample["source_filename"]
page_num = sample["page_number"]

print(f"Testing with: {filename}, page {page_num}")
print(f"Image size: {len(img_bytes):,} bytes")

# Preview the image
img = Image.open(io.BytesIO(img_bytes))
display(img)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Single-page test with ai_query (SQL)
# MAGIC
# MAGIC We use `ai_query()` with the `files =>` parameter which natively
# MAGIC passes binary image data to a vision-capable endpoint. No base64
# MAGIC encoding needed — Databricks handles it.

# COMMAND ----------

# Test on one page — use a simple prompt without special characters for SQL safety
sample_result = spark.sql(f"""
    SELECT
        doc_id,
        source_filename,
        page_number,
        ai_query(
            '{VLM_ENDPOINT}',
            'Extract all text content from this document page as clean markdown. Preserve headings, paragraphs, lists, and tables. Do NOT describe the image. Only extract the text content.',
            files => page_image_png,
            failOnError => false
        ) AS parsed_text
    FROM {SOURCE_TABLE}
    LIMIT 1
""")

display(sample_result)

# Show extracted text
rows = sample_result.collect()
if rows:
    row = rows[0]
    print(f"File: {row['source_filename']}, Page: {row['page_number']}")
    text = row['parsed_text'] or 'No output'
    print(f"Extracted {len(text)} characters:")
    print("=" * 60)
    print(text[:2000])
else:
    print("No pages found in source table. Run Split Documents first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Batch Processing with ai_query (SQL)
# MAGIC
# MAGIC `ai_query()` automatically handles parallelization, retries, and
# MAGIC scaling. Submit the full dataset in a single query rather than
# MAGIC manually splitting into batches.
# MAGIC
# MAGIC This is the recommended approach — no Python UDF needed, no
# MAGIC `spark`/`dbutils` serialization issues, and Databricks manages
# MAGIC the concurrency for you.

# COMMAND ----------

import time

OCR_PROMPT = (
    "Extract all text content from this document page as clean markdown. "
    "Preserve the document structure (headings, paragraphs, lists). "
    "Format tables as markdown tables. "
    "Format equations as LaTeX (wrapped in dollar signs). "
    "Do NOT describe the image - only extract the text content. "
    "If a section is illegible, mark it as [illegible]."
)

# Escape single quotes for safe SQL interpolation
_prompt_sql = OCR_PROMPT.replace("'", "''")

print(f"Processing all pages from {SOURCE_TABLE} with {VLM_ENDPOINT}...")
start_time = time.time()

spark.sql(f"""
    CREATE OR REPLACE TABLE {OUTPUT_TABLE} AS
    SELECT
        doc_id,
        source_filename,
        page_number,
        total_pages,
        ai_query(
            '{VLM_ENDPOINT}',
            '{_prompt_sql}',
            files => page_image_png,
            failOnError => false
        ) AS parsed_markdown,
        '{VLM_ENDPOINT}' AS vlm_model,
        current_timestamp() AS parsed_at
    FROM {SOURCE_TABLE}
""")

elapsed = time.time() - start_time
total = spark.table(OUTPUT_TABLE).count()
successful = spark.sql(f"""
    SELECT count(*) FROM {OUTPUT_TABLE}
    WHERE parsed_markdown IS NOT NULL
""").collect()[0][0]

print(f"Done in {elapsed:.1f}s")
print(f"Total pages: {total:,}")
print(f"Successful: {successful:,} ({successful/total*100 if total else 0:.1f}%)")
if elapsed > 0:
    print(f"Throughput: {total/elapsed:.1f} pages/sec")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Alternative: Python REST API Approach
# MAGIC
# MAGIC If you need more control (custom retry logic, streaming, etc.),
# MAGIC you can call the serving endpoint directly. See `Parsing w OpenAI API`
# MAGIC for a production-grade Python approach with adaptive concurrency.
# MAGIC
# MAGIC The SQL `ai_query()` approach above is recommended for most use cases
# MAGIC because Databricks handles parallelization and retries automatically.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Compare Quality: PyMuPDF4LLM vs ai_query VLM
# MAGIC
# MAGIC Let's compare the output of basic text extraction vs VLM parsing
# MAGIC on the same document to see the quality difference.

# COMMAND ----------

# Compare page 1 across methods: PyMuPDF4LLM vs VLM
# Note: requires Parsing Documents notebook to have been run first (creates document_markdown table)
comparison_df = spark.sql(f"""
    SELECT
        v.source_filename,
        v.page_number,
        SUBSTRING(m.markdown_content, 1, 500) AS pymupdf_extract,
        SUBSTRING(v.parsed_markdown, 1, 500) AS vlm_extract,
        LENGTH(m.markdown_content) AS pymupdf_chars,
        LENGTH(v.parsed_markdown) AS vlm_chars
    FROM {OUTPUT_TABLE} v
    LEFT JOIN {CATALOG}.{SCHEMA}.document_markdown m
        ON v.source_filename = m.file_name
    WHERE v.page_number = 1
        AND v.parsed_markdown IS NOT NULL
    LIMIT 5
""")
display(comparison_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Cost Estimation
# MAGIC
# MAGIC Rough cost estimate based on token usage.

# COMMAND ----------

from pyspark.sql.functions import col, avg, length

stats = spark.sql(f"""
    SELECT
        count(*) AS total_pages,
        avg(length(parsed_markdown)) AS avg_output_chars
    FROM {OUTPUT_TABLE}
    WHERE parsed_markdown IS NOT NULL
""").collect()[0]

total_pages = stats["total_pages"]
avg_chars = stats["avg_output_chars"] or 0

# Rough estimates
est_input_tokens_per_page = 1500  # image + prompt
est_output_tokens_per_page = avg_chars / 4 if avg_chars else 1000

total_input = total_pages * est_input_tokens_per_page
total_output = total_pages * est_output_tokens_per_page

print(f"Cost Estimate for {total_pages:,} pages (avg {avg_chars:.0f} chars/page):")
print(f"  Est. input tokens:  {total_input:,.0f}")
print(f"  Est. output tokens: {total_output:,.0f}")
print()
print(f"  Sonnet 4.5 (~$3/$15 per M tokens):")
print(f"    Total: ${total_input / 1e6 * 3 + total_output / 1e6 * 15:.2f}")
print()
print(f"  Opus 4.5 (~$15/$75 per M tokens):")
print(f"    Total: ${total_input / 1e6 * 15 + total_output / 1e6 * 75:.2f}")
print()
print("  Note: ai_query billing is under AI_FUNCTIONS — check Databricks pricing for exact rates.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC - For **higher throughput at lower cost**, use the self-hosted VLM path:
# MAGIC   `Deploy LLM Server` → `Parsing w OpenAI API` → `Parsing w Ray`
# MAGIC - For **indexing and search**, proceed to:
# MAGIC   `Create Document Summaries` → `Create Document Summary Index` / `Create Page Index`
# MAGIC - For **a complete RAG agent**, see:
# MAGIC   `Building A Compound Chain` → `Deploy A Compound Chain`
```

**Step 2: Verify the file was created correctly**

Open it, scan for syntax issues, confirm it's a valid Databricks notebook format.

**Step 3: Commit**

```bash
git add "interactive_examples/Parsing w ai_query.py"
git commit -m "feat: add ai_query + Sonnet/Opus VLM parsing notebook"
```

---

## Task 4: Fix Known Bugs in Existing Notebooks

**Step 1: Fix `Parsing_w_ray.py` — max_tokens too low**

File: `interactive_examples/Parsing_w_ray.py`
Line ~218: `sampling_params=dict(max_tokens=128)` → change to `max_tokens=4096`

This is a real bug — 128 tokens truncates almost all OCR output.

**Step 2: Fix `Building A Compound Chain.py` — system prompt domain mismatch**

File: `interactive_examples/Building A Compound Chain.py`
The `BASE_SYSTEM_PROMPT_TEMPLATE` references "Australian listed company documentation and regulatory filings" but this should be generic since the repo is a general-purpose guide. Replace the domain-specific prompt with a generic document analysis prompt that works with any document collection.

Replace the system prompt template (lines ~85-170) with:

```python
BASE_SYSTEM_PROMPT_TEMPLATE = """You are a document analysis assistant with access to a parsed document archive.

## Current Context
- Today: {current_date} ({current_day})
- Quarter: {current_quarter}

## Search Strategy
You have access to two complementary search tools:

1. **Document Summary Search**: Searches high-level document summaries to identify relevant documents
2. **Detailed Content Search**: Searches page-level OCR text for specific content within identified documents

**Search Protocol:**
- Always start with the summary search to identify relevant documents
- Use the detailed search to find specific information within those documents
- Cross-reference information across multiple documents when available

## Temporal Intelligence
When users reference time:
- **"Recent"** = Within the last 30 days from today ({current_date})
- **"Latest"** = Most recent available documents as of {current_date}

## Response Guidelines
- Cite which documents and pages your information comes from
- If information spans multiple documents, synthesize across them
- Be transparent about what the documents do and don't cover
- Include relevant caveats about document age or completeness"""
```

**Step 3: Fix `Create Document Summary Index.py` — unstable IDs**

File: `interactive_examples/Create Document Summary Index.py`
Line ~87-90: Replace `monotonically_increasing_id()` with a deterministic hash:

```python
from pyspark.sql.functions import sha2, concat, col, lit

df_flattened_clean = df_flattened_clean.withColumn(
    "id",
    sha2(concat(col("file_name"), lit("_"), col("file_path")), 256)
)
```

**Step 4: Commit**

```bash
git add interactive_examples/Parsing_w_ray.py
git add "interactive_examples/Building A Compound Chain.py"
git add "interactive_examples/Create Document Summary Index.py"
git commit -m "fix: max_tokens bug, system prompt domain, unstable IDs"
```

---

## Task 5: Mark Legacy Notebooks

Rather than deleting, add a clear deprecation notice to notebooks that are outdated.

**Step 1: Add deprecation header to `Parsing Documents - OCR.py`**

Add at the top of the notebook (first markdown cell):

```
# MAGIC %md
# MAGIC > **LEGACY NOTEBOOK** — This uses Tesseract OCR which has been superseded by VLM-based
# MAGIC > parsing (see `Parsing w ai_query` or `Parsing w OpenAI API`). Kept for reference only.
```

**Step 2: Add deprecation header to `Parsing Documents - VLM.py`**

```
# MAGIC %md
# MAGIC > **LEGACY NOTEBOOK** — Incomplete exploration notebook. For VLM-based parsing,
# MAGIC > see `Parsing w ai_query` (managed) or `Parsing w OpenAI API` (self-hosted).
```

**Step 3: Add deprecation header to `Exploring LLM Parsing in DBX.py`**

```
# MAGIC %md
# MAGIC > **LEGACY NOTEBOOK** — Exploration of LLM-based parsing approaches. For current
# MAGIC > approaches, see `Parsing w ai_query` or `Parsing w OpenAI API`.
```

**Step 4: Commit**

```bash
git add "interactive_examples/Parsing Documents - OCR.py"
git add "interactive_examples/Parsing Documents - VLM.py"
git add "interactive_examples/Exploring LLM Parsing in DBX.py"
git commit -m "docs: mark legacy notebooks with deprecation notices"
```

---

## Task 6: Rewrite the README

**File:** `README.md`

**Step 1: Write the new README**

```markdown
# Parsing Documents with Databricks

A developer guide to building document parsing pipelines for RAG on Databricks.
Covers the full flow: ingestion → parsing → indexing → retrieval agent.

## Quick Start

Choose your parsing path based on your documents and infrastructure:

| Path | Best For | Requires |
|------|----------|----------|
| **PyMuPDF4LLM** | Text-layer PDFs (Word exports, LaTeX) | CPU cluster |
| **ai_parse_document** | General-purpose, zero-setup | DBR 17.1+, Unity Catalog |
| **ai_query + VLM** | Custom prompts, scanned docs, figures | FMAPI endpoint (no GPU) |
| **Self-hosted VLM** | High volume, cost-sensitive | GPU cluster |

All paths share the same ingestion and downstream steps (indexing, agent).

## Prerequisites

- Databricks workspace with Unity Catalog
- DBR 15.4+ (DBR 17.1+ for `ai_parse_document`)
- For VLM paths: a Foundation Model API endpoint with vision support (e.g. Claude Sonnet)
- For self-hosted VLM: GPU cluster (single A10/T4 for dev, multi-GPU for production)

## Setup

```bash
# Clone and deploy with Databricks Asset Bundles
databricks bundle deploy -t <your_target>
```

Or run the notebooks interactively — each one is self-contained.

## Tutorial Sequence

### Step 1: Ingest Documents

`interactive_examples/Create Document Database`

Recursively scans a Unity Catalog Volume for PDF/DOC files, reads binary content,
and stores everything in a Delta table. This is the starting point for all paths.

### Step 2: Extract Text (Choose One)

**Option A: PyMuPDF4LLM (fastest, CPU-only)**

`interactive_examples/Parsing Documents`

Converts PDFs to markdown using PyMuPDF4LLM. Works great for digitally-created PDFs
with an embedded text layer. No ML models, no GPU. Start here.

**Option B: ai_parse_document (zero-setup, Databricks-native)**

`interactive_examples/Parsing w ai_parse_document`

One SQL call parses entire documents — tables, figures, layout, bounding boxes.
No libraries to install, no model to manage. Requires DBR 17.1+ and Unity Catalog.
Best option if you want to get started fast and don't need custom parsing prompts.

**Option C: ai_query + Vision LLM (custom prompts, no GPU infra)**

`interactive_examples/Split Documents` → `interactive_examples/Parsing w ai_query`

First splits PDFs into page images, then sends each page to a vision-capable
model (Claude Sonnet/Opus) via the Foundation Model API. Higher quality than
PyMuPDF4LLM for scanned docs, complex tables, and figures. Full control over
the extraction prompt. Pay-per-token cost.

**Option D: Self-hosted VLM (best throughput, GPU required)**

`interactive_examples/Split Documents` → `interactive_examples/Deploy LLM Server` → `interactive_examples/Parsing w OpenAI API`

Stands up a vLLM server with a dedicated OCR model and calls it via the
OpenAI-compatible API. Best for large batches where you want to control cost.
Includes adaptive concurrency and backpressure handling.

To scale across a multi-GPU cluster:

`interactive_examples/Parsing w Ray`

Uses Ray Data + vLLM for distributed inference across multiple GPUs.

### Note: Output Normalization

Each parsing path produces a different output schema:
- **PyMuPDF4LLM** → `document_markdown` (one row per document, `markdown_content` column)
- **ai_parse_document** → `document_ai_parsed` (VARIANT with structured elements)
- **ai_query** → `document_vlm_parsed` (one row per page, `parsed_markdown` column)
- **Self-hosted VLM** → `document_store_ocr` (one row per page, OCR text)

The downstream steps (summaries, indexing, agent) expect text content. If you're switching
between parsing paths, you may need to create a view or table that normalizes the output
into a common schema (e.g., `file_name`, `page_number`, `text_content`). Each parsing
notebook's "Next Steps" section notes what format its output is in.

### Step 3: Summarize Documents

`interactive_examples/Create Document Summaries`

Uses `ai_query()` with an LLM to generate document-level summaries and analysis.
Creates a two-tier strategy: full analysis for small docs, truncated for large ones.

### Step 4: Create Vector Indexes

`interactive_examples/Create Document Summary Index` — index on document summaries
`interactive_examples/Create Page Index` — index on page-level OCR text

Both use Databricks Vector Search with GTE embeddings. The agent uses both indexes
for two-stage retrieval (find relevant docs → search within those docs).

### Step 5: Build and Deploy the RAG Agent

`interactive_examples/Building A Compound Chain`

A LangGraph agent with two-stage search: summary index for document discovery,
OCR index for detailed content retrieval. Includes temporal awareness and
dynamic system prompt injection.

`interactive_examples/Deploy A Compound Chain`

Logs the agent to MLflow, runs evaluation with Mosaic AI judges, registers in
Unity Catalog, and deploys via Model Serving.

## Architecture

```
UC Volume (PDFs)
    │
    ▼
┌───────────────────────┐
│  Create Document DB    │  ← Step 1: Ingest
└─────────┬─────────────┘
          │
    ┌─────┼────────────┬──────────────────┐
    ▼     ▼            ▼                  ▼
 PyMuPDF  ai_parse_  Split Pages      Split Pages
 4LLM     document   + ai_query       + vLLM/Ray
 (CPU)    (SQL)      (managed VLM)    (self-hosted)
    │     │            │                  │
    └─────┼────────────┼──────────────────┘
          ▼
┌───────────────────────┐
│  Document Summaries    │  ← Step 3
└─────────┬─────────────┘
          ▼
┌───────────────────────┐
│  Vector Indexes        │  ← Step 4
│  (summary + page)      │
└─────────┬─────────────┘
          ▼
┌───────────────────────┐
│  LangGraph Agent       │  ← Step 5
│  (two-stage search)    │
└───────────────────────┘
```

## Data Model

| Table | Created By | Contents |
|-------|-----------|----------|
| `document_store` | Create Document Database | File metadata + binary content |
| `document_markdown` | Parsing Documents | PyMuPDF4LLM markdown extraction |
| `document_ai_parsed` | Parsing w ai_parse_document | Native Databricks parsed output |
| `document_page_docs` | Split Documents | Per-page PNG images |
| `document_vlm_parsed` | Parsing w ai_query | VLM-extracted markdown per page |
| `document_store_ocr` | Parsing w OpenAI API | Self-hosted VLM OCR text per page |
| `parsed_markdown_pages` | Parsing w Ray | Ray-processed VLM OCR text |
| `document_analysis_simple` | Create Document Summaries | AI-generated document summaries |
| `flattened_documents` | Create Document Summary Index | Flattened summaries for vector search |
| `document_ocr_vector_ready` | Create Page Index | Cleaned OCR text for vector search |

## Databricks Asset Bundles

The repo includes DAB configuration for automated workflows:

- `src/workflows/parsing_workflow.yml` — CPU parsing pipeline (ingest → PyMuPDF4LLM)
- `src/workflows/ocr_parsing_workflow.yml` — OCR pipeline (ingest → split → images)

Deploy with: `databricks bundle deploy -t mg_dev`

## Legacy Notebooks

These notebooks are kept for reference but have been superseded:

- `Parsing Documents - OCR.py` — Tesseract OCR (replaced by VLM approaches)
- `Parsing Documents - VLM.py` — Early VLM exploration (incomplete)
- `Exploring LLM Parsing in DBX.py` — LLM parsing experiments

## Viewer App

`apps/viewer_app/` contains a Streamlit app for browsing documents, viewing
PDF pages side-by-side with extracted markdown, and running vector search queries.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: complete README rewrite with four-option tutorial flow"
```

---

## Task 7: Final Review and Cleanup

**Step 1: Review all changes**

```bash
git diff --stat HEAD~5
git log --oneline -5
```

Verify:
- All notebooks have standardized config blocks
- No `brian_gen_ai` or `brian_serving_test` hardcoded references remain
- New ai_query notebook exists and follows the same style
- README accurately describes all notebooks and the flow
- Legacy notebooks have deprecation notices
- Bug fixes are in place (max_tokens, system prompt, IDs)

**Step 2: Run a quick grep for leftover hardcoded values**

```bash
grep -r "brian_gen_ai" interactive_examples/ --include="*.py"
grep -r "brian_serving" interactive_examples/ --include="*.py"
grep -r "nicholas_anile" interactive_examples/ --include="*.py"
```

All should return zero results (except possibly in commented-out legacy code within deprecated notebooks).

**Step 3: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "chore: final cleanup of hardcoded references"
```

---

## Summary of Deliverables

1. **Standardized config** across all 11+ notebooks (widgets, no hardcoded catalogs)
2. **New notebook**: `Parsing w ai_parse_document` — native Databricks SQL one-liner parsing (DBR 17.1+)
3. **New notebook**: `Parsing w ai_query` — Sonnet/Opus VLM-OCR via FMAPI
4. **Bug fixes**: max_tokens in Ray notebook, system prompt domain, unstable IDs
5. **Legacy markers**: 3 outdated notebooks marked with deprecation notices
6. **New README**: Complete rewrite with four-option flow, architecture diagram, data model, prerequisites
7. **Zero leftover personal references** (brian_gen_ai, nicholas_anile, etc.)
