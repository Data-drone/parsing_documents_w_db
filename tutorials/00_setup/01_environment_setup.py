# Databricks notebook source
# MAGIC %md
# MAGIC # Environment Setup
# MAGIC
# MAGIC **Time**: 10 minutes
# MAGIC **Prerequisites**: Databricks workspace with Unity Catalog
# MAGIC
# MAGIC This notebook prepares your environment for the document parsing tutorials.
# MAGIC
# MAGIC ## What it does
# MAGIC 1. Creates a Unity Catalog catalog, schema, and volume
# MAGIC 2. Copies sample PDF documents into the volume
# MAGIC 3. Runs preflight checks (FMAPI endpoint access, DBR version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Use the widgets at the top of the notebook to override defaults.
# MAGIC Catalog name is auto-derived from your username.

# COMMAND ----------

import re

current_user = spark.sql("SELECT current_user()").first()[0]
# Sanitize: keep only alphanumeric + underscore, strip leading digits
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

print(f"Catalog:    {CATALOG}")
print(f"Schema:     {SCHEMA}")
print(f"Volume:     {VOLUME}")
print(f"Demo mode:  {DEMO_MODE}")
print(f"User:       {current_user}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create Unity Catalog Objects

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS `{CATALOG}`")
print(f"Catalog '{CATALOG}' ready")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")
print(f"Schema '{CATALOG}.{SCHEMA}' ready")

spark.sql(f"CREATE VOLUME IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`.`{VOLUME}`")
print(f"Volume '{CATALOG}.{SCHEMA}.{VOLUME}' ready")

print(f"\nVolume path: {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Copy Sample PDFs to Volume

# COMMAND ----------

import shutil
from pathlib import Path

# Look for PDFs in the repo docs folder (adjust path if needed)
repo_docs = Path(f"/Workspace/Users/{current_user}/parsing_documents_w_db/docs")
dst = Path(VOLUME_PATH)
dst.mkdir(parents=True, exist_ok=True)

pdf_count = 0
if repo_docs.exists():
    for src in repo_docs.rglob("*.pdf"):
        shutil.copy2(src, dst / src.name)
        pdf_count += 1

    # Also copy page_images if they exist
    page_images_src = repo_docs / "page_images"
    page_images_dst = dst / "page_images"
    img_count = 0
    if page_images_src.exists():
        page_images_dst.mkdir(exist_ok=True)
        for img in page_images_src.iterdir():
            if img.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                shutil.copy2(img, page_images_dst / img.name)
                img_count += 1

    print(f"Copied {pdf_count} PDF(s) and {img_count} image(s) to volume")
else:
    print(f"Repo docs not found at {repo_docs}")
    print("You can manually upload PDFs to the volume path above.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Download a sample PDF (if no local docs found)

# COMMAND ----------

import os, requests

if pdf_count == 0:
    sample_pdfs = {
        "delta_lake_guide.pdf": "https://delta.io/pdfs/dldg_databricks.pdf",
    }
    for filename, url in sample_pdfs.items():
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                with open(os.path.join(VOLUME_PATH, filename), "wb") as f:
                    f.write(resp.content)
                print(f"Downloaded: {filename}")
            else:
                print(f"Failed to download: {filename} (HTTP {resp.status_code})")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

# List files in volume
print(f"\nFiles in volume:")
for f in dbutils.fs.ls(f"dbfs:{VOLUME_PATH}"):
    print(f"  {f.name} ({f.size:,} bytes)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Preflight Checks

# COMMAND ----------

# Verify catalog access
print("Checking catalog access...")
catalogs = [r["catalog"] for r in spark.sql("SHOW CATALOGS").collect()]
if CATALOG in catalogs:
    print(f"  Catalog '{CATALOG}' accessible")
else:
    print(f"  Catalog '{CATALOG}' NOT found")

# Verify schema
spark.sql(f"USE CATALOG `{CATALOG}`")
schemas = [r["databaseName"] for r in spark.sql("SHOW SCHEMAS").collect()]
if SCHEMA in schemas:
    print(f"  Schema '{SCHEMA}' accessible")
else:
    print(f"  Schema '{SCHEMA}' NOT found")

# COMMAND ----------

# Check FMAPI endpoint access (needed for ai_query and ai_parse_document)
print("Checking FMAPI endpoint access...\n")

test_endpoints = [
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-claude-sonnet-4",
]

for endpoint in test_endpoints:
    try:
        spark.sql(f"SELECT ai_query('{endpoint}', 'ping')").first()
        print(f"  {endpoint}: accessible")
    except Exception as e:
        err = str(e)[:120]
        print(f"  {endpoint}: FAILED — {err}")

# Check ai_parse_document availability
print("\nChecking ai_parse_document availability...")
try:
    # Just check the function exists — don't actually parse anything
    spark.sql("SELECT ai_parse_document('test', 'test') AS test").first()
    print("  ai_parse_document: available")
except Exception as e:
    err = str(e)[:120]
    if "UNRESOLVED_ROUTINE" in err:
        print("  ai_parse_document: NOT available (need DBR 17.1+)")
    else:
        # Function exists but failed on dummy input — that's fine
        print("  ai_parse_document: available")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Complete
# MAGIC
# MAGIC Your environment is ready for the parsing tutorials.
# MAGIC
# MAGIC ### What was created
# MAGIC - Catalog, schema, and volume for storing tutorial data
# MAGIC - Sample PDF documents in the volume
# MAGIC
# MAGIC ### Next steps
# MAGIC 1. Run `00_setup/02_prepare_documents.py` to load PDFs into Delta tables and create page images
# MAGIC 2. Then run any parser in `01_parse/` — they can be run independently:
# MAGIC    - `01_pymupdf.py` — open source, CPU-only, zero cost
# MAGIC    - `02_ai_parse_document.py` — native Databricks SQL
# MAGIC    - `03_ai_query_vlm.py` — managed VLMs via FMAPI
# MAGIC    - `04_docling.py` — open source with OCR + table detection
# MAGIC 3. Run `02_compare/01_quality_comparison.py` to compare parser results
