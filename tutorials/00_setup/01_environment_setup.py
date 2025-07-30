# Databricks notebook source
# MAGIC %md
# MAGIC # Environment Setup
# MAGIC 
# MAGIC **Time**: 10 minutes  
# MAGIC **Prerequisites**: Access to Databricks workspace with Unity Catalog
# MAGIC 
# MAGIC This notebook helps you set up your Databricks environment for the document parsing tutorials.
# MAGIC 
# MAGIC ## What This Notebook Does
# MAGIC 1. Creates a Unity Catalog catalog for your work
# MAGIC 2. Creates schemas for organizing your data
# MAGIC 3. Creates volumes for storing documents
# MAGIC 4. Downloads sample documents for tutorials
# MAGIC 5. Verifies your environment is ready

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC 
# MAGIC Use the widgets at the top of the notebook (right sidebar) to set or override the values below. Defaults come from your environment variables or sensible fall-backs, so you can usually just run the notebook as-is.

# COMMAND ----------

# Get current user for catalog naming
current_user = spark.sql("SELECT current_user()").first()[0]
username = current_user.split('@')[0].replace('.', '_')

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Default configurations - you can modify these
CATALOG_NAME = os.getenv("CATALOG_NAME", f"{username}_document_parsing")
SCHEMA_NAME  = os.getenv("SCHEMA_NAME", "tutorials")
VOLUME_NAME  = os.getenv("VOLUME_NAME", "sample_docs")
LLM_MODEL    = os.getenv("LLM_MODEL", "databricks-meta-llama-3-3-70b-instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "databricks-gte-large-en")
VECTOR_SEARCH_ENDPOINT = os.getenv("VECTOR_SEARCH_ENDPOINT", "one-env-shared-endpoint-2")

# Create widgets with current defaults (user can override in the UI)
dbutils.widgets.text("CATALOG_NAME", CATALOG_NAME, "Catalog")
dbutils.widgets.text("SCHEMA_NAME",  SCHEMA_NAME,  "Schema")
dbutils.widgets.text("VOLUME_NAME",  VOLUME_NAME,  "Volume")

dbutils.widgets.text("LLM_MODEL", LLM_MODEL, "LLM Model")
dbutils.widgets.text("EMBEDDING_MODEL", EMBEDDING_MODEL, "Embedding Model")
dbutils.widgets.text("VECTOR_SEARCH_ENDPOINT", VECTOR_SEARCH_ENDPOINT, "Vector Search Endpoint")

# Read the (possibly overridden) widget values
CATALOG_NAME            = dbutils.widgets.get("CATALOG_NAME")
SCHEMA_NAME             = dbutils.widgets.get("SCHEMA_NAME")
VOLUME_NAME             = dbutils.widgets.get("VOLUME_NAME")
LLM_MODEL               = dbutils.widgets.get("LLM_MODEL")
EMBEDDING_MODEL         = dbutils.widgets.get("EMBEDDING_MODEL")
VECTOR_SEARCH_ENDPOINT  = dbutils.widgets.get("VECTOR_SEARCH_ENDPOINT")

print("🎯 Environment Configuration:")
print(f"   Catalog:           {CATALOG_NAME}")
print(f"   Schema:            {SCHEMA_NAME}")
print(f"   Volume:            {VOLUME_NAME}")
print(f"   LLM Model:         {LLM_MODEL}")
print(f"   Embedding Model:   {EMBEDDING_MODEL}")
print(f"   Vector Search EP:  {VECTOR_SEARCH_ENDPOINT or '⟨not set⟩'}")
print(f"   User:              {current_user}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create Unity Catalog Objects & Copy Sample PDFs

# COMMAND ----------

# Create catalog
print("Creating catalog...")
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG_NAME}")
print(f"✅ Catalog '{CATALOG_NAME}' ready")

# Create schema
print("\nCreating schema...")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}")
print(f"✅ Schema '{CATALOG_NAME}.{SCHEMA_NAME}' ready")

# Create volume
print("\nCreating volume...")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}.{VOLUME_NAME}")
print(f"✅ Volume '{CATALOG_NAME}.{SCHEMA_NAME}.{VOLUME_NAME}' ready")

# Get volume path
volume_path = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/{VOLUME_NAME}"
print(f"\n📁 Volume path: {volume_path}")

# COMMAND ----------

import shutil, os
from pathlib import Path

repo_docs_path = Path(f"/Workspace/Users/{current_user}/powering_knowledge_driven_applications/files/docs")
dst_path       = Path(volume_path)          # '/Volumes/…/sample_docs'

# Ensure destination exists
dst_path.mkdir(parents=True, exist_ok=True)

# Copy PDFs
pdf_copied = sum(
    1 for src in repo_docs_path.rglob("*.pdf")
    if not shutil.copy2(src, dst_path / src.name)
)

print(f"✅ Copied {pdf_copied} PDF(s) to volume" if pdf_copied else "⚠️  No PDFs found to copy")

# Copy page images directory and its contents
page_images_src = repo_docs_path / "page_images"
page_images_dst = dst_path / "page_images"

if page_images_src.exists():
    print(f"📷 Copying page images from {page_images_src}...")
    
    # Create page_images directory in destination
    page_images_dst.mkdir(exist_ok=True)
    
    # Copy all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    images_copied = 0
    
    for img_file in page_images_src.iterdir():
        if img_file.is_file() and img_file.suffix.lower() in image_extensions:
            shutil.copy2(img_file, page_images_dst / img_file.name)
            img_size = img_file.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"   ✅ {img_file.name} ({img_size:.2f} MB)")
            images_copied += 1
    
    print(f"✅ Copied {images_copied} image(s) to volume/page_images/")
else:
    print("⚠️  No page_images directory found in docs folder")

# Verify the copy operation
print(f"\n📋 Copy Summary:")
print(f"   PDFs copied: {pdf_copied}")
print(f"   Images copied: {images_copied if 'images_copied' in locals() else 0}")
print(f"   Destination: {volume_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Download Sample Documents

# COMMAND ----------

import requests

# Sample PDF URLs (you can replace with your own)
sample_pdfs = {
    "delta_lake_guide.pdf": "https://delta.io/pdfs/dldg_databricks.pdf",
    # Add more sample PDFs as needed
}

print("Downloading sample documents...")
for filename, url in sample_pdfs.items():
    try:
        response = requests.get(url)
        if response.status_code == 200:
            file_path = os.path.join(volume_path, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Downloaded: {filename}")
        else:
            print(f"❌ Failed to download: {filename}")
    except Exception as e:
        print(f"❌ Error downloading {filename}: {str(e)}")

# List files in volume
print(f"\n📄 Files in volume:")
files = dbutils.fs.ls(f"dbfs:{volume_path}")
for file in files:
    print(f"   - {file.name} ({file.size:,} bytes)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Verify Environment

# COMMAND ----------

# Verify we can query the catalog
print("🔍 Verifying catalog access...")
catalogs = spark.sql("SHOW CATALOGS").collect()
catalog_names = [row['catalog'] for row in catalogs]
if CATALOG_NAME in catalog_names:
    print(f"✅ Catalog '{CATALOG_NAME}' is accessible")
else:
    print(f"❌ Catalog '{CATALOG_NAME}' not found in available catalogs")

# Verify schema
print("\n🔍 Verifying schema access...")
spark.sql(f"USE CATALOG {CATALOG_NAME}")
schemas = spark.sql("SHOW SCHEMAS").collect()
schema_names = [row['databaseName'] for row in schemas]
if SCHEMA_NAME in schema_names:
    print(f"✅ Schema '{SCHEMA_NAME}' is accessible")
else:
    print(f"❌ Schema '{SCHEMA_NAME}' not found")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Verify GenAI Services
# MAGIC 
# MAGIC This step checks access to the model-serving endpoints (LLM and embedding) and the Vector Search endpoint defined in your environment variables.
# MAGIC 
# MAGIC Environment variables used:
# MAGIC - `LLM_MODEL`
# MAGIC - `EMBEDDING_MODEL`
# MAGIC - `VECTOR_SEARCH_ENDPOINT`

# COMMAND ----------

# Check model-serving and vector-search access

print("🔍 Checking model-serving endpoint access…\n")

# Use the values obtained from widgets/env vars earlier
for model_name in [LLM_MODEL, EMBEDDING_MODEL]:
    try:
        _ = spark.sql(f"SELECT ai_query('{model_name}', 'ping')").first()
        print(f"✅ Able to query model endpoint: {model_name}")
    except Exception as e:
        print(f"❌ Cannot query model endpoint: {model_name}")
        print(f"   Error: {e}")

print("\n🔍 Checking vector search endpoint access…\n")

try:
    from databricks.vector_search.client import VectorSearchClient
    vsc = VectorSearchClient()
    endpoints = vsc.list_endpoints()
    if VECTOR_SEARCH_ENDPOINT:
        match = any(ep.get('name') == VECTOR_SEARCH_ENDPOINT for ep in endpoints)
        if match:
            print(f"✅ Vector search endpoint '{VECTOR_SEARCH_ENDPOINT}' is available")
        else:
            avail = ', '.join(ep.get('name') for ep in endpoints) or 'none'
            print(f"❌ Endpoint '{VECTOR_SEARCH_ENDPOINT}' not found. Available endpoints: {avail}")
    else:
        print("⚠️ VECTOR_SEARCH_ENDPOINT not set. Use one of the following endpoints or create a new one:")
        for ep in endpoints:
            print(f"   - {ep.get('name')}")
except ImportError:
    print("⚠️ databricks-vectorsearch package not installed. Run `%pip install databricks-vectorsearch` in this notebook and retry.")
except Exception as e:
    print(f"❌ Vector search access issue: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Setup Complete!
# MAGIC 
# MAGIC Your environment is now ready for the document-parsing tutorials.
# MAGIC 
# MAGIC ### What we created & verified:
# MAGIC - **Catalog**, **Schema**, and **Volume** for storing tutorial data
# MAGIC - Verified access to configured model-serving endpoints and Vector Search endpoint
# MAGIC 
# MAGIC ### Next Steps:
# MAGIC 1. Explore the other setup notebooks or proceed to Module 1: Foundations
# MAGIC 2. If any check above failed, resolve it (permissions, packages, endpoint names) and re-run this notebook
# MAGIC 
# MAGIC ### Tips:
# MAGIC - Add your own PDFs to the volume path configured earlier
# MAGIC - Re-run this notebook anytime to retest your environment 