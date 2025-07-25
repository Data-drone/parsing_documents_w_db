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
# MAGIC Update these values to match your preferences:

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

print(f"🎯 Environment Configuration:")
print(f"   Catalog: {CATALOG_NAME}")
print(f"   Schema: {SCHEMA_NAME}")
print(f"   Volume: {VOLUME_NAME}")
print(f"   User: {current_user}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create Unity Catalog Objects

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

# NEW: Copy sample PDFs from repository docs folder into the volume using os & shutil
import shutil, os
from pathlib import Path

repo_docs_path = Path(f"/Workspace/Users/{current_user}/powering_knowledge_driven_applications/files/docs")
dst_path       = Path(volume_path)          # '/Volumes/…/sample_docs'

# Ensure destination exists
dst_path.mkdir(parents=True, exist_ok=True)

# Copy PDFs
copied = sum(
    1 for src in repo_docs_path.rglob("*.pdf")
    if not shutil.copy2(src, dst_path / src.name)
)

print(f"✅ Copied {copied} PDF(s) to volume" if copied else "⚠️  No PDFs found to copy")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Download Sample Documents

# COMMAND ----------

import requests
import os

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

# Check compute access
print("\n🔍 Checking compute access...")
print(f"✅ Running on: {spark.conf.get('spark.databricks.clusterUsageTags.clusterName', 'Unknown')}")
print(f"✅ Spark version: {spark.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Save Configuration for Later Use

# COMMAND ----------

# Save configuration to a temp view for other notebooks
config_df = spark.createDataFrame([
    {
        "catalog_name": CATALOG_NAME,
        "schema_name": SCHEMA_NAME,
        "volume_name": VOLUME_NAME,
        "volume_path": volume_path,
        "username": username
    }
])

# Create a table to store config
table_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.tutorial_config"
config_df.write.mode("overwrite").saveAsTable(table_name)

print(f"💾 Configuration saved to: {table_name}")
print("\nYou can load this configuration in other notebooks with:")
print(f"config = spark.table('{table_name}').first()")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Setup Complete!
# MAGIC 
# MAGIC Your environment is now ready for the document parsing tutorials.
# MAGIC 
# MAGIC ### What we created:
# MAGIC - **Catalog**: `{CATALOG_NAME}` - Your workspace for all tutorial data
# MAGIC - **Schema**: `{SCHEMA_NAME}` - Organization for tables
# MAGIC - **Volume**: `{VOLUME_NAME}` - Storage for PDF documents
# MAGIC - **Config Table**: `tutorial_config` - Saved configuration
# MAGIC 
# MAGIC ### Next Steps:
# MAGIC 1. Continue to `02_verify_permissions.py` to check all permissions
# MAGIC 2. Then explore the other setup notebooks
# MAGIC 3. Move on to Module 1: Foundations when ready
# MAGIC 
# MAGIC ### Tips:
# MAGIC - You can add your own PDFs to the volume at: `{volume_path}`
# MAGIC - All tutorial notebooks will use the configuration saved here
# MAGIC - If you need to reset, just re-run this notebook 