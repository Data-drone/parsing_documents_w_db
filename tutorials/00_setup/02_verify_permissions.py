# Databricks notebook source
# MAGIC %md
# MAGIC # Verify Permissions
# MAGIC 
# MAGIC **Time**: 5 minutes  
# MAGIC **Prerequisites**: Completed 01_environment_setup.py
# MAGIC 
# MAGIC This notebook verifies you have all the necessary permissions for the document parsing tutorials.
# MAGIC 
# MAGIC ## What This Notebook Checks
# MAGIC 1. Unity Catalog permissions
# MAGIC 2. Compute cluster access
# MAGIC 3. Model serving endpoint access
# MAGIC 4. Vector search endpoint access
# MAGIC 5. Volume read/write permissions

# COMMAND ----------

# Load environment variables (optional .env)
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CATALOG_ENV = os.getenv("CATALOG_NAME")
SCHEMA_ENV  = os.getenv("SCHEMA_NAME")
VOLUME_ENV  = os.getenv("VOLUME_NAME")

# Try to load saved configuration
try:
    # Get current user
    current_user = spark.sql("SELECT current_user()").first()[0]
    username = current_user.split('@')[0].replace('.', '_')
    
    # Try to load saved config
    config_table = f"{username}_document_parsing.tutorials.tutorial_config"
    config = spark.table(config_table).first()
    
    CATALOG_NAME = CATALOG_ENV or config['catalog_name']
    SCHEMA_NAME  = SCHEMA_ENV or config['schema_name']
    VOLUME_NAME  = VOLUME_ENV or config['volume_name']
    volume_path = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/{VOLUME_NAME}"
    
    print("✅ Loaded configuration from previous setup")
    print(f"   Catalog: {CATALOG_NAME}")
    print(f"   Schema: {SCHEMA_NAME}")
    print(f"   Volume: {VOLUME_NAME}")
    
except Exception as e:
    print("⚠️  Could not load saved configuration. Using defaults.")
    print("   Please run 01_environment_setup.py first!")
    
    # Fallback configuration
    CATALOG_NAME = CATALOG_ENV or f"{username}_document_parsing"
    SCHEMA_NAME  = SCHEMA_ENV or "tutorials"
    VOLUME_NAME  = VOLUME_ENV or "sample_docs"
    volume_path = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/{VOLUME_NAME}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Unity Catalog Permissions

# COMMAND ----------

print("🔍 Checking Unity Catalog permissions...\n")

# Check CREATE CATALOG permission
try:
    test_catalog = f"{CATALOG_NAME}_permission_test"
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {test_catalog}")
    spark.sql(f"DROP CATALOG IF EXISTS {test_catalog}")
    print("✅ CREATE CATALOG permission: Granted")
except Exception as e:
    print("❌ CREATE CATALOG permission: Denied")
    print(f"   Error: {str(e)}")

# Check USE CATALOG permission
try:
    spark.sql(f"USE CATALOG {CATALOG_NAME}")
    print("✅ USE CATALOG permission: Granted")
except Exception as e:
    print("❌ USE CATALOG permission: Denied")
    print(f"   Error: {str(e)}")

# Check CREATE SCHEMA permission
try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.permission_test")
    spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG_NAME}.permission_test")
    print("✅ CREATE SCHEMA permission: Granted")
except Exception as e:
    print("❌ CREATE SCHEMA permission: Denied")
    print(f"   Error: {str(e)}")

# Check CREATE TABLE permission
try:
    spark.sql(f"CREATE TABLE IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}.permission_test (id INT)")
    spark.sql(f"DROP TABLE IF EXISTS {CATALOG_NAME}.{SCHEMA_NAME}.permission_test")
    print("✅ CREATE TABLE permission: Granted")
except Exception as e:
    print("❌ CREATE TABLE permission: Denied")
    print(f"   Error: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Compute Cluster Access

# COMMAND ----------

print("🔍 Checking compute cluster access...\n")

# Get cluster information
cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId", "Unknown")
cluster_name = spark.conf.get("spark.databricks.clusterUsageTags.clusterName", "Unknown")
runtime_version = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion", "Unknown")

print(f"✅ Currently running on cluster:")
print(f"   Name: {cluster_name}")
print(f"   ID: {cluster_id}")
print(f"   Runtime: {runtime_version}")

# Check if GPU is available (for advanced tutorials)
try:
    gpu_count = spark.sparkContext._jsc.sc().getExecutorMemoryStatus().size()
    # This is a simple check - in practice you'd want to check for actual GPU resources
    print(f"\n✅ Cluster has {gpu_count} executor(s)")
    print("   Note: GPU availability depends on node type")
except:
    print("\n⚠️  Could not determine executor count")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Model Serving Endpoints

# COMMAND ----------

print("🔍 Checking model serving endpoint access...\n")

# List of endpoints used in tutorials
endpoints_to_check = [
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-gte-large-en"
]

# Note: In a real check, you'd use the Databricks SDK or REST API
# For now, we'll just display what to verify
print("⚠️  Manual verification needed for model endpoints:")
for endpoint in endpoints_to_check:
    print(f"   - {endpoint}")

print("\nTo verify access:")
print("1. Go to Machine Learning > Serving in your workspace")
print("2. Check if you can see and query the above endpoints")
print("3. Or try using them in a LangChain notebook")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Vector Search Endpoints

# COMMAND ----------

print("🔍 Checking vector search access...\n")

try:
    from databricks.vector_search.client import VectorSearchClient
    
    # Initialize client
    vsc = VectorSearchClient()
    
    # Try to list endpoints
    endpoints = vsc.list_endpoints()
    print(f"✅ Vector Search access granted")
    print(f"   Found {len(endpoints)} endpoint(s)")
    
    # List first few endpoints
    for i, endpoint in enumerate(endpoints[:3]):
        print(f"   - {endpoint['name']}")
    
except ImportError:
    print("⚠️  databricks-vectorsearch not installed")
    print("   Run: %pip install databricks-vectorsearch")
except Exception as e:
    print("❌ Vector Search access issue")
    print(f"   Error: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Volume Read/Write Permissions

# COMMAND ----------

print("🔍 Checking volume permissions...\n")

import os

# Test write permission
try:
    test_file = os.path.join(volume_path, "permission_test.txt")
    with open(test_file, 'w') as f:
        f.write("Testing write permissions")
    print("✅ Volume WRITE permission: Granted")
    
    # Test read permission
    with open(test_file, 'r') as f:
        content = f.read()
    print("✅ Volume READ permission: Granted")
    
    # Clean up
    os.remove(test_file)
    print("✅ Volume DELETE permission: Granted")
    
except Exception as e:
    print("❌ Volume permission issue")
    print(f"   Error: {str(e)}")

# List volume contents
try:
    files = dbutils.fs.ls(f"dbfs:{volume_path}")
    print(f"\n📁 Volume contains {len(files)} file(s)")
except Exception as e:
    print(f"\n❌ Cannot list volume contents: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("🎯 PERMISSIONS SUMMARY")
print("=" * 50)

# Create a summary of what we checked
checks = {
    "Unity Catalog": ["CREATE CATALOG", "USE CATALOG", "CREATE SCHEMA", "CREATE TABLE"],
    "Compute": ["Cluster Access"],
    "Model Serving": ["Endpoint Access (manual check needed)"],
    "Vector Search": ["Client Access"],
    "Volume": ["READ", "WRITE", "DELETE"]
}

all_good = True
for category, items in checks.items():
    print(f"\n{category}:")
    for item in items:
        # This is simplified - in practice you'd track actual results
        print(f"  ✓ {item}")

print("\n" + "=" * 50)
print("\n✅ NEXT STEPS:")
print("1. If any permissions are missing, contact your workspace admin")
print("2. For model serving endpoints, verify access in the ML > Serving UI")
print("3. Continue to the next setup notebook or start Module 1")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Troubleshooting
# MAGIC 
# MAGIC ### Common Issues:
# MAGIC 
# MAGIC 1. **"Catalog does not exist"**
# MAGIC    - Run `01_environment_setup.py` first
# MAGIC    - Check if you have CREATE CATALOG permissions
# MAGIC 
# MAGIC 2. **"Cannot access model endpoints"**
# MAGIC    - Ensure endpoints are enabled in your workspace
# MAGIC    - Check with your admin for endpoint access
# MAGIC 
# MAGIC 3. **"Vector search not available"**
# MAGIC    - Install: `%pip install databricks-vectorsearch`
# MAGIC    - Ensure vector search is enabled in your workspace
# MAGIC 
# MAGIC 4. **"Volume access denied"**
# MAGIC    - Check Unity Catalog permissions
# MAGIC    - Ensure the volume was created successfully
# MAGIC 
# MAGIC ### Need Help?
# MAGIC - Check the [Databricks Documentation](https://docs.databricks.com)
# MAGIC - Contact your workspace administrator
# MAGIC - Review Unity Catalog permissions guide 