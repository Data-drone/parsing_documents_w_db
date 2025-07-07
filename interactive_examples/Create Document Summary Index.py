# Databricks notebook source
# MAGIC %md
# MAGIC # Delta Table Flattening and Vector Index Creation
# MAGIC 
# MAGIC This notebook demonstrates:
# MAGIC 1. Flattening the nested `analysis_result` struct in your Delta table
# MAGIC 2. Creating a GTE Vector Index using Databricks Vector Search API

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Setup and Configuration

# COMMAND ----------

import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnotnull
from pyspark.sql.types import *
from databricks.vector_search.client import VectorSearchClient

# Initialize Vector Search client
vsc = VectorSearchClient()

# Configuration
CATALOG = "brian_gen_ai"
SCHEMA = "parsing_test"

SOURCE_TABLE_NAME = f"{CATALOG}.{SCHEMA}.document_analysis_simple"  # Replace with your table name
FLATTENED_TABLE_NAME = f"{CATALOG}.{SCHEMA}.flattened_documents"  # Target table name
VECTOR_SEARCH_ENDPOINT = "one-env-shared-endpoint-1"  # Replace with your endpoint name
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.document_analysis_simple_index"  # Name for your vector index

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Read and Examine Current Data

# COMMAND ----------

# Read the source Delta table
df_source = spark.table(SOURCE_TABLE_NAME)

# Display schema to confirm structure
print("Original Schema:")
df_source.printSchema()

# Show a few sample records
print("\nSample data:")
df_source.select("file_name", "analysis_result").show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Flatten the analysis_result Struct

# COMMAND ----------

# Flatten the analysis_result struct
df_flattened = df_source.select(
    col("file_path"),
    col("file_name"),
    col("character_count"),
    col("estimated_tokens"),
    col("file_size_bytes"),
    col("file_size_mb"),
    col("processing_category"),
    col("analysis_result.result").alias("analysis_result"),
    col("analysis_result.errorMessage").alias("analysis_error_message"),
    col("model_used"),
    col("processed_at")
)

# Filter out records where analysis_result is null or empty
df_flattened_clean = df_flattened.filter(
    col("analysis_result").isNotNull() & 
    (col("analysis_result") != "")
)

# Add a unique ID column for vector indexing
from pyspark.sql.functions import monotonically_increasing_id, concat, lit
df_flattened_clean = df_flattened_clean.withColumn(
    "id", 
    concat(lit("doc_"), monotonically_increasing_id().cast("string"))
)

print("Flattened Schema:")
df_flattened_clean.printSchema()

print(f"\nTotal records with valid analysis results: {df_flattened_clean.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Save Flattened Data to Delta Table

# COMMAND ----------

# Write the flattened data to a new Delta table
df_flattened_clean.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(FLATTENED_TABLE_NAME)

print(f"Flattened data saved to: {FLATTENED_TABLE_NAME}")

# Enable Change Data Feed for Vector Search
spark.sql(f"""
    ALTER TABLE {FLATTENED_TABLE_NAME} 
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

print("Change Data Feed enabled for vector sync")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Create Vector Search Endpoint (if not exists)

# COMMAND ----------

try:
    # Check if endpoint already exists
    endpoint_info = vsc.get_endpoint(VECTOR_SEARCH_ENDPOINT)
    print(f"Vector Search endpoint '{VECTOR_SEARCH_ENDPOINT}' already exists")
    print(f"Endpoint status: {endpoint_info.get('endpoint_status', {}).get('state', 'Unknown')}")
except Exception as e:
    # Create the endpoint if it doesn't exist
    print(f"Creating Vector Search endpoint: {VECTOR_SEARCH_ENDPOINT}")
    try:
        vsc.create_endpoint(
            name=VECTOR_SEARCH_ENDPOINT,
            endpoint_type="STANDARD"  # or "SERVERLESS" based on your needs
        )
        print(f"Vector Search endpoint '{VECTOR_SEARCH_ENDPOINT}' created successfully")
    except Exception as create_error:
        print(f"Error creating endpoint: {create_error}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Create Vector Index with GTE Embeddings

# COMMAND ----------

# Configuration for the vector index
INDEX_CONFIG = {
    "primary_key": "id",
    "text_column": "analysis_result",  # The flattened result column
    "embedding_model": {
        "model_name": "databricks-gte-large-en",  # GTE model
        "embedding_dimension": 1024  # Dimension for GTE-large
    },
    "sync_config": {
        "source_table": FLATTENED_TABLE_NAME,
        "sync_mode": "TRIGGERED"  # or "CONTINUOUS" for real-time sync
    }
}

try:
    # Create the vector index
    index = vsc.create_delta_sync_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=VECTOR_INDEX_NAME,
        source_table_name=FLATTENED_TABLE_NAME,
        pipeline_type="TRIGGERED",  # or "CONTINUOUS"
        primary_key="id",
        embedding_source_column="analysis_result",
        embedding_model_endpoint_name="databricks-gte-large-en"
    )
    
    print(f"Vector index '{VECTOR_INDEX_NAME}' created successfully!")
    print(f"Index details: {index}")
    
except Exception as e:
    print(f"Error creating vector index: {e}")
    # If index already exists, you might want to update or recreate it
    try:
        print("Attempting to get existing index...")
        existing_index = vsc.get_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=VECTOR_INDEX_NAME
        )
        print(f"Existing index found: {existing_index}")
    except:
        print("No existing index found")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Monitor Index Creation and Sync

# COMMAND ----------

import time

def wait_for_index_ready(endpoint_name, index_name, timeout_minutes=30):
    """Wait for the vector index to be ready"""
    timeout_seconds = timeout_minutes * 60
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        try:
            index = vsc.get_index(endpoint_name, index_name)
            status = index.describe().get('status').get('detailed_state')
            
            print(f"Index status: {status}")
            
            if status.startswith('ONLINE'):
                print("✅ Vector index is ready!")
                return True
                
            print("⏳ Waiting for index to be ready...")
            time.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            print(f"Error checking index status: {e}")
            time.sleep(30)
    
    print("❌ Timeout waiting for index to be ready")
    return False

## TODO Fix this code
# Wait for the index to be ready
if wait_for_index_ready(VECTOR_SEARCH_ENDPOINT, VECTOR_INDEX_NAME):
    # Trigger initial sync
    try:
        sync_result = vsc.sync_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=VECTOR_INDEX_NAME
        )
        print(f"Sync triggered: {sync_result}")
    except Exception as e:
        print(f"Error triggering sync: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Test Vector Search Functionality

# COMMAND ----------

def test_vector_search(query_text, num_results=5):
    """Test the vector search with a sample query"""
    try:
        results = index.similarity_search(

            query_text=query_text,
            columns=["id", "file_name", "analysis_result", "processing_category"],
            num_results=num_results
        )
        
        print(f"Search results for query: '{query_text}'")
        print("-" * 80)
        
        for i, result in enumerate(results["result"]["data_array"]):
            print(f"\n{i}. File: {result[1]}")
            print(f"   Category: {result[3]}")
            print(f"   Score: {result[4]}")
            print(f"   Content: {result[2][:200]}...")
            print(f"   ID: {result[0]}")
        
        return results
        
    except Exception as e:
        print(f"Error performing vector search: {e}")
        return None

# Test with a sample query (replace with relevant text from your domain)
sample_query = "error analysis report"  # Modify this based on your data
test_results = test_vector_search(sample_query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Utility Functions for Ongoing Operations

# COMMAND ----------

def get_index_stats():
    """Get statistics about the vector index"""
    try:
        index_info = vsc.get_index(VECTOR_SEARCH_ENDPOINT, VECTOR_INDEX_NAME)
        
        print("=== Vector Index Statistics ===")
        print(f"Index Name: {VECTOR_INDEX_NAME}")
        print(f"Endpoint: {VECTOR_SEARCH_ENDPOINT}")
        print(f"Status: {index_info.describe()['status']}")
        
        return index_info
        
    except Exception as e:
        print(f"Error getting index stats: {e}")
        return None

def manual_sync():
    """Manually trigger a sync of the vector index"""
    try:
        sync_result = vsc.sync_index(VECTOR_SEARCH_ENDPOINT, VECTOR_INDEX_NAME)
        print(f"Manual sync triggered: {sync_result}")
        return sync_result
    except Exception as e:
        print(f"Error triggering manual sync: {e}")
        return None

def update_source_data_and_sync():
    """Example of how to update source data and trigger sync"""
    # If you add new data to your source table, you can trigger a sync
    print("To update the vector index with new data:")
    print("1. Insert/update records in the source table")
    print("2. Call manual_sync() to update the vector index")
    
    # Example of checking for new data
    current_count = spark.table(FLATTENED_TABLE_NAME).count()
    print(f"Current record count in flattened table: {current_count}")

# Display current index statistics
get_index_stats()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC This notebook has:
# MAGIC 1. ✅ Flattened the `analysis_result` struct from your original Delta table
# MAGIC 2. ✅ Created a new Delta table with the flattened structure
# MAGIC 3. ✅ Set up a Databricks Vector Search endpoint
# MAGIC 4. ✅ Created a GTE vector index on the `analysis_result` text column
# MAGIC 5. ✅ Provided utilities for testing and managing the vector index
# MAGIC 
# MAGIC ### Next Steps:
# MAGIC - **Search**: Use `test_vector_search("your query")` to find similar documents
# MAGIC - **Sync**: Use `manual_sync()` when you add new data to keep the index updated  
# MAGIC - **Monitor**: Use `get_index_stats()` to check index health and statistics
# MAGIC 
# MAGIC ### Key Configuration Variables:
# MAGIC ```python
# MAGIC SOURCE_TABLE_NAME = "your_catalog.your_schema.your_table_name"
# MAGIC FLATTENED_TABLE_NAME = "your_catalog.your_schema.flattened_documents"  
# MAGIC VECTOR_SEARCH_ENDPOINT = "your-vector-search-endpoint"
# MAGIC VECTOR_INDEX_NAME = "document_analysis_index"
# MAGIC ```