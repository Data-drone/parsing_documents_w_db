# Databricks notebook source
# MAGIC %md
# MAGIC # OCR Text Vector Index Creation
# MAGIC 
# MAGIC This notebook demonstrates:
# MAGIC 1. Reading OCR text data from your Delta table
# MAGIC 2. Creating a GTE Vector Index using Databricks Vector Search API on the OCR text

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Setup and Configuration

# COMMAND ----------

import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnotnull, concat, lit, monotonically_increasing_id
from pyspark.sql.types import *
from databricks.vector_search.client import VectorSearchClient

# Initialize Vector Search client
vsc = VectorSearchClient()

# Configuration
CATALOG = "brian_gen_ai"
SCHEMA = "parsing_test"

SOURCE_TABLE_NAME = f"{CATALOG}.{SCHEMA}.document_store_ocr"  # Replace with your table name
VECTOR_SEARCH_ENDPOINT = "one-env-shared-endpoint-1"  # Replace with your endpoint name
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.document_page_ocr_index"  # Name for your vector index

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Read and Examine Current Data

# COMMAND ----------

# Read the source Delta table
df_source = spark.table(SOURCE_TABLE_NAME)

# Display schema to confirm structure
print("Original Schema:")
df_source.printSchema()

# Show basic statistics
print(f"\nTotal records: {df_source.count()}")
print(f"Records with OCR text: {df_source.filter(col('ocr_text').isNotNull() & (col('ocr_text') != '')).count()}")

# Show a few sample records
print("\nSample data:")
df_source.select("doc_id", "source_filename", "page_number", "ocr_text").show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Prepare Data for Vector Indexing

# COMMAND ----------

# Filter out records where ocr_text is null or empty
df_clean = df_source.filter(
    col("ocr_text").isNotNull() & 
    (col("ocr_text") != "")
)

# Add a unique ID column for vector indexing if doc_id is not unique enough
# (combining doc_id with page_number to ensure uniqueness)
df_clean = df_clean.withColumn(
    "unique_id", 
    concat(col("doc_id"), lit("_page_"), col("page_number").cast("string"))
)

# Select relevant columns for the vector index
df_vector_ready = df_clean.select(
    col("unique_id").alias("id"),
    col("doc_id"),
    col("source_filename"), 
    col("page_number"),
    col("total_pages"),
    col("file_size_bytes"),
    col("processing_timestamp"),
    col("metadata_json"),
    col("ocr_text"),
    col("ocr_timestamp"),
    col("ocr_model")
)

print("Vector-ready Schema:")
df_vector_ready.printSchema()

print(f"\nTotal records ready for vector indexing: {df_vector_ready.count()}")

# Show sample of clean data
print("\nSample vector-ready data:")
df_vector_ready.select("id", "source_filename", "page_number", "ocr_text").show(3, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Save Prepared Data to Delta Table (Optional)

# COMMAND ----------

# Optionally save the prepared data to a new table for vector indexing
PREPARED_TABLE_NAME = f"{CATALOG}.{SCHEMA}.document_ocr_vector_ready"

# Write the prepared data to a new Delta table
df_vector_ready.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(PREPARED_TABLE_NAME)

print(f"Vector-ready data saved to: {PREPARED_TABLE_NAME}")

# Enable Change Data Feed for Vector Search
spark.sql(f"""
    ALTER TABLE {PREPARED_TABLE_NAME} 
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

try:
    # Create the vector index on OCR text
    index = vsc.create_delta_sync_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=VECTOR_INDEX_NAME,
        source_table_name=PREPARED_TABLE_NAME,  # Use the prepared table
        pipeline_type="TRIGGERED",  # or "CONTINUOUS" for real-time sync
        primary_key="id",
        embedding_source_column="ocr_text",  # The OCR text column
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
        # Get the index object for searching
        index = vsc.get_index(VECTOR_SEARCH_ENDPOINT, VECTOR_INDEX_NAME)
        
        results = index.similarity_search(
            query_text=query_text,
            columns=["id", "doc_id", "source_filename", "page_number", "ocr_text"],
            num_results=num_results
        )
        
        print(f"Search results for query: '{query_text}'")
        print("-" * 80)
        
        for i, result in enumerate(results["result"]["data_array"]):
            print(f"\n{i+1}. Document: {result[2]} (Page {result[3]})")
            print(f"   Doc ID: {result[1]}")
            print(f"   Score: {result[-1]}")  # Score is typically the last element
            print(f"   OCR Text Preview: {result[4][:200]}...")
            print(f"   Unique ID: {result[0]}")
        
        return results
        
    except Exception as e:
        print(f"Error performing vector search: {e}")
        return None

# Test with a sample query (replace with text relevant to your OCR documents)
sample_query = "invoice total amount"  # Modify this based on your OCR content
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

def search_by_document(doc_id, query_text, num_results=3):
    """Search for similar text within a specific document"""
    try:
        index = vsc.get_index(VECTOR_SEARCH_ENDPOINT, VECTOR_INDEX_NAME)
        
        results = index.similarity_search(
            query_text=query_text,
            columns=["id", "doc_id", "source_filename", "page_number", "ocr_text"],
            num_results=num_results,
            filters={"doc_id": doc_id}  # Filter by specific document
        )
        
        print(f"Search results in document '{doc_id}' for query: '{query_text}'")
        print("-" * 60)
        
        for i, result in enumerate(results["result"]["data_array"]):
            print(f"\n{i+1}. Page {result[3]}")
            print(f"   Score: {result[-1]}")
            print(f"   OCR Text: {result[4][:150]}...")
        
        return results
        
    except Exception as e:
        print(f"Error performing document-specific search: {e}")
        return None

def get_document_pages(doc_id):
    """Get all pages for a specific document"""
    try:
        df_doc = spark.table(PREPARED_TABLE_NAME).filter(col("doc_id") == doc_id)
        pages = df_doc.select("page_number", "ocr_text").orderBy("page_number").collect()
        
        print(f"Document '{doc_id}' has {len(pages)} pages:")
        for page in pages:
            print(f"  Page {page.page_number}: {page.ocr_text[:100]}...")
        
        return pages
        
    except Exception as e:
        print(f"Error getting document pages: {e}")
        return None

# Display current index statistics
get_index_stats()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC This notebook has:
# MAGIC 1. ✅ Read OCR text data from your Delta table
# MAGIC 2. ✅ Prepared the data for vector indexing (filtering null/empty OCR text)
# MAGIC 3. ✅ Created a unique ID combining doc_id and page_number
# MAGIC 4. ✅ Set up a Databricks Vector Search endpoint
# MAGIC 5. ✅ Created a GTE vector index on the `ocr_text` column
# MAGIC 6. ✅ Provided utilities for testing and managing the vector index
# MAGIC 
# MAGIC ### Next Steps:
# MAGIC - **Search Across All Documents**: Use `test_vector_search("your query")` to find similar OCR text across all documents
# MAGIC - **Search Within Document**: Use `search_by_document("doc_id", "query")` to search within a specific document
# MAGIC - **Sync**: Use `manual_sync()` when you add new OCR data to keep the index updated  
# MAGIC - **Monitor**: Use `get_index_stats()` to check index health and statistics
# MAGIC - **Explore**: Use `get_document_pages("doc_id")` to see all pages of a document
# MAGIC 
# MAGIC ### Key Configuration Variables:
# MAGIC ```python
# MAGIC SOURCE_TABLE_NAME = "brian_gen_ai.parsing_test.document_store_ocr"
# MAGIC PREPARED_TABLE_NAME = "brian_gen_ai.parsing_test.document_ocr_vector_ready"  
# MAGIC VECTOR_SEARCH_ENDPOINT = "one-env-shared-endpoint-1"
# MAGIC VECTOR_INDEX_NAME = "brian_gen_ai.parsing_test.document_page_ocr_index"
# MAGIC ```
# MAGIC 
# MAGIC ### Schema Used:
# MAGIC - **Primary Key**: `id` (combination of doc_id and page_number)
# MAGIC - **Embedding Source**: `ocr_text` column
# MAGIC - **Searchable Fields**: doc_id, source_filename, page_number, ocr_text