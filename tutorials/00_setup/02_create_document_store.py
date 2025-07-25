# Databricks notebook source
# MAGIC %md
# MAGIC # Document Store Creation with Binary Content
# MAGIC 
# MAGIC ## Overview
# MAGIC This notebook creates a comprehensive document inventory table that includes:
# MAGIC - **File metadata** (name, size, modification time, directory structure)
# MAGIC - **Binary content** of each document stored directly in Delta Lake
# MAGIC 
# MAGIC ## Benefits of This Approach
# MAGIC - 🗃️ **Centralized Storage**: Documents stored directly in Delta Lake tables
# MAGIC - 🔍 **Easy Querying**: Use SQL to search and filter documents by metadata
# MAGIC - 🚀 **Downstream Processing**: Ready for text extraction, parsing, or ML workflows
# MAGIC - 📊 **Version Control**: Delta Lake provides ACID transactions and time travel
# MAGIC - 🔗 **Integration**: Works seamlessly with Databricks ML and AI workflows
# MAGIC 
# MAGIC ## Supported File Types
# MAGIC - **PDF** documents (.pdf)
# MAGIC - **Microsoft Word** documents (.doc, .docx)
# MAGIC 
# MAGIC ## What You'll Get
# MAGIC A Delta Lake table containing both metadata and binary content for all your documents, ready for advanced document processing and analysis.

# COMMAND ----------
# SECTION 1: SETUP AND CONFIGURATION
# ==================================
# Import required libraries and set up our workspace configuration

import os
from datetime import datetime
from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType, BinaryType

# --------------------------------------------
# Widget-driven configuration (similar to 01_environment_setup)
# --------------------------------------------

# Load environment variables from optional .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Determine current user to build sensible defaults
current_user = spark.sql("SELECT current_user()").first()[0]
username = current_user.split('@')[0].replace('.', '_')

# Default values from env vars (or fallbacks)
_CATALOG_DEF = os.getenv("CATALOG_NAME", f"{username}_document_parsing")
_SCHEMA_DEF  = os.getenv("SCHEMA_NAME", "tutorials")
_VOLUME_DEF  = os.getenv("VOLUME_NAME", "sample_docs")

# Widgets allow users to override
dbutils.widgets.text("CATALOG_NAME", _CATALOG_DEF, "Catalog")
dbutils.widgets.text("SCHEMA_NAME",  _SCHEMA_DEF,  "Schema")
dbutils.widgets.text("VOLUME_NAME",  _VOLUME_DEF,  "Volume")

# Fetch the (possibly overridden) values
CATALOG = dbutils.widgets.get("CATALOG_NAME")
SCHEMA  = dbutils.widgets.get("SCHEMA_NAME")
VOLUME  = dbutils.widgets.get("VOLUME_NAME")

# Paths / table names derived from final values
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.document_store"

print("🎯 Configuration:")
print(f"   Catalog: {CATALOG}")
print(f"   Schema:  {SCHEMA}")
print(f"   Volume:  {VOLUME}")
print(f"   Volume path: {VOLUME_PATH}")
print(f"   Output table: {OUTPUT_TABLE}")

print("Note: This script will scan all subdirectories for PDF files")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Document Discovery and Content Extraction
# MAGIC 
# MAGIC This section recursively scans the volume for documents and reads their content.
# MAGIC 
# MAGIC ### Key Features:
# MAGIC - **Recursive scanning** of nested directory structures
# MAGIC - **Binary content extraction** for document processing
# MAGIC - **Error handling** to preserve metadata even if content reading fails
# MAGIC - **Path normalization** for library compatibility

# COMMAND ----------
# SECTION 2: DOCUMENT DISCOVERY AND CONTENT EXTRACTION
# ====================================================
# This section recursively scans the volume for documents and reads their content

def list_files_with_metadata_and_content(volume_path, exts=(".pdf")):
    """
    Recursively scan a volume for documents and extract both metadata and binary content.
    
    WHY THIS FUNCTION EXISTS:
    - Databricks volumes can contain nested directory structures
    - We need both file information (metadata) AND the actual file content
    - Binary storage allows us to preserve the original document format
    
    PARAMETERS:
    - volume_path: The root path to scan (e.g., /Volumes/catalog/schema/volume)
    - exts: Tuple of file extensions to include (case-insensitive)
    
    RETURNS:
    - List of dictionaries, each containing file metadata and binary content
    """
    all_files = []
    processed_count = 0
    
    def _recurse(path):
        """
        Internal recursive function to walk through directory tree.
        This is a nested function so it can access the all_files list.
        """
        nonlocal processed_count
        try:
            # dbutils.fs.ls() lists contents of a directory in Databricks
            for entry in dbutils.fs.ls(path):
                if entry.isDir():
                    # If it's a directory, recurse into it
                    print(f"Scanning directory: {entry.path}")
                    _recurse(entry.path)
                else:
                    # If it's a file, check if it matches our target extensions
                    if entry.name.lower().endswith(exts):
                        try:
                            # CRITICAL STEP: Read the binary content of the file
                            # Handle Unity Catalog volume paths correctly for library compatibility
                            # Many document processing libraries expect /Volumes/ format, not dbfs: format
                            
                            # Clean up the path format
                            if entry.path.startswith("dbfs:"):
                                volume_path_clean = entry.path.replace("dbfs:", "")
                            else:
                                volume_path_clean = entry.path
                            
                            # Ensure the path starts with /Volumes/ for Unity Catalog compatibility
                            # This format is required for proper integration with document processing libraries
                            if not volume_path_clean.startswith("/Volumes/"):
                                # If it doesn't start with /Volumes/, prepend it
                                if volume_path_clean.startswith("/"):
                                    volume_path_clean = "/Volumes" + volume_path_clean
                                else:
                                    volume_path_clean = "/Volumes/" + volume_path_clean
                            
                            binary_content = None
                            with open(volume_path_clean, "rb") as f:
                                binary_content = f.read()
                            
                            # Debug: Show path transformation (can be removed in production)
                            if entry.path != volume_path_clean:
                                print(f"    Path transformed: {entry.path} → {volume_path_clean}")
                            
                            # Create a comprehensive record for this document
                            file_record = {
                                "file_name": entry.name,
                                "volume_path": volume_path_clean,  # Use the clean volume path
                                "file_extension": os.path.splitext(entry.name)[1].lower(),
                                "file_size_bytes": entry.size,
                                "modification_time": datetime.fromtimestamp(entry.modificationTime / 1000.0),
                                "directory": os.path.dirname(volume_path_clean.rstrip("/")),
                                "binary_content": binary_content  # The actual file content as bytes
                            }
                            
                            all_files.append(file_record)
                            processed_count += 1
                            print(f"✓ Processed ({processed_count}): {entry.name} ({entry.size:,} bytes)")
                            
                        except Exception as e:
                            print(f"✗ Error reading content for {entry.name}: {e}")
                            # Still add the file record without binary content
                            # This ensures we don't lose metadata even if content reading fails
                            
                            # Use the same path cleaning logic for consistency
                            if entry.path.startswith("dbfs:"):
                                volume_path_clean = entry.path.replace("dbfs:", "")
                            else:
                                volume_path_clean = entry.path
                            
                            if not volume_path_clean.startswith("/Volumes/"):
                                if volume_path_clean.startswith("/"):
                                    volume_path_clean = "/Volumes" + volume_path_clean
                                else:
                                    volume_path_clean = "/Volumes/" + volume_path_clean
                            
                            file_record = {
                                "file_name": entry.name,
                                "volume_path": volume_path_clean,  # Use the clean volume path
                                "file_extension": os.path.splitext(entry.name)[1].lower(),
                                "file_size_bytes": entry.size,
                                "modification_time": datetime.fromtimestamp(entry.modificationTime / 1000.0),
                                "directory": os.path.dirname(volume_path_clean.rstrip("/")),
                                "binary_content": None  # Null for failed reads
                            }
                            all_files.append(file_record)
                            processed_count += 1
        except Exception as e:
            print(f"Error accessing {path}: {e}")
    
    # Start the recursive scan
    print(f"Starting recursive scan of: {volume_path}")
    print(f"Looking for files with extensions: {exts}")
    _recurse(volume_path)
    return all_files

# Execute the file discovery and content extraction
print("=" * 50)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Spark DataFrame Creation
# MAGIC 
# MAGIC Convert our Python list of file records into a Spark DataFrame with proper schema definition.
# MAGIC 
# MAGIC ### Schema Components:
# MAGIC - **file_name**: Original filename (e.g., "report.pdf")
# MAGIC - **volume_path**: Full Unity Catalog volume path
# MAGIC - **file_extension**: File type (.pdf, .doc, .docx)
# MAGIC - **file_size_bytes**: Size in bytes for storage monitoring
# MAGIC - **modification_time**: Last modified timestamp
# MAGIC - **directory**: Parent directory path
# MAGIC - **binary_content**: Actual file content as binary data

# COMMAND ----------
print("SCANNING FOR DOCUMENTS...")
print("=" * 50)
files = list_files_with_metadata_and_content(VOLUME_PATH)
print("=" * 50)
print(f"SCAN COMPLETE: Found {len(files)} document files.")
print("=" * 50)

# COMMAND ----------
# SECTION 3: SPARK DATAFRAME CREATION
# ===================================
# Convert our Python list of file records into a Spark DataFrame

def make_spark_df(files):
    """
    Create a Spark DataFrame from our file records with proper schema definition.
    
    Returns: Spark DataFrame with explicit schema for optimal performance and type safety
    """
    
    # Define the schema explicitly for better performance and type safety
    schema = StructType([
        StructField("file_name", StringType(), True),           # Filename
        StructField("volume_path", StringType(), True),         # Full path
        StructField("file_extension", StringType(), True),      # File type
        StructField("file_size_bytes", LongType(), True),       # File size
        StructField("modification_time", TimestampType(), True), # Last modified
        StructField("directory", StringType(), True),           # Parent directory
        StructField("binary_content", BinaryType(), True)       # File content as bytes
    ])
    
    print("Creating Spark DataFrame with schema:")
    for field in schema.fields:
        print(f"  - {field.name}: {field.dataType}")
    
    return spark.createDataFrame(files, schema=schema)

# Create the DataFrame
print("\n" + "=" * 50)
print("CREATING SPARK DATAFRAME...")
print("=" * 50)
df = make_spark_df(files)

# Display a preview WITHOUT binary content (binary data isn't human-readable)
print("Preview of document metadata (excluding binary content for readability):")
df_display = df.select(
    "file_name", 
    "volume_path", 
    "file_extension", 
    "file_size_bytes", 
    "modification_time", 
    "directory"
)
display(df_display)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Persisting to Delta Lake Table
# MAGIC 
# MAGIC Save our DataFrame as a managed Delta Lake table with ACID transactions and versioning.
# MAGIC 
# MAGIC ### Benefits:
# MAGIC - **ACID Transactions**: Ensure data consistency
# MAGIC - **Time Travel**: Access historical versions of your data
# MAGIC - **Performance**: Optimized for analytics workloads
# MAGIC - **Unity Catalog Integration**: Proper governance and security

# COMMAND ----------
# SECTION 4: PERSISTING TO DELTA LAKE TABLE
# ==========================================
# Save our DataFrame as a managed Delta Lake table

print("\n" + "=" * 50)
print("SAVING TO DELTA LAKE TABLE...")
print("=" * 50)

# Write to Delta Lake table
# mode("overwrite") replaces the entire table if it exists
# saveAsTable() creates a managed table in Unity Catalog
df.write.mode("overwrite").saveAsTable(OUTPUT_TABLE)

# Verify the save was successful
row_count = df.count()
print(f"✓ Table {OUTPUT_TABLE} successfully created!")
print(f"✓ Stored {row_count:,} document files with binary content")

# Calculate total storage used
total_bytes = df.agg({"file_size_bytes": "sum"}).collect()[0][0] or 0
total_mb = total_bytes / (1024 * 1024)
print(f"✓ Total document storage: {total_mb:.2f} MB ({total_bytes:,} bytes)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Verification and Usage Examples
# MAGIC 
# MAGIC Verify our data was stored correctly and demonstrate how to use the resulting table.
# MAGIC 
# MAGIC ### Verification Checks:
# MAGIC - **Binary content presence**: Ensure files were read successfully
# MAGIC - **Size validation**: Confirm binary size matches file size
# MAGIC - **Data integrity**: Validate all metadata fields
# MAGIC 
# MAGIC ### Next Steps:
# MAGIC - Extract text using document parsing libraries
# MAGIC - Perform similarity searches across documents
# MAGIC - Create embeddings for semantic search
# MAGIC - Build document classification models
# MAGIC - Set up automated document processing pipelines

# COMMAND ----------
# SECTION 5: VERIFICATION AND USAGE EXAMPLES
# ==========================================
# Verify our data was stored correctly and show how to use it

print("VERIFYING BINARY CONTENT STORAGE...")

# Query to verify binary content is properly stored
verification_query = f"""
SELECT 
    file_name,
    file_extension,
    file_size_bytes,
    CASE 
        WHEN binary_content IS NOT NULL THEN 'Yes' 
        ELSE 'No' 
    END as has_binary_content,
    LENGTH(binary_content) as binary_size_bytes,
    -- Check if binary size matches file size (they should be equal)
    CASE 
        WHEN LENGTH(binary_content) = file_size_bytes THEN 'Match ✓'
        WHEN binary_content IS NULL THEN 'No Content ✗'
        ELSE 'Mismatch ⚠'
    END as size_validation
FROM {OUTPUT_TABLE}
ORDER BY file_name
"""

result_df = spark.sql(verification_query)
display(result_df)

print("\nVERIFICATION COMPLETE!")
print(f"\nTo access your documents programmatically:")
print(f"  df = spark.table('{OUTPUT_TABLE}')")
print(f"  # Get binary content for a specific file:")
print(f"  binary_data = df.filter(df.file_name == 'your_file.pdf').select('binary_content').collect()[0][0]")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Document Collection Summary
# MAGIC 
# MAGIC Generate interesting statistics about your document collection.
# MAGIC 
# MAGIC ### Summary Metrics:
# MAGIC - **File type distribution**: Count and size by extension
# MAGIC - **Storage utilization**: Total and average file sizes
# MAGIC - **Timeline analysis**: Oldest and newest documents
# MAGIC - **Collection health**: Success rates and data quality

# COMMAND ----------
# BONUS: SUMMARY STATISTICS
# ========================
# Get some interesting stats about your document collection

summary_query = f"""
SELECT 
    file_extension,
    COUNT(*) as file_count,
    ROUND(SUM(file_size_bytes) / 1024 / 1024, 2) as total_mb,
    ROUND(AVG(file_size_bytes) / 1024, 2) as avg_size_kb,
    MIN(modification_time) as oldest_file,
    MAX(modification_time) as newest_file
FROM {OUTPUT_TABLE}
WHERE binary_content IS NOT NULL
GROUP BY file_extension
ORDER BY file_count DESC
"""

summary_df = spark.sql(summary_query)
display(summary_df)

print("Your document store is ready for use! 🎉")