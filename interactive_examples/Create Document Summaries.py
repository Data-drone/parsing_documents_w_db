# Databricks notebook source
# MAGIC %md
# MAGIC # Simple Document Analysis with AI Query
# MAGIC 
# MAGIC This notebook processes documents from a Delta table with two simple processing strategies:
# MAGIC - **Small documents** (<100k tokens AND <4MB): Full analysis
# MAGIC - **Large documents** (≥100k tokens OR ≥4MB): Brief analysis with truncation
# MAGIC 
# MAGIC Edge cases may fail - that's okay for this initial implementation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

### Configuration Variables
CATALOG = "brian_gen_ai"
SCHEMA = "parsing_test" 
SOURCE_TABLE = "document_markdown"
OUTPUT_TABLE = "document_analysis_simple"

# AI Model configuration
AI_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# Size thresholds
TOKEN_THRESHOLD = 100000  # 100k tokens
SIZE_THRESHOLD_MB = 4     # 4MB in bytes
SIZE_THRESHOLD_BYTES = SIZE_THRESHOLD_MB * 1024 * 1024

print(f"Processing: {CATALOG}.{SCHEMA}.{SOURCE_TABLE}")
print(f"Output: {CATALOG}.{SCHEMA}.{OUTPUT_TABLE}")
print(f"Thresholds: {TOKEN_THRESHOLD:,} tokens, {SIZE_THRESHOLD_MB}MB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick Data Overview

# COMMAND ----------

# Check document distribution
size_distribution = spark.sql(f"""
SELECT 
    CASE 
        WHEN (character_count / 4) < {TOKEN_THRESHOLD} 
         AND file_size_bytes < {SIZE_THRESHOLD_BYTES} THEN 'SMALL_DOCUMENTS'
        ELSE 'LARGE_DOCUMENTS'
    END AS size_category,
    COUNT(*) as count,
    AVG(character_count / 4) as avg_estimated_tokens,
    AVG(file_size_bytes) as avg_size_bytes
FROM {CATALOG}.{SCHEMA}.{SOURCE_TABLE}
WHERE extraction_success = true
  AND markdown_content IS NOT NULL
GROUP BY 1
""")

display(size_distribution)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process Documents with Simple Strategy

# COMMAND ----------

# Process all documents with two-tier strategy
spark.sql(f"""
CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.{OUTPUT_TABLE} AS
SELECT 
    volume_path as file_path,
    file_name,
    character_count,
    ROUND(character_count / 4, 0) as estimated_tokens,
    file_size_bytes,
    ROUND(file_size_bytes / 1024 / 1024, 2) as file_size_mb,
    
    -- Determine processing category
    CASE 
        WHEN (character_count / 4) < {TOKEN_THRESHOLD} 
         AND file_size_bytes < {SIZE_THRESHOLD_BYTES} THEN 'SMALL_DOC'
        ELSE 'LARGE_DOC'
    END AS processing_category,
    
    -- AI analysis with simple two-tier approach
    CASE 
        -- Full analysis for small documents
        WHEN (character_count / 4) < {TOKEN_THRESHOLD} 
         AND file_size_bytes < {SIZE_THRESHOLD_BYTES} THEN
            ai_query(
                '{AI_ENDPOINT}',
                CONCAT(
                    'Analyze this document and provide a comprehensive summary:\\n\\n',
                    markdown_content,
                    '\\n\\nProvide:\\n',
                    '1. Key Topics\\n',
                    '2. Main Insights\\n',
                    '3. Why someone would need this document\\n',
                    '4. Summary\\n\\n',
                    'Analysis:'
                ),
                failOnError => false
            )
        
        -- Brief analysis for large documents (truncated)
        ELSE
            ai_query(
                '{AI_ENDPOINT}',
                CONCAT(
                    'Analyze this large document (truncated excerpt):\\n\\n',
                    LEFT(markdown_content, 300000),  -- Truncate to ~75k tokens
                    '\\n\\n[Document truncated due to size]\\n\\n',
                    'Provide brief analysis:\\n',
                    '1. Main Topics\\n',
                    '2. Document Purpose\\n',
                    '3. Key Points\\n\\n',
                    'Analysis:'
                ),
                failOnError => false
            )
    END AS analysis_result,
    
    '{AI_ENDPOINT}' AS model_used,
    CURRENT_TIMESTAMP() AS processed_at
    
FROM {CATALOG}.{SCHEMA}.{SOURCE_TABLE}
WHERE extraction_success = true
  AND markdown_content IS NOT NULL
""")

print("✅ Processing completed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

# Basic results summary
results_summary = spark.sql(f"""
SELECT 
    processing_category,
    COUNT(*) as total_docs,
    COUNT(CASE WHEN analysis_result IS NOT NULL THEN 1 END) as successful,
    COUNT(CASE WHEN analysis_result IS NULL THEN 1 END) as failed,
    ROUND(COUNT(CASE WHEN analysis_result IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 2) as success_rate,
    AVG(estimated_tokens) as avg_tokens,
    AVG(file_size_mb) as avg_size_mb
FROM {CATALOG}.{SCHEMA}.{OUTPUT_TABLE}
GROUP BY processing_category
""")

display(results_summary)

# COMMAND ----------

# Show sample results
sample_results = spark.sql(f"""
SELECT 
    file_name,
    processing_category,
    estimated_tokens,
    file_size_mb,
    substr(cast(analysis_result as string), 1, 200) as analysis_preview
FROM {CATALOG}.{SCHEMA}.{OUTPUT_TABLE}
WHERE analysis_result IS NOT NULL
ORDER BY estimated_tokens
LIMIT 10
""")

display(sample_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Failed Documents (Optional Review)

# COMMAND ----------

# Check failed documents if any
failed_docs = spark.sql(f"""
SELECT 
    processing_category,
    COUNT(*) as failed_count,
    AVG(estimated_tokens) as avg_tokens,
    AVG(file_size_mb) as avg_size_mb
FROM {CATALOG}.{SCHEMA}.{OUTPUT_TABLE}
WHERE analysis_result IS NULL
GROUP BY processing_category
""")

display(failed_docs)

# COMMAND ----------

print("=== PROCESSING COMPLETE ===")
print(f"Results saved to: {CATALOG}.{SCHEMA}.{OUTPUT_TABLE}")
print("Review the summary tables above for processing results.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Simple Processing Complete**
# MAGIC 
# MAGIC - Small documents get full analysis
# MAGIC - Large documents get truncated analysis  
# MAGIC - Failed edge cases are acceptable for this version