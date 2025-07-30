# Databricks notebook source
# MAGIC %md
# MAGIC # Document Analysis and Summarization with AI
# MAGIC 
# MAGIC ## Overview
# MAGIC This notebook demonstrates how to use AI models to analyze and summarize documents that have been processed into markdown format. We'll implement a **two-tier processing strategy** that adapts to document size:
# MAGIC 
# MAGIC - **Small documents** (<100k tokens AND <4MB): Full comprehensive analysis
# MAGIC - **Large documents** (≥100k tokens OR ≥4MB): Brief analysis with intelligent truncation
# MAGIC 
# MAGIC ## What You'll Learn
# MAGIC - 📊 **Size-based Processing**: Adapt AI analysis strategy based on document characteristics
# MAGIC - 🤖 **AI Query Integration**: Use Databricks AI functions for document analysis
# MAGIC - 📈 **Quality Assessment**: Measure and track processing success rates
# MAGIC - 💾 **Results Management**: Store analysis results in Delta Lake for downstream use
# MAGIC 
# MAGIC ## Processing Strategy
# MAGIC This tutorial shows a pragmatic approach where edge cases may fail - that's acceptable for this initial implementation as we focus on the core workflow patterns.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration
# MAGIC 
# MAGIC Just like our previous tutorials, we'll start by loading environment variables and setting up configurable widgets. This ensures consistency across our document processing pipeline.

# COMMAND ----------

# Install required packages and load environment
%pip install python-dotenv
%restart_python

# COMMAND ----------

# Load environment variables from .env (if present) **before** we read them via os.getenv
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # returns True if a .env is found and parsed

import os
from datetime import datetime
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ### Environment Configuration with Widgets
# MAGIC 
# MAGIC Following the same pattern as our previous tutorials, we'll create widgets that allow runtime configuration while defaulting to environment variables. This makes the notebook flexible for different users and environments.

# COMMAND ----------

# -----------------------------------------------------------------------------
# Environment configuration via widgets
# -----------------------------------------------------------------------------
# Pull defaults from environment variables (populated via `.env` or workspace)

# Create Databricks widgets so users can override at run-time
dbutils.widgets.text("catalog_name", os.getenv("CATALOG_NAME", "brian_gen_ai"), "Catalog Name")
dbutils.widgets.text("schema_name",  os.getenv("SCHEMA_NAME", "parsing_test"),   "Schema Name")
dbutils.widgets.text("ai_endpoint",  os.getenv("LLM_MODEL", "databricks-meta-llama-3-3-70b-instruct"), "AI Model Endpoint")

# Read values back from the widgets
CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA  = dbutils.widgets.get("schema_name")
AI_ENDPOINT = dbutils.widgets.get("ai_endpoint")

# Construct table names using the configured catalog and schema
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.document_markdown"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.document_analysis_simple"

print(f"📊 Document Analysis Configuration:")
print(f"   Source table: {SOURCE_TABLE}")
print(f"   Output table: {OUTPUT_TABLE}")
print(f"   AI endpoint:  {AI_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Processing Parameters
# MAGIC 
# MAGIC We define size thresholds that determine our processing strategy. These thresholds help us balance thoroughness with processing efficiency:
# MAGIC 
# MAGIC - **Token Threshold**: 100k tokens (roughly 75k words) - beyond this, documents get truncated
# MAGIC - **Size Threshold**: 4MB file size - large files often indicate complex layouts or many images
# MAGIC 
# MAGIC The combination of BOTH criteria ensures we handle different types of large documents appropriately.

# COMMAND ----------

# Size thresholds for determining processing strategy
TOKEN_THRESHOLD = 100000  # 100k tokens (estimated as character_count / 4)
SIZE_THRESHOLD_MB = 4     # 4MB file size threshold
SIZE_THRESHOLD_BYTES = SIZE_THRESHOLD_MB * 1024 * 1024

# AI Analysis Prompts - Define once for easy experimentation
# 🔧 MODIFY THESE PROMPTS TO EXPERIMENT WITH DIFFERENT ANALYSIS APPROACHES
# The {content} placeholder will be replaced with the actual document content

SMALL_DOC_PROMPT = """Analyze this document and provide a comprehensive summary:

{content}

Provide:
1. Key Topics: What are the main subjects covered?
2. Main Insights: What are the key findings or important information?
3. Use Cases: Why would someone need this document? What problems does it solve?
4. Executive Summary: A concise overview in 2-3 sentences

Format your response clearly with headers for each section.

Analysis:"""

LARGE_DOC_PROMPT = """Analyze this large document (showing first portion due to size):

{content}

[DOCUMENT TRUNCATED - This is a large document showing first 300k characters]

Based on this content, provide brief analysis:
1. Main Topics: What are the primary subjects?
2. Document Purpose: What is this document intended for?
3. Key Points: What are the most important takeaways?

Note: Analysis based on document beginning due to size limitations.

Analysis:"""

# Escape prompts for SQL (replace newlines and escape quotes)
SMALL_DOC_PROMPT_SQL = SMALL_DOC_PROMPT.replace('\n', '\\n').replace("'", "\\'")
LARGE_DOC_PROMPT_SQL = LARGE_DOC_PROMPT.replace('\n', '\\n').replace("'", "\\'")

print(f"📏 Processing Thresholds:")
print(f"   Token limit:     {TOKEN_THRESHOLD:,} tokens")
print(f"   File size limit: {SIZE_THRESHOLD_MB}MB ({SIZE_THRESHOLD_BYTES:,} bytes)")
print(f"   Strategy: Documents exceeding EITHER threshold get brief analysis")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Exploration and Validation
# MAGIC 
# MAGIC Before processing, let's examine our source data to understand the distribution of document sizes. This helps us validate our thresholds and understand what proportion of documents will use each processing strategy.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Document Size Distribution
# MAGIC 
# MAGIC This query categorizes our documents based on our processing thresholds and provides statistics about each category. Understanding this distribution helps us:
# MAGIC 
# MAGIC - **Validate thresholds**: Ensure our criteria create reasonable groupings
# MAGIC - **Estimate processing time**: Larger documents take longer to analyze
# MAGIC - **Plan resources**: Know how many documents will use each strategy

# COMMAND ----------

# Check document distribution across our processing categories
size_distribution = spark.sql(f"""
SELECT 
    CASE 
        WHEN (character_count / 4) < {TOKEN_THRESHOLD} 
         AND file_size_bytes < {SIZE_THRESHOLD_BYTES} THEN 'SMALL_DOCUMENTS'
        ELSE 'LARGE_DOCUMENTS'
    END AS size_category,
    COUNT(*) as document_count,
    ROUND(AVG(character_count / 4), 0) as avg_estimated_tokens,
    ROUND(AVG(file_size_bytes / 1024 / 1024), 2) as avg_size_mb,
    ROUND(MIN(character_count / 4), 0) as min_tokens,
    ROUND(MAX(character_count / 4), 0) as max_tokens
FROM {SOURCE_TABLE}
WHERE extraction_success = true
  AND markdown_content IS NOT NULL
GROUP BY 1
ORDER BY document_count DESC
""")

print("📈 Document Size Distribution:")
display(size_distribution)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. AI-Powered Document Analysis
# MAGIC 
# MAGIC Now we'll process all documents using our two-tier strategy. This single SQL statement handles the complexity of:
# MAGIC 
# MAGIC - **Size categorization**: Automatically determining which strategy to use
# MAGIC - **Prompt customization**: Different analysis prompts for different document sizes
# MAGIC - **Error handling**: Graceful handling of analysis failures
# MAGIC - **Metadata tracking**: Recording processing details for quality assessment

# COMMAND ----------

# MAGIC %md
# MAGIC ### Two-Tier Processing Strategy
# MAGIC 
# MAGIC Our processing logic uses a `CASE` statement to apply different analysis approaches:
# MAGIC 
# MAGIC **Small Documents (Full Analysis):**
# MAGIC - Complete document content sent to AI model
# MAGIC - Comprehensive analysis covering key topics, insights, use cases, and summary
# MAGIC - Suitable for documents under our size thresholds
# MAGIC 
# MAGIC **Large Documents (Brief Analysis):**
# MAGIC - Content truncated to ~75k tokens (300k characters)
# MAGIC - Focused on main topics, purpose, and key points
# MAGIC - Includes truncation notice for transparency
# MAGIC 
# MAGIC Both approaches use the `ai_query()` function with `failOnError => false` to ensure processing continues even if individual documents fail.

# COMMAND ----------

# Process all documents with our two-tier analysis strategy
print("🚀 Starting AI-powered document analysis...")
print(f"   Using model: {AI_ENDPOINT}")
print(f"   Processing strategy: Size-based (small vs large)")

spark.sql(f"""
CREATE OR REPLACE TABLE {OUTPUT_TABLE} AS
SELECT 
    volume_path as file_path,
    file_name,
    character_count,
    ROUND(character_count / 4, 0) as estimated_tokens,
    file_size_bytes,
    ROUND(file_size_bytes / 1024 / 1024, 2) as file_size_mb,
    processing_category,
    
    -- Extract result and error from ai_query struct
    ai_response.result AS analysis_result,
    ai_response.errorMessage AS analysis_error,
    
    -- Metadata for tracking and debugging
    '{AI_ENDPOINT}' AS model_used,
    CURRENT_TIMESTAMP() AS processed_at
    
FROM (
    SELECT 
        volume_path,
        file_name,
        character_count,
        file_size_bytes,
        
        -- Determine processing category based on our thresholds
        CASE 
            WHEN (character_count / 4) < {TOKEN_THRESHOLD} 
             AND file_size_bytes < {SIZE_THRESHOLD_BYTES} THEN 'SMALL_DOC'
            ELSE 'LARGE_DOC'
        END AS processing_category,
        
        -- AI analysis with size-appropriate strategy (store full struct)
        CASE 
            -- Full comprehensive analysis for small documents
            WHEN (character_count / 4) < {TOKEN_THRESHOLD} 
             AND file_size_bytes < {SIZE_THRESHOLD_BYTES} THEN
                ai_query(
                    '{AI_ENDPOINT}',
                    REPLACE('{SMALL_DOC_PROMPT_SQL}', '{{content}}', markdown_content),
                    failOnError => false
                )
            
            -- Brief focused analysis for large documents (with truncation)
            ELSE
                ai_query(
                    '{AI_ENDPOINT}',
                    REPLACE('{LARGE_DOC_PROMPT_SQL}', '{{content}}', LEFT(markdown_content, 300000)),
                    failOnError => false
                )
        END AS ai_response
        
    FROM {SOURCE_TABLE}
    WHERE extraction_success = true
      AND markdown_content IS NOT NULL
      AND TRIM(markdown_content) != ''
) ai_analysis
""")

print("✅ Document analysis processing completed!")
print(f"   Results saved to: {OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Results Analysis and Quality Assessment
# MAGIC 
# MAGIC After processing, we need to evaluate the success of our analysis. This section provides comprehensive insights into:
# MAGIC 
# MAGIC - **Success rates** by document category
# MAGIC - **Processing statistics** for performance monitoring  
# MAGIC - **Sample outputs** for quality validation
# MAGIC - **Failure analysis** for troubleshooting

# COMMAND ----------

# MAGIC %md
# MAGIC ### Processing Summary Statistics
# MAGIC 
# MAGIC This analysis shows us how well our two-tier strategy performed across different document sizes. Key metrics include:
# MAGIC 
# MAGIC - **Success Rate**: Percentage of documents that received successful analysis
# MAGIC - **Average Metrics**: Token counts and file sizes for each category
# MAGIC - **Volume Distribution**: How many documents fell into each processing tier

# COMMAND ----------

# Generate comprehensive results summary
results_summary = spark.sql(f"""
SELECT 
    processing_category,
    COUNT(*) as total_documents,
    COUNT(CASE WHEN analysis_result IS NOT NULL AND TRIM(analysis_result) != '' THEN 1 END) as successful_analyses,
    COUNT(CASE WHEN analysis_result IS NULL OR TRIM(analysis_result) = '' THEN 1 END) as failed_analyses,
    ROUND(
        COUNT(CASE WHEN analysis_result IS NOT NULL AND TRIM(analysis_result) != '' THEN 1 END) * 100.0 / COUNT(*), 
        2
    ) as success_rate_percent,
    ROUND(AVG(estimated_tokens), 0) as avg_tokens,
    ROUND(AVG(file_size_mb), 2) as avg_size_mb,
    ROUND(MIN(estimated_tokens), 0) as min_tokens,
    ROUND(MAX(estimated_tokens), 0) as max_tokens
FROM {OUTPUT_TABLE}
GROUP BY processing_category
ORDER BY processing_category
""")

print("📊 Processing Results Summary:")
display(results_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample Analysis Results
# MAGIC 
# MAGIC Let's examine some actual analysis outputs to assess quality. This preview shows:
# MAGIC 
# MAGIC - **Diverse examples**: Both small and large document analyses
# MAGIC - **Content preview**: First 200 characters of each analysis
# MAGIC - **Size correlation**: How document size relates to analysis approach

# COMMAND ----------

# Show sample results to evaluate analysis quality
sample_results = spark.sql(f"""
SELECT 
    file_name,
    processing_category,
    estimated_tokens,
    file_size_mb,
    CASE 
        WHEN analysis_result IS NOT NULL THEN
            CONCAT(
                substr(cast(analysis_result as string), 1, 200),
                CASE WHEN length(cast(analysis_result as string)) > 200 THEN '...' ELSE '' END
            )
        ELSE 'FAILED - No analysis result'
    END as analysis_preview,
    CASE 
        WHEN analysis_result IS NOT NULL THEN length(cast(analysis_result as string))
        ELSE 0
    END as analysis_length
FROM {OUTPUT_TABLE}
ORDER BY estimated_tokens ASC
LIMIT 15
""")

print("📝 Sample Analysis Results (ordered by document size):")
display(sample_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Failed Documents Analysis
# MAGIC 
# MAGIC Understanding why some documents failed helps improve our processing pipeline. Common failure reasons include:
# MAGIC 
# MAGIC - **Content issues**: Empty or corrupted markdown content
# MAGIC - **Model limits**: Documents that exceed model context windows even after truncation
# MAGIC - **Format problems**: Unusual characters or encoding issues
# MAGIC - **Rate limiting**: Temporary API constraints

# COMMAND ----------

# Analyze failed documents for troubleshooting
failed_analysis = spark.sql(f"""
SELECT 
    processing_category,
    COUNT(*) as failed_count,
    ROUND(AVG(estimated_tokens), 0) as avg_tokens_failed,
    ROUND(AVG(file_size_mb), 2) as avg_size_mb_failed,
    -- Sample some failed document characteristics
    COLLECT_LIST(
        CASE WHEN row_num <= 3 
        THEN CONCAT(file_name, ' (', estimated_tokens, ' tokens)')
        END
    ) as sample_failed_docs
FROM (
    SELECT 
        processing_category,
        file_name,
        estimated_tokens,
        file_size_mb,
        ROW_NUMBER() OVER (PARTITION BY processing_category ORDER BY estimated_tokens) as row_num
    FROM {OUTPUT_TABLE}
    WHERE analysis_result IS NULL OR TRIM(analysis_result) = ''
) failed_docs
GROUP BY processing_category
ORDER BY failed_count DESC
""")

print("🔍 Failed Documents Analysis:")
failed_count = spark.sql(f"SELECT COUNT(*) as cnt FROM {OUTPUT_TABLE} WHERE analysis_result IS NULL OR TRIM(analysis_result) = ''").collect()[0]['cnt']

if failed_count > 0:
    print(f"   Found {failed_count} failed documents")
    display(failed_analysis)
else:
    print("   🎉 No failed documents found - 100% success rate!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Analysis Quality Deep Dive
# MAGIC 
# MAGIC Let's examine the quality and characteristics of our AI-generated analyses to understand what we're producing.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Content Length and Quality Metrics
# MAGIC 
# MAGIC This analysis helps us understand if our two-tier approach is producing appropriately detailed results for each document type.

# COMMAND ----------

# Analyze the quality and characteristics of our analyses
quality_analysis = spark.sql(f"""
SELECT 
    processing_category,
    COUNT(*) as total_analyses,
    -- Analysis length statistics
    ROUND(AVG(length(cast(analysis_result as string))), 0) as avg_analysis_length,
    ROUND(MIN(length(cast(analysis_result as string))), 0) as min_analysis_length,  
    ROUND(MAX(length(cast(analysis_result as string))), 0) as max_analysis_length,
    -- Content quality indicators
    ROUND(AVG(
        CASE WHEN analysis_result LIKE '%Key Topics%' THEN 1 ELSE 0 END * 100
    ), 1) as pct_with_key_topics,
    ROUND(AVG(
        CASE WHEN analysis_result LIKE '%Summary%' OR analysis_result LIKE '%summary%' THEN 1 ELSE 0 END * 100
    ), 1) as pct_with_summary,
    -- Document size correlation
    ROUND(AVG(estimated_tokens), 0) as avg_input_tokens,
    ROUND(AVG(file_size_mb), 2) as avg_input_size_mb
FROM {OUTPUT_TABLE}
WHERE analysis_result IS NOT NULL AND TRIM(analysis_result) != ''
GROUP BY processing_category
ORDER BY processing_category
""")

print("📈 Analysis Quality Metrics:")
display(quality_analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Processing Complete - Next Steps
# MAGIC 
# MAGIC Congratulations! You've successfully implemented AI-powered document analysis with size-adaptive processing strategies.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🎯 **What We Accomplished:**
# MAGIC 
# MAGIC 1. **Environment Integration**: Used `.env` files and widgets for flexible configuration
# MAGIC 2. **Size-Based Processing**: Implemented intelligent document categorization based on size and complexity
# MAGIC 3. **AI Integration**: Leveraged Databricks AI functions for scalable document analysis  
# MAGIC 4. **Quality Assessment**: Built comprehensive metrics to evaluate processing success
# MAGIC 5. **Error Handling**: Implemented graceful failure handling with detailed reporting
# MAGIC 
# MAGIC ### 📊 **Processing Strategy Benefits:**
# MAGIC 
# MAGIC - **Efficient Resource Use**: Small documents get full analysis, large ones get focused analysis
# MAGIC - **Scalable Approach**: Handles diverse document sizes without manual intervention
# MAGIC - **Quality Tracking**: Built-in metrics help monitor and improve the pipeline
# MAGIC - **Flexible Configuration**: Easy to adjust thresholds and strategies as needed
# MAGIC 
# MAGIC ### 🚀 **Next Steps in Your Document Processing Journey:**
# MAGIC 
# MAGIC - **Advanced Parsing** (Tutorial 02): Explore OCR and vision model techniques for complex documents
# MAGIC - **Vector Search** (Tutorial 03): Create searchable indexes from your analyzed documents  
# MAGIC - **Production Deployment** (Tutorial 04): Scale your pipeline for production workloads
# MAGIC - **Custom Models**: Fine-tune models for domain-specific document types

# COMMAND ----------

# Display final processing summary
total_docs = spark.sql(f"SELECT COUNT(*) as cnt FROM {OUTPUT_TABLE}").collect()[0]['cnt']
successful_docs = spark.sql(f"SELECT COUNT(*) as cnt FROM {OUTPUT_TABLE} WHERE analysis_result IS NOT NULL AND TRIM(analysis_result) != ''").collect()[0]['cnt']
overall_success_rate = (successful_docs / total_docs * 100) if total_docs > 0 else 0

print("=" * 60)
print("🎉 DOCUMENT ANALYSIS COMPLETE")
print("=" * 60)
print(f"📊 Final Results:")
print(f"   Total documents processed: {total_docs:,}")
print(f"   Successful analyses: {successful_docs:,}")
print(f"   Overall success rate: {overall_success_rate:.1f}%")
print(f"   Results table: {OUTPUT_TABLE}")
print()
print("✅ Your analyzed documents are now ready for:")
print("   • Search and retrieval applications")  
print("   • Further processing and enrichment")
print("   • Integration into knowledge management systems")
print("   • Vector database indexing for RAG applications")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC 
# MAGIC ## Tutorial Summary
# MAGIC 
# MAGIC **Adaptive Document Analysis Complete** ✅
# MAGIC 
# MAGIC You've built a robust document analysis pipeline that:
# MAGIC - 🔧 **Adapts processing** based on document characteristics
# MAGIC - 🎯 **Maximizes quality** while managing computational resources  
# MAGIC - 📊 **Tracks performance** with comprehensive metrics
# MAGIC - 🛡️ **Handles failures** gracefully with detailed error reporting
# MAGIC 
# MAGIC The results are stored in Delta Lake and ready for downstream applications like search, recommendation systems, or knowledge management platforms.