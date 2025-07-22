# Databricks notebook source
# MAGIC %md
# MAGIC ## Analyze Your Dataset for OCR Token Requirements
# MAGIC 
# MAGIC Run this analysis to determine if you need higher max-model-len or lower parallelism

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col, length, avg, max as spark_max, min as spark_min, percentile_approx
import matplotlib.pyplot as plt
import seaborn as sns

# Your table names
CATALOG = 'brian_gen_ai'
SCHEMA = 'parsing_test'
SOURCE_TABLE = 'document_page_docs'

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Analyze Image Sizes and Complexity

# COMMAND ----------

# Analyze your input images
image_analysis = spark.sql(f"""
SELECT 
    doc_id,
    source_filename,
    page_number,
    LENGTH(page_image_png) as image_size_bytes,
    total_pages,
    CASE 
        WHEN LENGTH(page_image_png) < 100000 THEN 'SMALL'
        WHEN LENGTH(page_image_png) < 500000 THEN 'MEDIUM' 
        WHEN LENGTH(page_image_png) < 1000000 THEN 'LARGE'
        ELSE 'VERY_LARGE'
    END as image_size_category
FROM {CATALOG}.{SCHEMA}.{SOURCE_TABLE}
WHERE page_image_png IS NOT NULL
""")

print("=== IMAGE SIZE DISTRIBUTION ===")
image_stats = image_analysis.groupBy("image_size_category").count().orderBy("count", ascending=False)
display(image_stats)

# Get detailed stats
detailed_stats = image_analysis.agg(
    spark_min("image_size_bytes").alias("min_size"),
    avg("image_size_bytes").alias("avg_size"), 
    spark_max("image_size_bytes").alias("max_size"),
    percentile_approx("image_size_bytes", 0.5).alias("median_size"),
    percentile_approx("image_size_bytes", 0.95).alias("p95_size")
).collect()[0]

print(f"Image Size Stats:")
print(f"  Min: {detailed_stats['min_size']:,.0f} bytes")
print(f"  Avg: {detailed_stats['avg_size']:,.0f} bytes") 
print(f"  Median: {detailed_stats['median_size']:,.0f} bytes")
print(f"  95th percentile: {detailed_stats['p95_size']:,.0f} bytes")
print(f"  Max: {detailed_stats['max_size']:,.0f} bytes")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Analyze Existing OCR Output Lengths (if available)

# COMMAND ----------

# Check if you have any OCR results already
try:
    ocr_results = spark.table(f"{CATALOG}.{SCHEMA}.document_store_ocr")
    
    ocr_analysis = ocr_results.filter(col("ocr_text").isNotNull()).select(
        "doc_id",
        "source_filename", 
        "page_number",
        length("ocr_text").alias("text_length")
    )
    
    print("=== EXISTING OCR TEXT LENGTH ANALYSIS ===")
    
    # Text length distribution
    text_length_dist = ocr_analysis.select(
        ((col("text_length") / 1000).cast("int") * 1000).alias("length_bucket")
    ).groupBy("length_bucket").count().orderBy("length_bucket")
    
    display(text_length_dist)
    
    # Stats
    text_stats = ocr_analysis.agg(
        spark_min("text_length").alias("min_length"),
        avg("text_length").alias("avg_length"),
        spark_max("text_length").alias("max_length"),
        percentile_approx("text_length", 0.5).alias("median_length"),
        percentile_approx("text_length", 0.95).alias("p95_length"),
        percentile_approx("text_length", 0.99).alias("p99_length")
    ).collect()[0]
    
    print(f"OCR Text Length Stats:")
    print(f"  Min: {text_stats['min_length']:,.0f} chars")
    print(f"  Avg: {text_stats['avg_length']:,.0f} chars")
    print(f"  Median: {text_stats['median_length']:,.0f} chars") 
    print(f"  95th percentile: {text_stats['p95_length']:,.0f} chars")
    print(f"  99th percentile: {text_stats['p99_length']:,.0f} chars")
    print(f"  Max: {text_stats['max_length']:,.0f} chars")
    
    # Estimate token requirements (roughly 4 chars per token)
    estimated_tokens_p95 = text_stats['p95_length'] / 4
    estimated_tokens_p99 = text_stats['p99_length'] / 4
    estimated_tokens_max = text_stats['max_length'] / 4
    
    print(f"\nEstimated Token Requirements:")
    print(f"  95th percentile: ~{estimated_tokens_p95:,.0f} tokens")
    print(f"  99th percentile: ~{estimated_tokens_p99:,.0f} tokens") 
    print(f"  Max: ~{estimated_tokens_max:,.0f} tokens")
    
    # Recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    if estimated_tokens_p95 < 2000:
        print("✅ Current max-model-len (4096) should be sufficient for 95% of documents")
        print("🔧 Focus on optimizing parallelism instead")
    elif estimated_tokens_p95 < 4000:
        print("⚠️  Current max-model-len (4096) is borderline for complex documents")
        print("🔧 Consider increasing to 6144 or 8192")
    else:
        print("❌ Current max-model-len (4096) is too low for your dataset")
        print("🔧 Increase to at least 8192 or 12288")
        
    if estimated_tokens_max > 8000:
        print("⚠️  Some documents may require very large context windows")
        print("🔧 Consider preprocessing to split large documents")
    
except Exception as e:
    print("No existing OCR results found. Let's estimate from image complexity...")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Sample a Few Images for Token Estimation

# COMMAND ----------

# Sample a few images for testing
sample_images = spark.sql(f"""
SELECT 
    doc_id,
    source_filename,
    page_number,
    page_image_png,
    LENGTH(page_image_png) as image_size_bytes
FROM {CATALOG}.{SCHEMA}.{SOURCE_TABLE}
WHERE page_image_png IS NOT NULL
ORDER BY RAND()
LIMIT 5
""")

print("=== SAMPLE IMAGES FOR TESTING ===")
display(sample_images.select("doc_id", "source_filename", "page_number", "image_size_bytes"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Test OCR on Sample Images

# COMMAND ----------

# Test OCR on sample to estimate token requirements
import requests
import base64
import io
from PIL import Image

def estimate_ocr_tokens_sample():
    """Test OCR on a few sample images to estimate token requirements"""
    
    # Get sample data
    sample_data = sample_images.collect()
    
    if not sample_data:
        print("No sample data available")
        return
    
    results = []
    
    for row in sample_data[:3]:  # Test first 3 samples
        try:
            # Convert bytes to PIL Image
            image_bytes = row['page_image_png']
            image = Image.open(io.BytesIO(image_bytes))
            
            # Quick estimation based on image properties
            width, height = image.size
            pixel_count = width * height
            
            # Rough estimation: complex images with lots of text
            # can produce 0.1-0.5 tokens per pixel for dense documents
            estimated_tokens_low = pixel_count * 0.05 / 1000  # Conservative
            estimated_tokens_high = pixel_count * 0.2 / 1000  # Dense text
            
            results.append({
                'filename': row['source_filename'],
                'page': row['page_number'],
                'image_size_mb': row['image_size_bytes'] / 1024 / 1024,
                'resolution': f"{width}x{height}",
                'pixels': pixel_count,
                'estimated_tokens_low': estimated_tokens_low,
                'estimated_tokens_high': estimated_tokens_high
            })
            
        except Exception as e:
            print(f"Error processing sample: {e}")
    
    if results:
        print("=== TOKEN ESTIMATION FROM SAMPLE IMAGES ===")
        for r in results:
            print(f"📄 {r['filename']} (page {r['page']}):")
            print(f"   Size: {r['image_size_mb']:.1f}MB, Resolution: {r['resolution']}")
            print(f"   Estimated tokens: {r['estimated_tokens_low']:.0f} - {r['estimated_tokens_high']:.0f}")
        
        # Overall recommendations
        max_estimated = max(r['estimated_tokens_high'] for r in results)
        avg_estimated = sum(r['estimated_tokens_high'] for r in results) / len(results)
        
        print(f"\n=== SAMPLE-BASED RECOMMENDATIONS ===")
        print(f"Average estimated tokens: {avg_estimated:.0f}")
        print(f"Max estimated tokens: {max_estimated:.0f}")
        
        if max_estimated < 2000:
            print("✅ max-model-len 4096 should be sufficient")
            recommended_len = 4096
        elif max_estimated < 4000:
            print("⚠️  Consider max-model-len 6144-8192")
            recommended_len = 8192
        else:
            print("❌ Need max-model-len 8192+ for complex documents")
            recommended_len = 12288
            
        return recommended_len
    
    return 4096

recommended_max_len = estimate_ocr_tokens_sample()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Configuration Recommendations

# COMMAND ----------

def generate_recommendations(max_model_len, total_pages):
    """Generate vLLM server and client configurations based on analysis"""
    
    print(f"=== CONFIGURATION RECOMMENDATIONS ===")
    print(f"Based on your dataset analysis:\n")
    
    # Calculate memory requirements
    # Rough estimation: each token in KV cache uses ~16 bytes (fp16) or 8 bytes (fp8)
    tokens_per_seq = max_model_len
    
    if max_model_len <= 4096:
        print("📋 **SCENARIO: Standard Documents (max-model-len 4096)**")
        
        # Higher parallelism possible
        configs = [
            {
                'name': 'High Throughput',
                'max_num_seqs': 24,
                'max_num_batched_tokens': 16384,
                'max_workers': 8,
                'max_concurrent_images': 8
            },
            {
                'name': 'Balanced',
                'max_num_seqs': 16,
                'max_num_batched_tokens': 12288,
                'max_workers': 6,
                'max_concurrent_images': 6
            }
        ]
        
    elif max_model_len <= 8192:
        print("📋 **SCENARIO: Complex Documents (max-model-len 8192)**")
        
        # Medium parallelism 
        configs = [
            {
                'name': 'Balanced',
                'max_num_seqs': 12,
                'max_num_batched_tokens': 16384,
                'max_workers': 4,
                'max_concurrent_images': 4
            },
            {
                'name': 'Conservative',
                'max_num_seqs': 8,
                'max_num_batched_tokens': 12288,
                'max_workers': 3,
                'max_concurrent_images': 3
            }
        ]
        
    else:
        print("📋 **SCENARIO: Very Complex Documents (max-model-len 12288+)**")
        
        # Lower parallelism
        configs = [
            {
                'name': 'Quality Focused',
                'max_num_seqs': 6,
                'max_num_batched_tokens': 12288,
                'max_workers': 2,
                'max_concurrent_images': 2
            },
            {
                'name': 'Ultra Conservative',
                'max_num_seqs': 4,
                'max_num_batched_tokens': 8192,
                'max_workers': 2,
                'max_concurrent_images': 2
            }
        ]
    
    for config in configs:
        print(f"\n### {config['name']} Configuration")
        
        # Server command
        print("**vLLM Server:**")
        print("```bash")
        print(f"vllm serve nanonets/Nanonets-OCR-s \\")
        print(f"  --host 0.0.0.0 --port 8000 \\")
        print(f"  --max-num-batched-tokens {config['max_num_batched_tokens']} \\")
        print(f"  --max-num-seqs {config['max_num_seqs']} \\")
        print(f"  --max-model-len {max_model_len} \\")
        print(f"  --limit-mm-per-prompt \"image=1\" \\")
        print(f"  --gpu-memory-utilization 0.85 \\")
        print(f"  --trust-remote-code")
        print("```")
        
        # Client environment
        print("\n**Client Environment Variables:**")
        print("```bash")
        print(f"export MAX_WORKERS={config['max_workers']}")
        print(f"export MAX_CONCURRENT_IMAGES={config['max_concurrent_images']}")
        print(f"export INITIAL_WORKERS={max(2, config['max_workers']//2)}")
        print("```")
        
        # Estimated performance
        est_throughput = config['max_workers'] * 0.5  # rough estimate
        est_time = total_pages / est_throughput / 60 if total_pages > 0 else 0
        print(f"\n**Estimated Performance:**")
        print(f"  - Throughput: ~{est_throughput:.1f} pages/second")
        if est_time > 0:
            print(f"  - Total time for {total_pages:,} pages: ~{est_time:.1f} minutes")

# Get total pages for estimation
total_pages = spark.table(f"{CATALOG}.{SCHEMA}.{SOURCE_TABLE}").count()

generate_recommendations(recommended_max_len, total_pages)