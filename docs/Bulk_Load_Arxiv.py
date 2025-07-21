# Databricks notebook source
# MAGIC %md
# MAGIC # LLM Papers Downloader - Unity Catalog Volume Edition
# MAGIC
# MAGIC This notebook downloads 100 foundational and recent papers on Large Language Models from arXiv and stores them in a Unity Catalog Volume.
# MAGIC
# MAGIC ## Features:
# MAGIC - 100 carefully curated LLM papers from 2017-2025
# MAGIC - Organized by research categories (foundational, survey, training, etc.)
# MAGIC - Automatic retry logic and error handling
# MAGIC - Progress tracking and resumable downloads
# MAGIC - **Stores in Unity Catalog Volume for better governance**
# MAGIC - **Parameterized volume configuration**
# MAGIC
# MAGIC ## Unity Catalog Benefits:
# MAGIC - **Governance**: Fine-grained access control
# MAGIC - **Lineage**: Track data usage and dependencies  
# MAGIC - **Security**: Enterprise-grade security controls
# MAGIC - **Cross-workspace**: Access from any workspace in your metastore
# MAGIC - **Versioning**: Better data versioning capabilities

# COMMAND ----------

# Clean up corrupted/blank PDF files
def cleanup_corrupted_files():
    """Remove files that are too small (likely corrupted)"""
    try:
        items = dbutils.fs.ls(UC_VOLUME_PATH)
        removed_count = 0
        
        for item in items:
            if item.isDir():
                # Check files in category directories
                try:
                    files = dbutils.fs.ls(item.path)
                    for file in files:
                        if file.name.endswith('.pdf'):
                            size_mb = file.size / (1024 * 1024)
                            if size_mb < 0.1:  # Less than 100KB
                                print(f"🗑️ Removing corrupted: {file.name} ({size_mb:.1f} MB)")
                                dbutils.fs.rm(file.path)
                                removed_count += 1
                except:
                    pass
            elif item.name.endswith('.pdf'):
                # Check files in root
                size_mb = item.size / (1024 * 1024)
                if size_mb < 0.1:
                    print(f"🗑️ Removing corrupted: {item.name} ({size_mb:.1f} MB)")
                    dbutils.fs.rm(item.path)
                    removed_count += 1
        
        print(f"✅ Cleaned up {removed_count} corrupted files")
        
        # Also clean up any leftover local files
        import os
        import glob
        local_pdfs = glob.glob("/local_disk0/*.pdf")
        for pdf in local_pdfs:
            try:
                os.remove(pdf)
                print(f"🧹 Cleaned local temp file: {os.path.basename(pdf)}")
            except:
                pass
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧹 Clean Up Corrupted Files
# MAGIC
# MAGIC Run this to remove any blank/corrupted PDF files from previous downloads:

# COMMAND ----------

# Run cleanup of corrupted files
cleanup_corrupted_files()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎛️ Configuration Parameters
# MAGIC
# MAGIC Configure your Unity Catalog volume details below:

# COMMAND ----------

# Create widgets for UC volume configuration
dbutils.widgets.text("catalog_name", "main", "UC Catalog Name")
dbutils.widgets.text("schema_name", "default", "UC Schema Name") 
dbutils.widgets.text("volume_name", "llm_papers", "UC Volume Name")
dbutils.widgets.dropdown("organize_by_category", "true", ["true", "false"], "Organize by Category")
dbutils.widgets.multiselect("categories_to_download", "all", 
                           ["all", "foundational", "survey", "training", "alignment", "efficiency", 
                            "capabilities", "multimodal", "code", "safety", "reasoning"], 
                           "Categories to Download")

# Get parameter values
CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name") 
VOLUME_NAME = dbutils.widgets.get("volume_name")
ORGANIZE_BY_CATEGORY = dbutils.widgets.get("organize_by_category").lower() == "true"
CATEGORIES_TO_DOWNLOAD = dbutils.widgets.get("categories_to_download").split(",")

# Construct UC volume path
UC_VOLUME_PATH = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/{VOLUME_NAME}"

print("🎛️ CONFIGURATION")
print("=" * 50)
print(f"📁 UC Volume Path: {UC_VOLUME_PATH}")
print(f"🗂️  Organize by Category: {ORGANIZE_BY_CATEGORY}")
print(f"🎯 Categories to Download: {CATEGORIES_TO_DOWNLOAD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup and Imports

# COMMAND ----------

import os
import requests
import time
from urllib.parse import urlparse
import re
from collections import defaultdict, Counter

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏗️ Unity Catalog Volume Setup

# COMMAND ----------

def setup_uc_volume():
    """Create UC volume if it doesn't exist"""
    try:
        # Check if volume exists
        volumes = spark.sql(f"SHOW VOLUMES IN {CATALOG_NAME}.{SCHEMA_NAME}").collect()
        volume_exists = any(vol.volume_name == VOLUME_NAME for vol in volumes)
        
        if not volume_exists:
            print(f"📁 Creating UC Volume: {CATALOG_NAME}.{SCHEMA_NAME}.{VOLUME_NAME}")
            spark.sql(f"""
                CREATE VOLUME IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}.{VOLUME_NAME}
                COMMENT 'LLM Research Papers Collection - arXiv PDFs organized by research category'
            """)
            print("✅ Volume created successfully")
        else:
            print(f"✅ Volume {CATALOG_NAME}.{SCHEMA_NAME}.{VOLUME_NAME} already exists")
            
        # Test volume access
        try:
            dbutils.fs.ls(UC_VOLUME_PATH)
            print(f"✅ Volume accessible at: {UC_VOLUME_PATH}")
        except Exception as e:
            print(f"❌ Error accessing volume: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error setting up UC volume: {e}")
        print("Please ensure you have CREATE VOLUME permissions on the catalog/schema")
        return False

# Setup the volume
volume_ready = setup_uc_volume()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 Papers Database
# MAGIC Complete collection of 100 LLM papers with metadata

# COMMAND ----------

PAPERS = [
    # Foundational & Seminal Papers
    {
        "title": "Attention Is All You Need",
        "authors": "Vaswani et al.",
        "year": "2017",
        "url": "https://arxiv.org/pdf/1706.03762.pdf",
        "category": "foundational",
        "arxiv_id": "1706.03762"
    },
    {
        "title": "Improving Language Understanding by Generative Pre-Training (GPT-1)",
        "authors": "Radford et al.",
        "year": "2018", 
        "url": "https://arxiv.org/pdf/1810.04805.pdf",
        "category": "foundational",
        "arxiv_id": "1810.04805"
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": "Devlin et al.",
        "year": "2018",
        "url": "https://arxiv.org/pdf/1810.04805.pdf",
        "category": "foundational",
        "arxiv_id": "1810.04805"
    },
    {
        "title": "Language Models are Unsupervised Multitask Learners (GPT-2)",
        "authors": "Radford et al.",
        "year": "2019",
        "url": "https://arxiv.org/pdf/1909.11942.pdf",
        "category": "foundational",
        "arxiv_id": "1909.11942"
    },
    {
        "title": "Language Models are Few-Shot Learners (GPT-3)",
        "authors": "Brown et al.",
        "year": "2020",
        "url": "https://arxiv.org/pdf/2005.14165.pdf",
        "category": "foundational",
        "arxiv_id": "2005.14165"
    },
    {
        "title": "T5: Exploring the Limits of Transfer Learning",
        "authors": "Raffel et al.",
        "year": "2019",
        "url": "https://arxiv.org/pdf/1910.10683.pdf",
        "category": "foundational",
        "arxiv_id": "1910.10683"
    },
    {
        "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        "authors": "Liu et al.",
        "year": "2019",
        "url": "https://arxiv.org/pdf/1907.11692.pdf",
        "category": "foundational",
        "arxiv_id": "1907.11692"
    },
    {
        "title": "ELECTRA: Pre-training Text Encoders as Discriminators",
        "authors": "Clark et al.",
        "year": "2020",
        "url": "https://arxiv.org/pdf/2003.10555.pdf",
        "category": "foundational",
        "arxiv_id": "2003.10555"
    },
    {
        "title": "XLNet: Generalized Autoregressive Pretraining",
        "authors": "Yang et al.",
        "year": "2019",
        "url": "https://arxiv.org/pdf/1906.08237.pdf",
        "category": "foundational",
        "arxiv_id": "1906.08237"
    },
    {
        "title": "DeBERTa: Decoding-enhanced BERT with Disentangled Attention",
        "authors": "He et al.",
        "year": "2020",
        "url": "https://arxiv.org/pdf/2006.03654.pdf",
        "category": "foundational",
        "arxiv_id": "2006.03654"
    },
    
    # Survey Papers
    {
        "title": "A Survey of Large Language Models",
        "authors": "Zhao et al.",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2303.18223.pdf",
        "category": "survey",
        "arxiv_id": "2303.18223"
    },
    {
        "title": "Large Language Models: A Survey",
        "authors": "Minaee et al.",
        "year": "2024",
        "url": "https://arxiv.org/pdf/2402.06196.pdf",
        "category": "survey",
        "arxiv_id": "2402.06196"
    },
    {
        "title": "A Comprehensive Overview of Large Language Models",
        "authors": "Naveed et al.",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2307.06435.pdf",
        "category": "survey",
        "arxiv_id": "2307.06435"
    },
    {
        "title": "Foundations of Large Language Models",
        "authors": "Xiao et al.",
        "year": "2025",
        "url": "https://arxiv.org/pdf/2501.09223.pdf",
        "category": "survey",
        "arxiv_id": "2501.09223"
    },
    {
        "title": "Large Language Models",
        "authors": "Douglas",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2307.05782.pdf",
        "category": "survey",
        "arxiv_id": "2307.05782"
    },
    {
        "title": "Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods",
        "authors": "Liu et al.",
        "year": "2021",
        "url": "https://arxiv.org/pdf/2107.13586.pdf",
        "category": "survey",
        "arxiv_id": "2107.13586"
    },
    
    # Training & Scaling
    {
        "title": "Scaling Laws for Neural Language Models",
        "authors": "Kaplan et al.",
        "year": "2020",
        "url": "https://arxiv.org/pdf/2001.08361.pdf",
        "category": "training",
        "arxiv_id": "2001.08361"
    },
    {
        "title": "Training Compute-Optimal Large Language Models",
        "authors": "Hoffmann et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2203.15556.pdf",
        "category": "training",
        "arxiv_id": "2203.15556"
    },
    {
        "title": "PaLM: Scaling Language Modeling with Pathways",
        "authors": "Chowdhery et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2204.02311.pdf",
        "category": "training",
        "arxiv_id": "2204.02311"
    },
    {
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "authors": "Touvron et al.",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2302.13971.pdf",
        "category": "training",
        "arxiv_id": "2302.13971"
    },
    {
        "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        "authors": "Touvron et al.",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2307.09288.pdf",
        "category": "training",
        "arxiv_id": "2307.09288"
    },
    {
        "title": "GPT-4 Technical Report",
        "authors": "OpenAI",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2303.08774.pdf",
        "category": "training",
        "arxiv_id": "2303.08774"
    },
    {
        "title": "Gopher: Scaling Language Modeling with 280B Parameters",
        "authors": "Rae et al.",
        "year": "2021",
        "url": "https://arxiv.org/pdf/2112.11446.pdf",
        "category": "training",
        "arxiv_id": "2112.11446"
    },
    {
        "title": "OPT: Open Pre-trained Transformer Language Models",
        "authors": "Zhang et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2205.01068.pdf",
        "category": "training",
        "arxiv_id": "2205.01068"
    },
    {
        "title": "GLM-130B: An Open Bilingual Pre-trained Model",
        "authors": "Zeng et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2210.02414.pdf",
        "category": "training",
        "arxiv_id": "2210.02414"
    },
    {
        "title": "The Pile: An 800GB Dataset of Diverse Text for Language Modeling",
        "authors": "Gao et al.",
        "year": "2020",
        "url": "https://arxiv.org/pdf/2101.00027.pdf",
        "category": "training",
        "arxiv_id": "2101.00027"
    },
    
    # Instruction Following & Alignment
    {
        "title": "Training Language Models to Follow Instructions with Human Feedback",
        "authors": "Ouyang et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2203.02155.pdf",
        "category": "alignment",
        "arxiv_id": "2203.02155"
    },
    {
        "title": "Constitutional AI: Harmlessness from AI Feedback",
        "authors": "Bai et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2212.08073.pdf",
        "category": "alignment",
        "arxiv_id": "2212.08073"
    },
    {
        "title": "Direct Preference Optimization",
        "authors": "Rafailov et al.",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2305.18290.pdf",
        "category": "alignment",
        "arxiv_id": "2305.18290"
    },
    {
        "title": "Self-Instruct: Aligning Language Models with Self-Generated Instructions",
        "authors": "Wang et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2212.10560.pdf",
        "category": "alignment",
        "arxiv_id": "2212.10560"
    },
    {
        "title": "Learning to Summarize from Human Feedback",
        "authors": "Stiennon et al.",
        "year": "2020",
        "url": "https://arxiv.org/pdf/2009.01325.pdf",
        "category": "alignment",
        "arxiv_id": "2009.01325"
    },
    {
        "title": "LIMA: Less Is More for Alignment",
        "authors": "Zhou et al.",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2305.11206.pdf",
        "category": "alignment",
        "arxiv_id": "2305.11206"
    },
    {
        "title": "Anthropic's Constitutional AI",
        "authors": "Askell et al.",
        "year": "2021",
        "url": "https://arxiv.org/pdf/2112.00861.pdf",
        "category": "alignment",
        "arxiv_id": "2112.00861"
    },
    
    # Efficiency & Optimization
    {
        "title": "LLM in a Flash: Efficient Large Language Model Inference with Limited Memory",
        "authors": "Alizadeh et al.",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2312.11514.pdf",
        "category": "efficiency",
        "arxiv_id": "2312.11514"
    },
    {
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": "Hu et al.",
        "year": "2021",
        "url": "https://arxiv.org/pdf/2106.09685.pdf",
        "category": "efficiency",
        "arxiv_id": "2106.09685"
    },
    {
        "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
        "authors": "Dettmers et al.",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2305.14314.pdf",
        "category": "efficiency",
        "arxiv_id": "2305.14314"
    },
    {
        "title": "FlashAttention: Fast and Memory-Efficient Exact Attention",
        "authors": "Dao et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2205.14135.pdf",
        "category": "efficiency",
        "arxiv_id": "2205.14135"
    },
    {
        "title": "AdapterHub: A Framework for Adapting Transformers",
        "authors": "Pfeiffer et al.",
        "year": "2020",
        "url": "https://arxiv.org/pdf/2007.07779.pdf",
        "category": "efficiency",
        "arxiv_id": "2007.07779"
    },
    {
        "title": "Parameter-Efficient Transfer Learning for NLP",
        "authors": "Houlsby et al.",
        "year": "2019",
        "url": "https://arxiv.org/pdf/1902.00751.pdf",
        "category": "efficiency",
        "arxiv_id": "1902.00751"
    },
    {
        "title": "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning",
        "authors": "Zhang et al.",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2303.10512.pdf",
        "category": "efficiency",
        "arxiv_id": "2303.10512"
    },
    {
        "title": "8-bit Methods for Efficient Deep Learning",
        "authors": "Dettmers et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2208.07339.pdf",
        "category": "efficiency",
        "arxiv_id": "2208.07339"
    },
    {
        "title": "FlashAttention-2: Faster Attention with Better Parallelism",
        "authors": "Dao",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2307.08691.pdf",
        "category": "efficiency",
        "arxiv_id": "2307.08691"
    },
    
    # Emergent Abilities & Capabilities
    {
        "title": "Emergent Abilities of Large Language Models",
        "authors": "Wei et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2206.07682.pdf",
        "category": "capabilities",
        "arxiv_id": "2206.07682"
    },
    {
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "authors": "Wei et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2201.11903.pdf",
        "category": "capabilities",
        "arxiv_id": "2201.11903"
    },
    {
        "title": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
        "authors": "Yao et al.",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2305.10601.pdf",
        "category": "capabilities",
        "arxiv_id": "2305.10601"
    },
    {
        "title": "In-Context Learning and Induction Heads",
        "authors": "Olsson et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2209.11895.pdf",
        "category": "capabilities",
        "arxiv_id": "2209.11895"
    },
    {
        "title": "What Can Transformers Learn In-Context?",
        "authors": "Garg et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2402.15049.pdf",
        "category": "capabilities",
        "arxiv_id": "2402.15049"
    },
    {
        "title": "Self-Consistency Improves Chain of Thought Reasoning",
        "authors": "Wang et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2203.11171.pdf",
        "category": "capabilities",
        "arxiv_id": "2203.11171"
    },
    {
        "title": "Least-to-Most Prompting Enables Complex Reasoning",
        "authors": "Zhou et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2205.10625.pdf",
        "category": "capabilities",
        "arxiv_id": "2205.10625"
    },
    {
        "title": "Large Language Models are Zero-Shot Reasoners",
        "authors": "Kojima et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2205.11916.pdf",
        "category": "capabilities",
        "arxiv_id": "2205.11916"
    },
    
    # Multimodal & Vision-Language
    {
        "title": "CLIP: Learning Transferable Visual Representations from Natural Language",
        "authors": "Radford et al.",
        "year": "2021",
        "url": "https://arxiv.org/pdf/2103.00020.pdf",
        "category": "multimodal",
        "arxiv_id": "2103.00020"
    },
    {
        "title": "GPT-4V(ision) System Card",
        "authors": "OpenAI",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2310.12809.pdf",
        "category": "multimodal",
        "arxiv_id": "2310.12809"
    },
    {
        "title": "LLaVA: Large Language and Vision Assistant",
        "authors": "Liu et al.",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2304.08485.pdf",
        "category": "multimodal",
        "arxiv_id": "2304.08485"
    },
    {
        "title": "DALL-E: Creating Images from Text",
        "authors": "Ramesh et al.",
        "year": "2021",
        "url": "https://arxiv.org/pdf/2102.12092.pdf",
        "category": "multimodal",
        "arxiv_id": "2102.12092"
    },
    {
        "title": "DALL-E 2: Hierarchical Text-Conditional Image Generation",
        "authors": "Ramesh et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2204.06125.pdf",
        "category": "multimodal",
        "arxiv_id": "2204.06125"
    },
    {
        "title": "Flamingo: a Visual Language Model for Few-Shot Learning",
        "authors": "Alayrac et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2204.14198.pdf",
        "category": "multimodal",
        "arxiv_id": "2204.14198"
    },
    {
        "title": "BLIP: Bootstrapping Language-Image Pre-training",
        "authors": "Li et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2201.12086.pdf",
        "category": "multimodal",
        "arxiv_id": "2201.12086"
    },
    {
        "title": "InstructBLIP: Towards General-purpose Vision-Language Models",
        "authors": "Dai et al.",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2305.06500.pdf",
        "category": "multimodal",
        "arxiv_id": "2305.06500"
    },
    
    # Code Generation
    {
        "title": "Codex: Evaluating Large Language Models Trained on Code",
        "authors": "Chen et al.",
        "year": "2021",
        "url": "https://arxiv.org/pdf/2107.03374.pdf",
        "category": "code",
        "arxiv_id": "2107.03374"
    },
    {
        "title": "CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models",
        "authors": "Wang et al.",
        "year": "2021",
        "url": "https://arxiv.org/pdf/2109.00859.pdf",
        "category": "code",
        "arxiv_id": "2109.00859"
    },
    {
        "title": "Competition-level Code Generation with AlphaCode",
        "authors": "Li et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2203.07814.pdf",
        "category": "code",
        "arxiv_id": "2203.07814"
    },
    {
        "title": "CodeBERT: A Pre-Trained Model for Programming and Natural Languages",
        "authors": "Feng et al.",
        "year": "2020",
        "url": "https://arxiv.org/pdf/2002.08155.pdf",
        "category": "code",
        "arxiv_id": "2002.08155"
    },
    {
        "title": "CodeGen: An Open Large Language Model for Code",
        "authors": "Nijkamp et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2203.13474.pdf",
        "category": "code",
        "arxiv_id": "2203.13474"
    },
    {
        "title": "InCoder: A Generative Model for Code Infilling and Synthesis",
        "authors": "Fried et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2204.05999.pdf",
        "category": "code",
        "arxiv_id": "2204.05999"
    },
    {
        "title": "PaLM-Coder: Better Code Generation from Pre-trained Language Models",
        "authors": "Chowdhery et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2204.11454.pdf",
        "category": "code",
        "arxiv_id": "2204.11454"
    },
    
    # Safety & Interpretability
    {
        "title": "Red Teaming Language Models with Language Models",
        "authors": "Perez et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2202.03286.pdf",
        "category": "safety",
        "arxiv_id": "2202.03286"
    },
    {
        "title": "Language Models (Mostly) Know What They Know",
        "authors": "Kadavath et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2207.05221.pdf",
        "category": "safety",
        "arxiv_id": "2207.05221"
    },
    {
        "title": "Discovering Latent Knowledge in Language Models Without Supervision",
        "authors": "Burns et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2212.03827.pdf",
        "category": "safety",
        "arxiv_id": "2212.03827"
    },
    {
        "title": "Language Models as Zero-Shot Planners",
        "authors": "Huang et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2201.07207.pdf",
        "category": "safety",
        "arxiv_id": "2201.07207"
    },
    {
        "title": "Measuring and Narrowing the Compositionality Gap in Language Models",
        "authors": "Press et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2210.03350.pdf",
        "category": "safety",
        "arxiv_id": "2210.03350"
    },
    {
        "title": "Truthful AI: Developing and Governing AI that Does Not Lie",
        "authors": "Evans et al.",
        "year": "2021",
        "url": "https://arxiv.org/pdf/2110.06674.pdf",
        "category": "safety",
        "arxiv_id": "2110.06674"
    },
    {
        "title": "Locating and Editing Factual Associations in GPT",
        "authors": "Meng et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2202.05262.pdf",
        "category": "safety",
        "arxiv_id": "2202.05262"
    },
    {
        "title": "Gender Bias in Neural Natural Language Processing",
        "authors": "Shah et al.",
        "year": "2020",
        "url": "https://arxiv.org/pdf/1807.11714.pdf",
        "category": "safety",
        "arxiv_id": "1807.11714"
    },
    
    # Reasoning & Problem Solving
    {
        "title": "PaLM 2 Technical Report",
        "authors": "Anil et al.",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2305.10403.pdf",
        "category": "reasoning",
        "arxiv_id": "2305.10403"
    },
    {
        "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
        "authors": "Schick et al.",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2302.04761.pdf",
        "category": "reasoning",
        "arxiv_id": "2302.04761"
    },
    {
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "authors": "Yao et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2210.03629.pdf",
        "category": "reasoning",
        "arxiv_id": "2210.03629"
    },
    {
        "title": "WebGPT: Browser-assisted Question-answering with Human Feedback",
        "authors": "Nakano et al.",
        "year": "2021",
        "url": "https://arxiv.org/pdf/2112.09332.pdf",
        "category": "reasoning",
        "arxiv_id": "2112.09332"
    },
    {
        "title": "Solving Quantitative Reasoning Problems with Language Models",
        "authors": "Lewkowycz et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2206.14858.pdf",
        "category": "reasoning",
        "arxiv_id": "2206.14858"
    },
    {
        "title": "STaR: Bootstrapping Reasoning With Reasoning",
        "authors": "Zelikman et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2203.14465.pdf",
        "category": "reasoning",
        "arxiv_id": "2203.14465"
    },
    {
        "title": "Program-aided Language Models",
        "authors": "Gao et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2211.10435.pdf",
        "category": "reasoning",
        "arxiv_id": "2211.10435"
    },
    {
        "title": "Language Model Cascades",
        "authors": "Dohan et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2207.10342.pdf",
        "category": "reasoning",
        "arxiv_id": "2207.10342"
    },
    
    # Fine-tuning & Adaptation
    {
        "title": "Finetuned Language Models Are Zero-Shot Learners",
        "authors": "Wei et al.",
        "year": "2021",
        "url": "https://arxiv.org/pdf/2109.01652.pdf",
        "category": "finetuning",
        "arxiv_id": "2109.01652"
    },
    {
        "title": "The Power of Scale for Parameter-Efficient Prompt Tuning",
        "authors": "Lester et al.",
        "year": "2021",
        "url": "https://arxiv.org/pdf/2104.08691.pdf",
        "category": "finetuning",
        "arxiv_id": "2104.08691"
    },
    {
        "title": "Learning to Retrieve Prompts for In-Context Learning",
        "authors": "Rubin et al.",
        "year": "2021",
        "url": "https://arxiv.org/pdf/2112.08633.pdf",
        "category": "finetuning",
        "arxiv_id": "2112.08633"
    },
    {
        "title": "What Makes Good In-Context Examples for GPT-3?",
        "authors": "Liu et al.",
        "year": "2021",
        "url": "https://arxiv.org/pdf/2101.06804.pdf",
        "category": "finetuning",
        "arxiv_id": "2101.06804"
    },
    {
        "title": "Calibrate Before Use: Improving Few-Shot Performance of Language Models",
        "authors": "Zhao et al.",
        "year": "2021",
        "url": "https://arxiv.org/pdf/2102.09690.pdf",
        "category": "finetuning",
        "arxiv_id": "2102.09690"
    },
    
    # Evaluation & Benchmarks
    {
        "title": "Measuring Massive Multitask Language Understanding",
        "authors": "Hendrycks et al.",
        "year": "2020",
        "url": "https://arxiv.org/pdf/2009.03300.pdf",
        "category": "evaluation",
        "arxiv_id": "2009.03300"
    },
    {
        "title": "BIG-bench: Beyond the Imitation Game Benchmark",
        "authors": "Srivastava et al.",
        "year": "2022",
        "url": "https://arxiv.org/pdf/2206.04615.pdf",
        "category": "evaluation",
        "arxiv_id": "2206.04615"
    },
    {
        "title": "HellaSwag: Can a Machine Really Finish Your Sentence?",
        "authors": "Zellers et al.",
        "year": "2019",
        "url": "https://arxiv.org/pdf/1905.07830.pdf",
        "category": "evaluation",
        "arxiv_id": "1905.07830"
    },
    {
        "title": "SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding",
        "authors": "Wang et al.",
        "year": "2019",
        "url": "https://arxiv.org/pdf/1905.00537.pdf",
        "category": "evaluation",
        "arxiv_id": "1905.00537"
    },
    {
        "title": "Beyond Accuracy: Behavioral Testing of NLP Models",
        "authors": "Ribeiro et al.",
        "year": "2020",
        "url": "https://arxiv.org/pdf/2005.04118.pdf",
        "category": "evaluation",
        "arxiv_id": "2005.04118"
    }
]

print(f"📚 Loaded {len(PAPERS)} papers across {len(set(p['category'] for p in PAPERS))} categories")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 Helper Functions

# COMMAND ----------

def sanitize_filename(filename):
    """Remove invalid characters from filename for UC volume compatibility"""
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Additional cleanup for UC volumes
    filename = re.sub(r'[^\w\s\-_\.]', '', filename)
    filename = re.sub(r'\s+', '_', filename)
    
    return filename.strip()


def get_file_size_mb(filepath):
    """Get file size in MB for UC volume"""
    try:
        file_info = dbutils.fs.ls(filepath)[0]
        return file_info.size / (1024 * 1024)
    except:
        return 0


def file_exists_uc(filepath):
    """Check if file exists in UC volume"""
    try:
        dbutils.fs.ls(filepath)
        return True
    except:
        return False


def create_uc_directory(dir_path):
    """Create directory in UC volume if it doesn't exist"""
    try:
        dbutils.fs.mkdirs(dir_path)
        return True
    except Exception as e:
        print(f"❌ Error creating directory {dir_path}: {e}")
        return False

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📥 Download Functions

# COMMAND ----------

def download_paper_to_uc_volume(paper, output_dir, max_retries=3):
    """Download a single paper to UC volume - always overwrite existing files"""

    tmp_path = '/tmp/test'
    
    # Create filename with arXiv ID for better organization  
    safe_title = sanitize_filename(paper['title'])
    filename = f"{paper['year']}_{paper['authors'].split(' et al.')[0]}_{paper['arxiv_id']}_{safe_title}.pdf"
    
    # UC volume path
    uc_path = f"{output_dir}/{filename}"
    
    # Check if file exists (for logging only)
    if file_exists_uc(uc_path):
        existing_size = get_file_size_mb(uc_path)
        print(f"🔄 Overwriting existing: {filename} ({existing_size:.1f} MB)")
    
    # Ensure output directory exists
    create_uc_directory(output_dir)
    
    # Download with retries
    for attempt in range(max_retries):
        try:
            print(f"⬇️  Downloading ({attempt+1}/{max_retries}): {filename}")
            
            # Download content with proper headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(paper['url'], timeout=60, headers=headers)
            response.raise_for_status()
            
            # Verify we got PDF content
            if not response.content.startswith(b'%PDF'):
                print(f"❌ Response is not a PDF file for {filename}")
                continue
                
            # Write using local file system first, then copy to UC volume
            # Use /local_disk0/ for Databricks local storage
            local_path = f"{tmp_path}/{filename}"
            
            # Create directory if it doesn't exist
            import os
            os.makedirs(tmp_path, exist_ok=True)
            
            # Write binary content to local temp file
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            # Copy from local to UC volume (will overwrite if exists)
            dbutils.fs.cp(f"file://{local_path}", uc_path)
            
            # Clean up local temp file
            os.remove(local_path)
            
            # Verify the file
            size = get_file_size_mb(uc_path)
            print(f"✅ Downloaded: {filename} ({size:.1f} MB)")
            return True, filename, size
            
        except Exception as e:
            print(f"❌ Error downloading {filename} (attempt {attempt+1}): {e}")
            
            # Clean up any partial files
            try:
                import os
                if os.path.exists(f"{tmp_path}/{filename}"):
                    os.remove(f"{tmp_path}/{filename}")
            except:
                pass
                
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    print(f"💥 Failed to download after {max_retries} attempts: {filename}")
    return False, filename, 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Analysis and Visualization Functions

# COMMAND ----------

def analyze_collection(papers_list=None):
    """Analyze the papers collection"""
    if papers_list is None:
        papers_list = PAPERS
        
    categories = Counter(paper['category'] for paper in papers_list)
    years = Counter(paper['year'] for paper in papers_list)
    
    print("📊 COLLECTION ANALYSIS")
    print("=" * 50)
    
    print(f"\n📚 Total Papers: {len(papers_list)}")
    
    print(f"\n📁 Papers by Category:")
    for category, count in sorted(categories.items()):
        print(f"  • {category.title()}: {count} papers")
    
    print(f"\n📅 Papers by Year:")
    for year, count in sorted(years.items()):
        print(f"  • {year}: {count} papers")
    
    return categories, years


def list_papers_by_category(papers_list=None):
    """Display all papers organized by category"""
    if papers_list is None:
        papers_list = PAPERS
        
    categories = defaultdict(list)
    for paper in papers_list:
        categories[paper['category']].append(paper)
    
    print("📚 LLM PAPERS COLLECTION")
    print("=" * 60)
    
    for category, papers in sorted(categories.items()):
        print(f"\n📁 {category.upper()} ({len(papers)} papers)")
        print("-" * 40)
        for paper in papers:
            print(f"  • {paper['title']}")
            print(f"    {paper['authors']} ({paper['year']}) - arXiv:{paper['arxiv_id']}")
            print()


def filter_papers_by_categories(categories_list):
    """Filter papers by specified categories"""
    if "all" in categories_list:
        return PAPERS
    
    filtered_papers = [p for p in PAPERS if p.get('category') in categories_list]
    return filtered_papers

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Main Download Functions

# COMMAND ----------

def download_all_papers_to_uc():
    """Download papers based on configuration parameters"""
    
    if not volume_ready:
        print("❌ UC Volume not ready. Please fix volume setup first.")
        return None, None, None, 0
    
    print("🚀 Starting LLM Papers Download to UC Volume")
    print("=" * 50)
    print(f"📁 Target Location: {UC_VOLUME_PATH}")
    print(f"🎯 Categories: {CATEGORIES_TO_DOWNLOAD}")
    print(f"🗂️  Organize by Category: {ORGANIZE_BY_CATEGORY}")
    
    # Filter papers by selected categories
    papers_to_download = filter_papers_by_categories(CATEGORIES_TO_DOWNLOAD)
    
    if not papers_to_download:
        print("❌ No papers found for selected categories")
        return None, None, None, 0
    
    print(f"📝 Papers to download: {len(papers_to_download)}")
    
    success_count = 0
    failed_papers = []
    total_size_mb = 0
    download_log = []
    
    if ORGANIZE_BY_CATEGORY:
        # Group by category
        categories = defaultdict(list)
        for paper in papers_to_download:
            categories[paper['category']].append(paper)
        
        # Download by category
        for category, papers in sorted(categories.items()):
            category_dir = f"{UC_VOLUME_PATH}/{category}"
            
            print(f"\n📁 Downloading {category.upper()} papers ({len(papers)} papers)")
            print("=" * 60)
            
            for i, paper in enumerate(papers, 1):
                print(f"\n[{i}/{len(papers)}] ", end="")
                
                success, filename, size = download_paper_to_uc_volume(paper, category_dir)
                
                download_log.append({
                    'paper': paper,
                    'success': success,
                    'filename': filename,
                    'size_mb': size,
                    'category': category,
                    'uc_path': f"{category_dir}/{filename}" if success else None
                })
                
                if success:
                    success_count += 1
                    total_size_mb += size
                else:
                    failed_papers.append(paper)
                
                # Rate limiting
                time.sleep(0.5)
    
    else:
        # Download all to single directory
        print(f"📁 Downloading all papers to {UC_VOLUME_PATH}")
        
        for i, paper in enumerate(papers_to_download, 1):
            print(f"\n[{i}/{len(papers_to_download)}] ", end="")
            
            success, filename, size = download_paper_to_uc_volume(paper, UC_VOLUME_PATH)
            
            download_log.append({
                'paper': paper,
                'success': success,
                'filename': filename,
                'size_mb': size,
                'category': paper['category'],
                'uc_path': f"{UC_VOLUME_PATH}/{filename}" if success else None
            })
            
            if success:
                success_count += 1
                total_size_mb += size
            else:
                failed_papers.append(paper)
            
            time.sleep(0.5)
    
    # Print summary
    print(f"\n\n🎉 DOWNLOAD COMPLETE!")
    print("=" * 50)
    print(f"✅ Successfully downloaded: {success_count}/{len(papers_to_download)} papers")
    print(f"📊 Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.1f} GB)")
    print(f"📁 Saved to UC Volume: {UC_VOLUME_PATH}")
    
    if failed_papers:
        print(f"\n❌ Failed downloads: {len(failed_papers)}")
        print("\nFailed papers:")
        for paper in failed_papers:
            print(f"  - {paper['title']} ({paper['authors']}, {paper['year']})")
    
    # Create download metadata
    create_download_metadata(download_log, success_count, total_size_mb)
    
    return success_count, failed_papers, download_log, total_size_mb


def create_download_metadata(download_log, success_count, total_size_mb):
    """Create metadata file with download information - no local filesystem"""
    import json
    from datetime import datetime
    
    metadata = {
        "download_timestamp": datetime.now().isoformat(),
        "uc_volume_path": UC_VOLUME_PATH,
        "catalog": CATALOG_NAME,
        "schema": SCHEMA_NAME, 
        "volume": VOLUME_NAME,
        "total_papers_attempted": len(download_log),
        "successful_downloads": success_count,
        "total_size_mb": total_size_mb,
        "organized_by_category": ORGANIZE_BY_CATEGORY,
        "categories_downloaded": CATEGORIES_TO_DOWNLOAD,
        "papers": download_log
    }
    
    # Save metadata to UC volume
    metadata_path = f"{UC_VOLUME_PATH}/download_metadata.json"
    
    # Convert to JSON string
    metadata_json = json.dumps(metadata, indent=2)
    
    # Write directly to UC volume
    dbutils.fs.put(metadata_path, metadata_json, overwrite=True)
    
    print(f"📝 Metadata saved to: {metadata_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 Browse and Manage Downloads

# COMMAND ----------

def browse_uc_volume():
    """Browse and analyze downloaded papers in UC volume"""
    
    try:
        # List all directories and files
        items = dbutils.fs.ls(UC_VOLUME_PATH)
        
        print(f"📁 Contents of UC Volume: {UC_VOLUME_PATH}")
        print("=" * 80)
        
        total_size = 0
        total_files = 0
        
        for item in items:
            if item.isDir():
                # Category directory
                category_name = item.name.rstrip('/')
                print(f"\n📁 {category_name.upper()}")
                try:
                    files = dbutils.fs.ls(item.path)
                    pdf_files = [f for f in files if f.name.endswith('.pdf')]
                    
                    category_size = sum(f.size for f in pdf_files)
                    category_count = len(pdf_files)
                    
                    print(f"   📊 {category_count} papers, {category_size/(1024*1024):.1f} MB")
                    
                    total_size += category_size
                    total_files += category_count
                    
                    # Show first few files
                    for i, f in enumerate(pdf_files[:3]):
                        print(f"   • {f.name}")
                    
                    if len(pdf_files) > 3:
                        print(f"   ... and {len(pdf_files)-3} more")
                        
                except Exception as e:
                    print(f"   ❌ Error reading directory: {e}")
            
            elif item.name.endswith('.pdf'):
                # PDF file in root
                total_size += item.size
                total_files += 1
                print(f"📄 {item.name} ({item.size/(1024*1024):.1f} MB)")
            
            elif item.name.endswith('.json'):
                # Metadata file
                print(f"📋 {item.name} (metadata)")
        
        print(f"\n📊 SUMMARY")
        print("=" * 30)
        print(f"Total PDF files: {total_files}")
        print(f"Total size: {total_size/(1024*1024):.1f} MB ({total_size/(1024*1024*1024):.2f} GB)")
        print(f"UC Volume: {CATALOG_NAME}.{SCHEMA_NAME}.{VOLUME_NAME}")
        
    except Exception as e:
        print(f"❌ Error browsing UC volume {UC_VOLUME_PATH}: {e}")
        print("Volume may not exist yet. Run download first.")


def get_volume_metadata():
    """Read and display download metadata - no local filesystem"""
    metadata_path = f"{UC_VOLUME_PATH}/download_metadata.json"
    
    try:
        # Read directly from UC volume using dbutils.fs.head
        # For small JSON files, head should get the entire content
        metadata_content = dbutils.fs.head(metadata_path, max_bytes=10000000)  # 10MB max
        
        import json
        metadata = json.loads(metadata_content)
        
        print("📋 DOWNLOAD METADATA")
        print("=" * 40)
        print(f"Download Date: {metadata['download_timestamp']}")
        print(f"UC Volume: {metadata['catalog']}.{metadata['schema']}.{metadata['volume']}")
        print(f"Total Papers: {metadata['successful_downloads']}/{metadata['total_papers_attempted']}")
        print(f"Total Size: {metadata['total_size_mb']:.1f} MB")
        print(f"Categories: {metadata['categories_downloaded']}")
        print(f"Organized by Category: {metadata['organized_by_category']}")
        
        return metadata
        
    except Exception as e:
        print(f"❌ Could not read metadata: {e}")
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Quick Start Examples

# COMMAND ----------

# 📊 Analyze the collection first
papers_to_download = filter_papers_by_categories(CATEGORIES_TO_DOWNLOAD)
print(f"🎯 Selected {len(papers_to_download)} papers based on configuration")
analyze_collection(papers_to_download)

# COMMAND ----------

# 📋 List sample papers from selected categories
print("🔍 Sample of papers to be downloaded:")
print("=" * 40)

categories = defaultdict(list)
for paper in papers_to_download:
    categories[paper['category']].append(paper)

# Show first 2 papers from each selected category
for category, papers in sorted(list(categories.items())[:5]):  # First 5 categories
    print(f"\n📁 {category.upper()}:")
    for paper in papers[:2]:  # First 2 papers
        print(f"  • {paper['title']} ({paper['year']}) - arXiv:{paper['arxiv_id']}")
    if len(papers) > 2:
        print(f"  ... and {len(papers)-2} more")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚀 Execute Download
# MAGIC
# MAGIC Run this cell to start the download based on your configuration:

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC mkdir /tmp/test

# COMMAND ----------

# Execute the download based on widget configuration
print("🚀 Starting download with current configuration...")
print(f"📁 UC Volume: {UC_VOLUME_PATH}")
print(f"🎯 Categories: {CATEGORIES_TO_DOWNLOAD}")

success_count, failed_papers, download_log, total_size_mb = download_all_papers_to_uc()

if success_count is not None:
    print(f"\n🎉 Download completed! {success_count} papers downloaded ({total_size_mb:.1f} MB)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📁 Browse Results

# COMMAND ----------

# Browse what's been downloaded
browse_uc_volume()

# COMMAND ----------

# View download metadata
metadata = get_volume_metadata()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 Utility Commands

# COMMAND ----------

# Query UC volume information
try:
    volume_info = spark.sql(f"DESCRIBE VOLUME {CATALOG_NAME}.{SCHEMA_NAME}.{VOLUME_NAME}").collect()
    print("📁 UC VOLUME INFORMATION")
    print("=" * 40)
    for row in volume_info:
        print(f"{row.info_name}: {row.info_value}")
except Exception as e:
    print(f"❌ Error getting volume info: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Create Papers Index DataFrame

# COMMAND ----------

def create_papers_index_dataframe():
    """Create a Spark DataFrame with papers information for analysis"""
    
    # Convert papers list to DataFrame
    papers_df = spark.createDataFrame(papers_to_download)
    
    # Register as temporary view for SQL queries
    papers_df.createOrReplaceTempView("llm_papers")
    
    print("📊 Created papers index DataFrame and registered as 'llm_papers' view")
    print(f"📝 Total records: {papers_df.count()}")
    
    # Show schema
    papers_df.printSchema()
    
    # Show sample data
    print("\n📋 Sample papers:")
    papers_df.select("title", "authors", "year", "category", "arxiv_id").show(5, truncate=False)
    
    return papers_df

# Create the DataFrame
papers_df = create_papers_index_dataframe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 SQL Analysis Examples

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Papers by category
# MAGIC SELECT category, COUNT(*) as paper_count 
# MAGIC FROM llm_papers 
# MAGIC GROUP BY category 
# MAGIC ORDER BY paper_count DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Papers by year
# MAGIC SELECT year, COUNT(*) as paper_count 
# MAGIC FROM llm_papers 
# MAGIC GROUP BY year 
# MAGIC ORDER BY year

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Recent foundational papers
# MAGIC SELECT title, authors, year, arxiv_id
# MAGIC FROM llm_papers 
# MAGIC WHERE category = 'foundational' 
# MAGIC ORDER BY year DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 Notes & Best Practices
# MAGIC
# MAGIC ### 🎛️ Configuration:
# MAGIC - **UC Catalog**: Configure your organization's catalog name
# MAGIC - **Schema**: Use appropriate schema (e.g., `research`, `ml`, `default`)
# MAGIC - **Volume**: Descriptive name like `llm_papers` or `research_papers`
# MAGIC - **Categories**: Start with `foundational` then expand to other categories
# MAGIC
# MAGIC ### 📁 File Organization:
# MAGIC - Files saved as: `{year}_{author}_{arxiv_id}_{title}.pdf`
# MAGIC - Organized by category folders when enabled
# MAGIC - Metadata saved as JSON for tracking
# MAGIC
# MAGIC ### 🔐 Unity Catalog Benefits:
# MAGIC - **Governance**: Fine-grained access control via UC permissions
# MAGIC - **Lineage**: Track usage across notebooks and workflows
# MAGIC - **Security**: Enterprise-grade security and audit trails
# MAGIC - **Cross-workspace**: Access from any workspace in your metastore
# MAGIC
# MAGIC ### 🚀 Performance Tips:
# MAGIC 1. Start with smaller categories (`foundational`, `survey`)
# MAGIC 2. Downloads are resumable (skips existing files)
# MAGIC 3. Use appropriate cluster size for faster downloads
# MAGIC 4. Monitor volume storage limits
# MAGIC
# MAGIC ### 📊 Analysis Ready:
# MAGIC - Papers indexed in Spark DataFrame
# MAGIC - SQL queries available via `llm_papers` view
# MAGIC - Metadata tracking for governance
# MAGIC - Ready for ML pipelines and analysis workflows