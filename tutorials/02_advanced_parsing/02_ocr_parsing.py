# Databricks notebook source
# MAGIC %md
# MAGIC # Document OCR and Parsing with Docling
# MAGIC
# MAGIC ## Overview
# MAGIC This notebook demonstrates how to use **Docling** for OCR and document parsing. Docling is a modern document processing library that can:
# MAGIC - Extract text from PDFs with OCR capabilities
# MAGIC - Handle complex document layouts
# MAGIC - Export to markdown format
# MAGIC - Process both text-based and scanned documents
# MAGIC
# MAGIC ## Benefits of Docling
# MAGIC - 🔍 **Smart OCR**: Automatically detects when OCR is needed
# MAGIC - 📝 **Markdown Export**: Clean, structured markdown output
# MAGIC - 📊 **Table Detection**: Preserves table structure
# MAGIC - 🚀 **Fast Processing**: Optimized for batch operations
# MAGIC - 🎯 **Flexible Options**: Configurable pipeline for different use cases

# COMMAND ----------

# Install docling and dependencies
%pip install -U docling python-dotenv
%restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# Load environment variables from .env (if present) before reading them via os.getenv
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # returns True if a .env is found and parsed

import os
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Environment configuration - pull defaults from environment variables
CATALOG_NAME_ENV = os.getenv("CATALOG_NAME")
SCHEMA_NAME_ENV  = os.getenv("SCHEMA_NAME", "tutorials")
VOLUME_NAME_ENV  = os.getenv("VOLUME_NAME", "sample_docs")
# Create Databricks widgets for runtime configuration
dbutils.widgets.text("catalog_name", CATALOG_NAME_ENV or "", "Catalog Name")
dbutils.widgets.text("schema_name",  SCHEMA_NAME_ENV,  "Schema Name") 
dbutils.widgets.text("volume_name",  VOLUME_NAME_ENV,  "Volume Name")

# Get final configuration values
CATALOG_NAME = dbutils.widgets.get("catalog_name") or CATALOG_NAME_ENV
SCHEMA_NAME  = dbutils.widgets.get("schema_name")  or SCHEMA_NAME_ENV
VOLUME_NAME  = dbutils.widgets.get("volume_name")  or VOLUME_NAME_ENV

# Construct volume path
VOLUME_PATH = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/{VOLUME_NAME}"

print(f"🎯 OCR Processing Setup:")
print(f"   Catalog: {CATALOG_NAME}")
print(f"   Schema: {SCHEMA_NAME}")
print(f"   Volume: {VOLUME_NAME}")
print(f"   Volume Path: {VOLUME_PATH}")

# COMMAND ----------

# DBTITLE 1,Check Available Sample Documents
# List available PDF files in the volume
try:
    files = dbutils.fs.ls(f"dbfs:{VOLUME_PATH}")
    pdf_files = [f for f in files if f.name.lower().endswith('.pdf')]
    
    print(f"📁 Found {len(pdf_files)} PDF files in volume:")
    for file in pdf_files:
        size_mb = file.size / (1024 * 1024)
        print(f"   📄 {file.name} ({size_mb:.2f} MB)")
        
    if len(pdf_files) == 0:
        print("⚠️ No PDF files found. Please ensure sample documents are uploaded to the volume.")
    else:
        # Store the first few files for processing
        sample_docs = [f"{VOLUME_PATH}/{f.name}" for f in pdf_files[:3]]
        print(f"\n🎯 Will process these sample documents:")
        for i, doc in enumerate(sample_docs):
            print(f"   {i+1}. {os.path.basename(doc)}")
            
except Exception as e:
    print(f"❌ Error accessing volume: {e}")
    print("Please check your catalog, schema, and volume configuration.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize Docling Converters

# COMMAND ----------

# DBTITLE 1,Setup Docling Converters
# Create basic converter with default settings
basic_converter = DocumentConverter()

# Create advanced converter with specific pipeline options
pipeline_options = PdfPipelineOptions(
    do_ocr=True,  # Enable OCR for scanned documents
    do_table_structure=True  # Enable table detection and structure preservation
)

advanced_converter = DocumentConverter(
    format_options={
        "pdf": PdfFormatOption(pipeline_options=pipeline_options)
    }
)

print("📖 Docling converters initialized:")
print("   ✅ Basic converter (default settings)")
print("   ✅ Advanced converter (OCR + table structure enabled)")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Document Processing Examples
# MAGIC
# MAGIC Let's demonstrate Docling's OCR capabilities with different configuration options.

# COMMAND ----------

# DBTITLE 1,Basic Document Processing
# Process the first sample document with basic converter
if 'sample_docs' in locals() and len(sample_docs) > 0:
    test_document = sample_docs[0]
    print(f"📄 Processing: {os.path.basename(test_document)}")
    
    print("\n📖 Basic Processing (default settings):")
    print("-"*50)
    try:
        # Convert with basic settings
        result_basic = basic_converter.convert(test_document)
        markdown_basic = result_basic.document.export_to_markdown()
        
        print(f"✅ Successfully extracted {len(markdown_basic)} characters")
        print("\n📝 First 500 characters of extracted text:")
        print("="*50)
        print(markdown_basic[:500])
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error during basic processing: {e}")
        markdown_basic = ""

# COMMAND ----------

# DBTITLE 1,Advanced OCR Processing
# Process the same document with advanced OCR settings
if 'sample_docs' in locals() and len(sample_docs) > 0:
    print(f"\n🔤 Advanced OCR Processing:")
    print("-"*50)
    try:
        # Convert with OCR and table structure detection
        result_advanced = advanced_converter.convert(test_document)
        markdown_advanced = result_advanced.document.export_to_markdown()
        
        print(f"✅ Successfully extracted {len(markdown_advanced)} characters")
        print("\n📝 First 500 characters of OCR text:")
        print("="*50)
        print(markdown_advanced[:500])
        print("="*50)
        
        # Store for comparison
        extracted_text = markdown_advanced
        
    except Exception as e:
        print(f"❌ Error during OCR processing: {e}")
        extracted_text = ""
else:
    print("⚠️ No sample documents available for processing")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Batch OCR Processing
# MAGIC
# MAGIC Let's process multiple sample documents to demonstrate batch OCR capabilities.

# COMMAND ----------

# DBTITLE 1,Process Multiple Documents with OCR
# Process all available sample documents
if 'sample_docs' in locals() and len(sample_docs) > 1:
    print(f"📚 Processing {len(sample_docs)} sample documents with OCR...")
    
    processed_docs = []
    
    for i, doc_path in enumerate(sample_docs):
        doc_name = os.path.basename(doc_path)
        print(f"\n📄 Processing document {i+1}/{len(sample_docs)}: {doc_name}")
        
        try:
            # Use advanced converter for all documents
            result = advanced_converter.convert(doc_path)
            markdown_content = result.document.export_to_markdown()
            
            # Store document info
            doc_info = {
                'name': doc_name,
                'path': doc_path,
                'content': markdown_content,
                'length': len(markdown_content),
                'status': 'success'
            }
            
            processed_docs.append(doc_info)
            print(f"   ✅ Successfully processed: {len(markdown_content)} characters")
            
        except Exception as e:
            print(f"   ❌ Failed to process {doc_name}: {e}")
            processed_docs.append({
                'name': doc_name,
                'path': doc_path,
                'content': '',
                'length': 0,
                'status': f'failed: {str(e)}'
            })
    
    # Summary of processed documents
    print(f"\n📊 OCR Processing Summary:")
    print("="*60)
    successful = sum(1 for doc in processed_docs if doc['status'] == 'success')
    print(f"✅ Successfully processed: {successful}/{len(processed_docs)} documents")
    
    for doc in processed_docs:
        status_emoji = "✅" if doc['status'] == 'success' else "❌"
        print(f"   {status_emoji} {doc['name']}: {doc['length']} chars ({doc['status']})")

else:
    print("⚠️ Need at least 2 sample documents for batch processing demo")

# COMMAND ----------

# DBTITLE 1,Document Content Analysis
# Analyze the extracted content from different documents
if 'processed_docs' in locals():
    for doc in processed_docs:
        if doc['status'] == 'success' and doc['content']:
            print(f"\n📄 Content Analysis: {doc['name']}")
            print("="*50)
            
            # Get first 800 characters for analysis
            content_sample = doc['content'][:800]
            
            # Basic content analysis
            word_count = len(content_sample.split())
            line_count = len(content_sample.split('\n'))
            
            print(f"📊 Content Statistics:")
            print(f"   Total characters: {doc['length']}")
            print(f"   Words in sample: {word_count}")
            print(f"   Lines in sample: {line_count}")
            
            print(f"\n📝 Content Preview (first 400 chars):")
            print("-"*40)
            print(content_sample[:400] + "...")
            print("-"*40)
            
            # Identify document patterns
            print(f"\n🔍 Document Pattern Analysis:")
            if any(word in content_sample.lower() for word in ['annual', 'report', 'financial', 'revenue']):
                print("   📈 Appears to be a financial/annual report")
            elif any(word in content_sample.lower() for word in ['insurance', 'policy', 'coverage', 'claim']):
                print("   🛡️ Appears to be an insurance document")
            elif any(word in content_sample.lower() for word in ['legal', 'law', 'regulation', 'compliance']):
                print("   ⚖️ Appears to be a legal document")
            else:
                print("   📄 General business document")
                
            # Check for structured content
            if '|' in content_sample and '-' in content_sample:
                print("   📊 Contains table-like structures")
            if content_sample.count('#') > 2:
                print("   📝 Contains markdown headers/structure")
            
            print("="*50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. OCR Configuration Comparison
# MAGIC
# MAGIC Let's compare different Docling processing configurations to understand their impact.

# COMMAND ----------

# DBTITLE 1,Compare Different OCR Configurations
# Compare different Docling configurations on the same document
if 'sample_docs' in locals() and len(sample_docs) > 0:
    test_doc = sample_docs[0]
    print(f"🔬 Comparing OCR configurations on: {os.path.basename(test_doc)}")
    
    configurations = []
    
    print("\n1️⃣ Basic Text Extraction (no OCR, no table structure):")
    print("-"*50)
    try:
        # Create a converter with minimal processing
        basic_pipeline = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
        minimal_converter = DocumentConverter(
            format_options={"pdf": PdfFormatOption(pipeline_options=basic_pipeline)}
        )
        
        result_minimal = minimal_converter.convert(test_doc)
        minimal_content = result_minimal.document.export_to_markdown()
        
        configurations.append({
            'name': 'Basic Text Extraction',
            'length': len(minimal_content),
            'content': minimal_content[:300],
            'status': 'success',
            'settings': 'OCR: False, Tables: False'
        })
        
        print(f"✅ Characters extracted: {len(minimal_content)}")
        print(f"📝 Preview: {minimal_content[:200]}...")
        
    except Exception as e:
        print(f"❌ Basic processing failed: {e}")
        configurations.append({
            'name': 'Basic Text Extraction',
            'status': 'failed',
            'error': str(e),
            'settings': 'OCR: False, Tables: False'
        })
    
    print("\n2️⃣ OCR Only (no table structure):")
    print("-"*50)
    try:
        # OCR enabled but no table structure
        ocr_only_pipeline = PdfPipelineOptions(do_ocr=True, do_table_structure=False)
        ocr_only_converter = DocumentConverter(
            format_options={"pdf": PdfFormatOption(pipeline_options=ocr_only_pipeline)}
        )
        
        result_ocr_only = ocr_only_converter.convert(test_doc)
        ocr_only_content = result_ocr_only.document.export_to_markdown()
        
        configurations.append({
            'name': 'OCR Only',
            'length': len(ocr_only_content),
            'content': ocr_only_content[:300],
            'status': 'success',
            'settings': 'OCR: True, Tables: False'
        })
        
        print(f"✅ Characters extracted: {len(ocr_only_content)}")
        print(f"📝 Preview: {ocr_only_content[:200]}...")
        
    except Exception as e:
        print(f"❌ OCR-only processing failed: {e}")
        configurations.append({
            'name': 'OCR Only',
            'status': 'failed',
            'error': str(e),
            'settings': 'OCR: True, Tables: False'
        })
    
    print("\n3️⃣ Full OCR + Table Structure:")
    print("-"*50)
    try:
        result_full = advanced_converter.convert(test_doc)
        full_content = result_full.document.export_to_markdown()
        
        configurations.append({
            'name': 'Full OCR + Tables',
            'length': len(full_content),
            'content': full_content[:300],
            'status': 'success',
            'settings': 'OCR: True, Tables: True'
        })
        
        print(f"✅ Characters extracted: {len(full_content)}")
        print(f"📝 Preview: {full_content[:200]}...")
        
    except Exception as e:
        print(f"❌ Full OCR processing failed: {e}")
        configurations.append({
            'name': 'Full OCR + Tables',
            'status': 'failed',
            'error': str(e),
            'settings': 'OCR: True, Tables: True'
        })
    
    # Configuration comparison summary
    print("\n📊 CONFIGURATION COMPARISON SUMMARY:")
    print("="*60)
    
    successful_configs = [c for c in configurations if c['status'] == 'success']
    
    if len(successful_configs) > 1:
        # Compare extraction lengths
        lengths = [(c['name'], c['length']) for c in successful_configs]
        lengths.sort(key=lambda x: x[1])
        
        print("📈 Content extraction comparison (characters):")
        for name, length in lengths:
            print(f"   {name}: {length:,} characters")
        
        if len(lengths) >= 2:
            improvement = lengths[-1][1] - lengths[0][1]
            base_length = lengths[0][1]
            improvement_pct = (improvement / base_length * 100) if base_length > 0 else 0
            print(f"\n🚀 Best vs baseline improvement: +{improvement:,} characters ({improvement_pct:.1f}% increase)")
    
    print(f"\n📋 Configuration Results:")
    for config in configurations:
        status_emoji = "✅" if config['status'] == 'success' else "❌"
        print(f"{status_emoji} {config['name']} ({config['settings']}): {config['status']}")
        if config['status'] == 'success':
            print(f"   → Extracted {config['length']:,} characters")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrated comprehensive OCR document processing using Docling:
# MAGIC
# MAGIC ### 🔧 **Technical Setup**
# MAGIC - Environment configuration with widgets and .env support
# MAGIC - Docling integration with configurable processing pipelines
# MAGIC - Sample document management from Unity Catalog volumes
# MAGIC - Error handling and status tracking throughout processing
# MAGIC
# MAGIC ### 📖 **OCR Processing Approaches**
# MAGIC - **Basic Text Extraction**: Fast processing for text-based PDFs
# MAGIC - **OCR-Only Processing**: Handles scanned documents with optical character recognition
# MAGIC - **Full OCR + Table Structure**: Advanced processing that preserves table layouts and structure
# MAGIC - **Batch Processing**: Scalable processing across multiple document types
# MAGIC
# MAGIC ### 🎯 **Key Capabilities Demonstrated**
# MAGIC - Docling `DocumentConverter` with configurable `PdfPipelineOptions`
# MAGIC - Comparison of different OCR configuration impacts
# MAGIC - Batch processing with comprehensive status tracking
# MAGIC - Content analysis and pattern recognition in extracted text
# MAGIC - Performance metrics and extraction quality assessment
# MAGIC
# MAGIC ### 📊 **Configuration Options Explored**
# MAGIC - **`do_ocr=False`**: Basic text extraction from text-based PDFs
# MAGIC - **`do_ocr=True`**: OCR processing for scanned or image-based content
# MAGIC - **`do_table_structure=True`**: Preserves table formatting in markdown output
# MAGIC - **Combined Settings**: Optimal configuration for comprehensive text extraction
# MAGIC
# MAGIC ### 💡 **When to Use Each Configuration**
# MAGIC - **Basic Text Extraction**: Clean, text-based PDFs where OCR is unnecessary
# MAGIC - **OCR Processing**: Scanned documents, images, or PDFs with embedded images
# MAGIC - **Table Structure Detection**: Documents with important tabular data to preserve
# MAGIC - **Full Configuration**: Production environments requiring maximum content fidelity
# MAGIC
# MAGIC ### 🚀 **Next Steps**
# MAGIC - Scale to distributed processing with Spark for large document batches
# MAGIC - Implement content classification based on extracted patterns
# MAGIC - Add quality scoring and confidence metrics for OCR results
# MAGIC - Integrate with downstream analysis workflows for structured data extraction
