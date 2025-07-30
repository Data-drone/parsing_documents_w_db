# Databricks notebook source
# MAGIC %md
# MAGIC # Vision Model Document Parsing
# MAGIC
# MAGIC ## Overview
# MAGIC This tutorial explores using **Vision Language Models (VLMs)** to parse PDF documents directly from images. Unlike traditional text extraction (covered in tutorial 01), vision models can process document images directly, understanding layout, visual structure, and extracting information from complex formats like tables, forms, and multi-column layouts.
# MAGIC
# MAGIC ## Why Vision Models for Documents?
# MAGIC - 🖼️ **Direct Image Processing**: No need for text extraction - works with scanned documents
# MAGIC - 📊 **Complex Layout Understanding**: Handles tables, forms, multi-column layouts
# MAGIC - 🎯 **Visual Context**: Understands positioning, formatting, and visual relationships
# MAGIC - 📋 **Form & Table Extraction**: Excels at structured data in visual formats
# MAGIC - 🔍 **OCR-Free**: Processes even poor quality scans without separate OCR steps
# MAGIC
# MAGIC ## What You'll Learn
# MAGIC 1. Setting up vision-capable models
# MAGIC 2. Processing document page images
# MAGIC 3. Extracting structured data from visual layouts
# MAGIC 4. Handling different document types (reports, forms, tables)
# MAGIC 5. Quality assessment for vision-based extraction

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# Install required packages for vision processing
%pip install -U pillow databricks_langchain python-dotenv
%restart_python

# COMMAND ----------

# Load environment variables and setup
from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv())

# Environment configuration
CATALOG_NAME_ENV = os.getenv("CATALOG_NAME")
SCHEMA_NAME_ENV  = os.getenv("SCHEMA_NAME", "knowledge_extraction")
VOLUME_NAME_ENV  = os.getenv("VOLUME_NAME", "raw_data")
LLM_ENDPOINT_ENV = os.getenv("LLM_MODEL", "databricks-meta-llama-3-3-70b-instruct")

# Databricks widgets for runtime configuration
dbutils.widgets.text("catalog_name", CATALOG_NAME_ENV or "", "Catalog Name")
dbutils.widgets.text("schema_name",  SCHEMA_NAME_ENV,  "Schema Name")
dbutils.widgets.text("volume_name",  VOLUME_NAME_ENV,  "Volume Name")
dbutils.widgets.text("llm_endpoint", LLM_ENDPOINT_ENV, "LLM Endpoint")

# Get final configuration values
CATALOG_NAME = dbutils.widgets.get("catalog_name") or CATALOG_NAME_ENV
SCHEMA_NAME  = dbutils.widgets.get("schema_name")  or SCHEMA_NAME_ENV
VOLUME_NAME  = dbutils.widgets.get("volume_name")  or VOLUME_NAME_ENV
LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint") or LLM_ENDPOINT_ENV

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Exploration
# MAGIC 
# MAGIC Let's examine the two sample page images available in our volume for vision processing.

# COMMAND ----------

# DBTITLE 1,Setup Workspace Paths
current_user = spark.sql("SELECT current_user()").first()[0]
username = current_user.split('@')[0].replace('.', '_')

input_schema = SCHEMA_NAME
volume = VOLUME_NAME
full_vol_path = f'/Volumes/{CATALOG_NAME}/{input_schema}/{volume}'

print(f"🎯 Vision Processing Setup:")
print(f"   Catalog: {CATALOG_NAME}")
print(f"   Schema: {input_schema}")
print(f"   Volume: {volume}")
print(f"   Path: {full_vol_path}")

# COMMAND ----------

# DBTITLE 1,Verify Sample Images
# Check that our expected sample images are present
expected_images = ['report_chart_page.png', 'report_table_page.png']
page_images_path = f"{full_vol_path}/page_images"

print(f"📁 Checking for sample images in: {page_images_path}")

try:
    files = dbutils.fs.ls(f"dbfs:{page_images_path}")
    image_files = [f for f in files if f.name.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"🖼️ Found {len(image_files)} image files:")
    for file in image_files:
        size_mb = file.size / (1024 * 1024)
        status = "✅" if file.name in expected_images else "📄"
        print(f"   {status} {file.name} ({size_mb:.2f} MB)")
    
    # Check if we have our expected sample images
    found_images = [f.name for f in image_files]
    missing_images = [img for img in expected_images if img not in found_images]
    
    if missing_images:
        print(f"\n⚠️  Missing expected images: {missing_images}")
        print("💡 Run the environment setup notebook to copy sample images")
    else:
        print(f"\n✅ All expected sample images found!")
        
except Exception as e:
    print(f"❌ Error accessing page_images directory: {e}")
    print("💡 Make sure you've run the environment setup notebook")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Sample Page Images
# MAGIC 
# MAGIC Load the pre-existing page images from the volume for vision model processing.

# COMMAND ----------

# DBTITLE 1,Load Sample Images
import io
from PIL import Image
import base64
import json

def load_sample_images(volume_path):
    """Load the two sample images from the volume"""
    images_path = f"{volume_path}/page_images"
    sample_images = {}
    
    expected_files = {
        'chart': 'report_chart_page.png',
        'table': 'report_table_page.png'
    }
    
    for image_type, filename in expected_files.items():
        try:
            file_path = f"{images_path}/{filename}"
            local_path = file_path.replace("dbfs:/", "/")
            
            # Load image using PIL
            with open(local_path, 'rb') as f:
                img_bytes = f.read()
            
            img = Image.open(io.BytesIO(img_bytes))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            sample_images[image_type] = {
                'filename': filename,
                'image': img,
                'image_bytes': img_bytes,
                'size': img.size,
                'file_size': len(img_bytes),
                'path': local_path,
                'type': image_type
            }
            
            size_mb = len(img_bytes) / (1024 * 1024)
            print(f"   ✅ Loaded {image_type} image: {filename} ({img.size} pixels, {size_mb:.2f} MB)")
            
        except Exception as e:
            print(f"   ❌ Error loading {filename}: {e}")
    
    return sample_images

def display_image(image_data, title=None):
    """Display a document image"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 16))
        plt.imshow(image_data['image'])
        plt.axis("off")
        
        display_title = title or f"{image_data['type'].title()} Page: {image_data['filename']}"
        size_info = f"Size: {image_data['size']}, {image_data['file_size']/1024:.1f} KB"
        plt.title(f"{display_title}\n{size_info}", fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Error displaying image: {e}")

# Load our sample images
print("📁 Loading sample images from volume...")
sample_images = load_sample_images(full_vol_path)

if len(sample_images) == 2:
    print(f"\n✅ Successfully loaded both sample images!")
    
    # Display both images for overview
    print(f"\n📊 Chart Page (for basic text extraction):")
    display_image(sample_images['chart'])
    
    print(f"\n📋 Table Page (for table extraction examples):")
    display_image(sample_images['table'])
    
else:
    print(f"❌ Only loaded {len(sample_images)} of 2 expected images")
    print("💡 Check that both sample images are present in page_images/")

# COMMAND ----------

# DBTITLE 1,Select Images for Processing
# We'll use specific images for specific examples
if len(sample_images) >= 1:
    # Use chart page for basic extraction examples
    chart_image = sample_images.get('chart')
    table_image = sample_images.get('table')
    
    if chart_image:
        print(f"📊 Chart image ready for basic extraction examples:")
        print(f"   File: {chart_image['filename']}")
        print(f"   Size: {chart_image['size']}")
        print(f"   Type: Chart/financial data page")
    
    if table_image:
        print(f"\n📋 Table image ready for table extraction examples:")
        print(f"   File: {table_image['filename']}")
        print(f"   Size: {table_image['size']}")  
        print(f"   Type: Table/structured data page")
else:
    print("❌ No sample images loaded - cannot proceed with examples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Vision Model Setup
# MAGIC 
# MAGIC Initialize vision-capable models for document processing.

# COMMAND ----------

# DBTITLE 1,Initialize Vision Model
from databricks_langchain import ChatDatabricks
from langchain_core.messages import HumanMessage

# Vision-capable LLM (ensure the endpoint supports vision)
vlm = ChatDatabricks(
    target_uri='databricks',
    endpoint=LLM_ENDPOINT,
    temperature=0.1
)

print(f"🤖 Vision Model Initialized:")
print(f"   Endpoint: {LLM_ENDPOINT}")

# Test vision capabilities
test_response = vlm.invoke([HumanMessage(content='Can you process document images and extract text from them?')])
print(f"✅ Model Response: {test_response.content[:150]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Image Processing Utilities
# MAGIC 
# MAGIC Helper functions for preparing document images for vision model processing.

# COMMAND ----------

# DBTITLE 1,Vision Processing Functions
def prepare_image_for_vision_model(image_data, max_width=1200, quality=90):
    """Prepare document image for vision model processing"""
    try:
        image = image_data['image']
        original_size = image.size
        
        # Resize if too large (while maintaining aspect ratio)
        if image.width > max_width:
            ratio = max_width / image.width
            new_height = int(image.height * ratio)
            image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to RGB if not already (for JPEG compatibility)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Encode as base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "base64": encoded_image,
            "original_size": original_size,
            "processed_size": image.size,
            "compression_ratio": image_data['file_size'] / len(buffer.getvalue()),
            "filename": image_data['filename']
        }
    except Exception as e:
        print(f"❌ Error preparing image: {e}")
        return None

def create_vision_prompt(task_description, image_base64):
    """Create a structured prompt for vision model processing"""
    return [
        {
            "type": "text",
            "text": task_description
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}",
                "detail": "high"
            }
        }
    ]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Basic Vision Extraction
# MAGIC 
# MAGIC Start with basic text extraction from document images.

# COMMAND ----------

# DBTITLE 1,Process Chart Image - Basic Text Extraction
if 'chart_image' in locals() and chart_image:
    print(f"📊 Processing Chart Image for Basic Text Extraction")
    print(f"   File: {chart_image['filename']}")
    print(f"   Size: {chart_image['size']}")
    
    # Prepare chart image for vision model
    prepared_chart = prepare_image_for_vision_model(chart_image)
    
    if prepared_chart:
        print(f"\n🔧 Chart image prepared for vision processing:")
        print(f"   Processed size: {prepared_chart['processed_size']}")
        print(f"   Compression ratio: {prepared_chart['compression_ratio']:.2f}x")
    else:
        print("❌ Failed to prepare chart image for vision processing")

# COMMAND ----------

# DBTITLE 1,Extract Text from Chart Page
basic_extraction_prompt = """
Extract ALL text from this financial document page and return it as clean, well-formatted markdown.

Instructions:
- Preserve the original document structure (headers, paragraphs, sections)
- Convert any tables to proper markdown table format
- Maintain the reading order (left-to-right, top-to-bottom)
- Include any footnotes, captions, or chart labels
- Use appropriate markdown formatting (headers, bold, italic, lists)
- Pay special attention to financial figures and percentages

Return only the markdown text, no explanations or metadata.
"""

if 'prepared_chart' in locals() and prepared_chart:
    # Create vision prompt for chart
    vision_prompt = create_vision_prompt(basic_extraction_prompt, prepared_chart["base64"])
    
    # Process with vision model
    print("🔍 Extracting text from chart page...")
    response = vlm.invoke([HumanMessage(content=vision_prompt)])
    chart_extracted_text = response.content
    
    print("📝 Extracted Text from Chart Page:")
    print("═" * 80)
    print(chart_extracted_text)
    print("═" * 80)
else:
    print("❌ No prepared chart image available for processing")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Table-Specific Processing
# MAGIC 
# MAGIC Now let's use the table page image to demonstrate table detection and extraction.

# COMMAND ----------

# DBTITLE 1,Process Table Image - Table Detection
if 'table_image' in locals() and table_image:
    print(f"📋 Processing Table Image for Table Detection")
    print(f"   File: {table_image['filename']}")
    print(f"   Size: {table_image['size']}")
    
    # Prepare table image for vision model
    prepared_table = prepare_image_for_vision_model(table_image)
    
    if prepared_table:
        print(f"\n🔧 Table image prepared for vision processing:")
        print(f"   Processed size: {prepared_table['processed_size']}")
        print(f"   Compression ratio: {prepared_table['compression_ratio']:.2f}x")
        
        # Display the table image for reference
        print(f"\n📋 Table Image Preview:")
        display_image(table_image, "Table Page - For Table Extraction")
    else:
        print("❌ Failed to prepare table image for vision processing")

# COMMAND ----------

# DBTITLE 1,Extract Tables from Table Page
table_detection_prompt = """
Analyze this document page and extract all table information.

Return your analysis as JSON with this structure:
{
    "tables_found": number_of_tables,
    "tables": [
        {
            "table_number": 1,
            "description": "brief description of what the table contains",
            "headers": ["list of column headers"],
            "data_markdown": "the complete table in markdown format",
            "row_count": number_of_data_rows,
            "column_count": number_of_columns
        }
    ],
    "page_summary": "brief summary of the overall page content"
}

Focus on extracting complete, accurate table data. Return only valid JSON.
"""

if 'prepared_table' in locals() and prepared_table:
    vision_prompt = create_vision_prompt(table_detection_prompt, prepared_table["base64"])
    
    print("📊 Analyzing table structure and extracting data...")
    response = vlm.invoke([HumanMessage(content=vision_prompt)])
    table_extraction_result = response.content
    
    print("🗂️ Table Extraction Results:")
    print("═" * 80)
    print(table_extraction_result)
    print("═" * 80)
    
    # Try to parse and display in a more structured way
    try:
        table_data = json.loads(table_extraction_result)
        print(f"\n📋 Structured Table Summary:")
        print(f"   Tables found: {table_data.get('tables_found', 0)}")
        
        for i, table in enumerate(table_data.get('tables', []), 1):
            print(f"\n   Table {i}: {table.get('description', 'No description')}")
            print(f"   Dimensions: {table.get('row_count', '?')} rows × {table.get('column_count', '?')} columns")
            
            # Display the markdown table if available
            if 'data_markdown' in table and table['data_markdown']:
                print(f"\n   Extracted Table Data:")
                print("   " + "\n   ".join(table['data_markdown'].split('\n')))
    except json.JSONDecodeError:
        print("   (Could not parse as JSON - raw response shown above)")
else:
    print("❌ No prepared table image available for processing")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Compare Both Sample Images
# MAGIC 
# MAGIC Process both sample images to compare vision model performance across different document types.

# COMMAND ----------

# DBTITLE 1,Compare Chart vs Table Processing
def process_sample_image(image_data, image_type):
    """Process one of our sample images with a summary prompt"""
    summary_prompt = f"""
Analyze this {image_type} document image and provide a brief summary in JSON format:

{{
    "content_type": "chart/table/mixed",
    "main_content": "brief description of the main content",
    "data_elements": {{
        "tables": number_of_tables,
        "charts_or_graphs": number_of_charts_or_figures,
        "text_sections": "few/moderate/many"
    }},
    "key_insights": ["list 2-3 key points from the image"],
    "extraction_confidence": "high/medium/low - based on image quality and text clarity"
}}

Return only JSON.
"""
    
    try:
        prepared_img = prepare_image_for_vision_model(image_data)
        if not prepared_img:
            return {"error": "Failed to prepare image", "filename": image_data['filename']}
        
        vision_prompt = create_vision_prompt(summary_prompt, prepared_img["base64"])
        response = vlm.invoke([HumanMessage(content=vision_prompt)])
        
        return {
            "filename": image_data['filename'],
            "type": image_type,
            "result": response.content,
            "image_info": {
                "original_size": prepared_img["original_size"],
                "processed_size": prepared_img["processed_size"]
            }
        }
    except Exception as e:
        return {"filename": image_data['filename'], "type": image_type, "error": str(e)}

# Process both sample images
if len(sample_images) >= 2:
    print("🔄 Processing both sample images for comparison...")
    
    results = []
    
    # Process chart image
    if 'chart_image' in locals() and chart_image:
        print("   📊 Processing chart image...")
        chart_result = process_sample_image(chart_image, "chart")
        results.append(chart_result)
    
    # Process table image  
    if 'table_image' in locals() and table_image:
        print("   📋 Processing table image...")
        table_result = process_sample_image(table_image, "table")
        results.append(table_result)
    
    # Display comparison results
    print("\n📊 Sample Images Comparison:")
    print("═" * 80)
    
    for result in results:
        if "error" in result:
            print(f"❌ {result['type'].title()} Image ({result['filename']}): {result['error']}")
        else:
            print(f"✅ {result['type'].title()} Image ({result['filename']}):")
            print(f"   {result['result']}")
        print("-" * 40)
else:
    print("❌ Need both sample images for comparison")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Quality Assessment
# MAGIC 
# MAGIC Evaluate the quality and accuracy of our vision-based extractions.

# COMMAND ----------

# DBTITLE 1,Assess Extraction Quality
def assess_vision_extraction(extracted_content, image_info, image_type):
    """Assess the quality of vision-based extraction"""
    
    if not extracted_content or "error" in str(extracted_content):
        return {"status": "failed", "error": extracted_content}
    
    # Basic content analysis
    metrics = {
        "content_length": len(extracted_content),
        "word_count": len(extracted_content.split()),
        "line_count": len(extracted_content.split('\n')),
        "has_structure": any(marker in extracted_content for marker in ['#', '|', '-', '*']),
        "has_tables": '|' in extracted_content and '-' in extracted_content,
        "has_numbers": any(char.isdigit() for char in extracted_content),
        "readability": len([c for c in extracted_content if c.isalpha()]) / len(extracted_content) if extracted_content else 0,
        "image_type": image_type,
        "image_info": image_info
    }
    
    # Quality score (simple heuristic)
    quality_score = 0
    if metrics["word_count"] > 50: quality_score += 25
    if metrics["has_structure"]: quality_score += 25  
    if metrics["readability"] > 0.6: quality_score += 25
    if metrics["has_numbers"]: quality_score += 25
    
    metrics["quality_score"] = quality_score
    metrics["quality_level"] = "High" if quality_score >= 75 else "Medium" if quality_score >= 50 else "Low"
    
    return metrics

# Assess our sample extractions
print("📈 Vision Extraction Quality Assessment:")
print("═" * 80)

# Assess chart extraction if available
if 'chart_extracted_text' in locals() and 'prepared_chart' in locals():
    chart_quality = assess_vision_extraction(chart_extracted_text, prepared_chart, "chart")
    
    print(f"📊 Quality Metrics for Chart Image ({prepared_chart['filename']}):")
    print(f"   Content length: {chart_quality['content_length']} characters")
    print(f"   Word count: {chart_quality['word_count']} words")
    print(f"   Structure detected: {chart_quality['has_structure']}")
    print(f"   Numbers present: {chart_quality['has_numbers']}")
    print(f"   Readability score: {chart_quality['readability']:.2f}")
    print(f"   Overall quality: {chart_quality['quality_level']} ({chart_quality['quality_score']}/100)")

# Assess table extraction if available
if 'table_extraction_result' in locals() and 'prepared_table' in locals():
    table_quality = assess_vision_extraction(table_extraction_result, prepared_table, "table")
    
    print(f"\n📋 Quality Metrics for Table Image ({prepared_table['filename']}):")
    print(f"   Content length: {table_quality['content_length']} characters")
    print(f"   Word count: {table_quality['word_count']} words") 
    print(f"   Structure detected: {table_quality['has_structure']}")
    print(f"   Tables detected: {table_quality['has_tables']}")
    print(f"   Numbers present: {table_quality['has_numbers']}")
    print(f"   Overall quality: {table_quality['quality_level']} ({table_quality['quality_score']}/100)")

if 'chart_extracted_text' not in locals() and 'table_extraction_result' not in locals():
    print("   No extraction data available for assessment")
    print("   💡 Run the previous extraction cells to generate assessment data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Key Takeaways & Next Steps
# MAGIC 
# MAGIC ### 🎯 **What We Demonstrated:**
# MAGIC - **Chart Processing**: Extracted text and financial data from chart-heavy pages
# MAGIC - **Table Processing**: Detected and extracted structured table data 
# MAGIC - **Quality Assessment**: Measured extraction accuracy and completeness
# MAGIC - **Comparison**: Evaluated vision model performance across different content types
# MAGIC 
# MAGIC #### ✅ **Vision Models Excel At:**
# MAGIC - Complex visual layouts (charts, tables, multi-column)
# MAGIC - Scanned or image-based documents  
# MAGIC - Documents where spatial relationships matter
# MAGIC - Extracting structured data from visual formats
# MAGIC 
# MAGIC #### 🎯 **Best Practices Learned:**
# MAGIC 1. **Use Specific Images**: Chart pages for text, table pages for structured data
# MAGIC 2. **Tailored Prompts**: Different extraction strategies for different content types
# MAGIC 3. **Quality Validation**: Always assess extraction quality and completeness
# MAGIC 4. **JSON Output**: Structured output makes downstream processing easier
# MAGIC 
# MAGIC #### 🚀 **Next Steps:**
# MAGIC - **Tutorial 03**: Document summarization and analysis
# MAGIC - **Advanced Tutorials**: OCR comparison, distributed processing
# MAGIC - **Production**: Building robust document processing pipelines
# MAGIC 
# MAGIC ### Complete Workflow Summary
# MAGIC 1. **Sample Images** → Load chart and table pages from volume
# MAGIC 2. **Targeted Processing** → Use specific images for specific extraction tasks
# MAGIC 3. **Quality Assessment** → Validate extraction accuracy
# MAGIC 4. **Structured Output** → JSON/markdown for downstream analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup
# MAGIC 
# MAGIC Clear any large variables to free up memory.

# COMMAND ----------

# Clear large image data to free memory
variables_to_clear = ['sample_images', 'chart_image', 'table_image', 'prepared_chart', 'prepared_table']

for var_name in variables_to_clear:
    if var_name in locals():
        del locals()[var_name]
        print(f"   Cleared: {var_name}")

print("🧹 Memory cleanup completed")