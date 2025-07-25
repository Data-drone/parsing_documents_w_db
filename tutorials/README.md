# Document Parsing Tutorial Path

Welcome to the Databricks Document Parsing Tutorial! This tutorial series will guide you through building a complete document processing pipeline from scratch.

## 🎯 Learning Path

### 📚 Module 0: Setup (30 mins)
**Goal**: Prepare your environment and understand the basics

- ✅ `00_setup/01_environment_setup.py` - Set up Unity Catalog, create volumes
- ✅ `00_setup/02_verify_permissions.py` - Verify access to required resources
- ✅ `00_setup/03_create_document_store.py` - Load PDFs into Delta tables

### 📚 Module 1: Foundations (2 hours)
**Goal**: Learn the core concepts of document processing

- 📄 `01_foundations/01_basic_pdf_parsing.py` - Extract text using PyMuPDF
- 📄 `01_foundations/02_exploring_llm_parsing.py` - Explore LLM capabilities for parsing
- 📄 `01_foundations/03_create_summaries.py` - Generate document summaries with LLMs

### 📚 Module 2: Advanced Parsing (3 hours)
**Goal**: Master advanced parsing techniques

- 🖼️ `02_advanced_parsing/01_split_to_images.py` - Convert PDFs to images
- 🔍 `02_advanced_parsing/02_ocr_parsing.py` - Extract text using OCR
- 🤖 `02_advanced_parsing/03_vlm_parsing.py` - Parse with Vision Language Models
- ⚡ `02_advanced_parsing/04_distributed_ray.py` - Scale with Ray on GPU clusters
- 🔌 `02_advanced_parsing/05_parsing_with_openai_api.py` - Use OpenAI-compatible APIs
- 🎯 `02_advanced_parsing/06_parsing_with_nanonets.py` - Specialized OCR models

### 📚 Module 3: Vector Search (2 hours)
**Goal**: Build semantic search capabilities

- 🧮 `03_vector_search/01_create_page_index.py` - Index individual pages
- 📊 `03_vector_search/02_create_summary_index.py` - Index document summaries

### 📚 Module 4: Production (3 hours)
**Goal**: Deploy production-ready solutions

- 🚀 `04_production_examples/01_deploy_llm_server.py` - Deploy vLLM server
- 🔗 `04_production_examples/02_build_rag_chain.py` - Build RAG application
- 📦 `04_production_examples/03_deploy_model.py` - Deploy as MLflow model

## 🛠️ Production Workflows

After completing the tutorials, you can run production workflows using Databricks bundles:

```bash
# Run basic parsing pipeline
databricks bundle run basic_parsing_pipeline

# Run OCR pipeline with image extraction
databricks bundle run ocr_pipeline

# Run complete pipeline with vector indexing
databricks bundle run full_pipeline
```

## 📁 Project Structure

```
tutorials/
├── 00_setup/           # Environment preparation
├── 01_foundations/     # Core concepts
├── 02_advanced_parsing/# Advanced techniques
├── 03_vector_search/   # Search capabilities
└── 04_production_examples/ # Production deployment
```

## 🚀 Getting Started

1. Start with Module 0 to set up your environment
2. Work through each module in order
3. Each notebook is self-contained with its own dependencies
4. Run notebooks on appropriate clusters (CPU for basic, GPU for advanced)

## 💡 Tips

- Each notebook includes `%pip install` for required dependencies
- Use widgets to parameterize catalog/schema names
- Check notebook headers for cluster requirements
- Refer to `docs/dependencies.md` for library versions

## 📚 Additional Resources

- [Databricks Vector Search Documentation](https://docs.databricks.com/en/generative-ai/vector-search.html)
- [MLflow Documentation](https://mlflow.org/)
- [Ray on Databricks](https://docs.databricks.com/en/machine-learning/ray-integration.html)

Happy Learning! 🎉 