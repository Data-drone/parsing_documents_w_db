# Module 0: Environment Setup

This module helps you prepare your Databricks environment for document parsing.

## Prerequisites

- Access to a Databricks workspace
- Unity Catalog enabled
- Permissions to create catalogs, schemas, and volumes
- Access to compute clusters (CPU for basic, GPU for advanced)

## Notebooks in this Module

### 01_environment_setup.py
**Time**: 10 minutes  
**Purpose**: Create the necessary Unity Catalog objects and verify environment
- Creates catalog and schema
- Sets up volumes for document storage
- Verifies Databricks configuration

### 02_verify_permissions.py
**Time**: 5 minutes  
**Purpose**: Check all required permissions and access
- Tests Unity Catalog access
- Verifies compute permissions
- Checks model serving endpoints

### 03_exploring_llm_parsing.py
**Time**: 10 minutes  
**Purpose**: Explore LLM capabilities for document parsing
- Test different LLM endpoints
- Understand parsing capabilities
- Compare parsing approaches

### 04_diagnose_dataset.py
**Time**: 5 minutes  
**Purpose**: Analyze your document dataset
- Check document types and sizes
- Estimate processing requirements
- Plan resource allocation

## Next Steps

Once you've completed all setup notebooks, proceed to Module 1: Foundations to start building your document processing pipeline. 