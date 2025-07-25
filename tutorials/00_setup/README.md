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

### 02_create_document_store.py
**Time**: 15 minutes  
**Purpose**: Load PDFs into a Delta document store for downstream parsing
- Scans a Unity Catalog volume for supported documents
- Stores binary content and metadata in Delta Lake
- Provides verification queries to validate the loaded data

## Next Steps

Once you've completed all setup notebooks, proceed to Module 1: Foundations to start building your document processing pipeline. 