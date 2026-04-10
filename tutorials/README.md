# Document Parsing Tutorials

Compare different PDF parsing techniques on Databricks and find the best one for your documents.

## Quick Start

1. Run `00_setup/01_environment_setup.py` — create catalog, schema, volume, preflight checks
2. Run `00_setup/02_prepare_documents.py` — load PDFs + create page images
3. Run any parser(s) from `01_parse/` (they're independent — run one or all)
4. Run `02_compare/01_quality_comparison.py` — compare results side by side
5. Run `02_compare/02_choose_and_export.py` — pick a parser, export for downstream use

## Parser Decision Matrix

| Parser | Cost | Speed | Scanned PDFs | Tables | Best For |
|--------|------|-------|-------------|--------|----------|
| PyMuPDF | Free | Fast | No | No | Clean digital PDFs |
| ai_parse_document | Per-page | Medium | Yes | Yes | General purpose, zero setup |
| ai_query VLM | Per-token | Slow | Yes | Partial | Complex layouts, highest accuracy |
| Docling | Free | Slow | Yes | Yes | Open-source with OCR + tables |

## Structure

```
tutorials/
├── 00_setup/
│   ├── 01_environment_setup.py     # Catalog, schema, volume, preflight
│   └── 02_prepare_documents.py     # Load PDFs + split to page images
│
├── 01_parse/
│   ├── 01_pymupdf.py               # Open source, CPU, free
│   ├── 02_ai_parse_document.py     # Native Databricks SQL
│   ├── 03_ai_query_vlm.py          # Managed VLMs (Claude, Gemini)
│   └── 04_docling.py               # Open source + OCR + tables
│
├── 02_compare/
│   ├── 01_quality_comparison.py    # Metrics + optional LLM judge
│   └── 02_choose_and_export.py     # Pick parser, export results
│
└── advanced/
    └── self_hosted_vlm/            # GPU cluster required
        ├── 01_deploy_vllm_server.py
        ├── 02_query_vlm_server.py
        └── 03_distributed_batch_ray.py
```

## Requirements

- Databricks workspace with Unity Catalog
- DBR 17.1+ (for ai_parse_document)
- CPU cluster for PyMuPDF and Docling
- FMAPI access for ai_query VLM
- GPU cluster only needed for advanced/self_hosted_vlm

## All Parsers Write the Same Schema

```sql
parsed_{method} (
  source_file STRING,
  file_name STRING,
  page_number INT,
  parsed_text STRING,
  contains_tables BOOLEAN,
  parse_method STRING,
  parse_duration_seconds FLOAT,
  estimated_cost_usd FLOAT,
  parsed_at TIMESTAMP
)
```

This makes the comparison notebook work automatically with any combination of parsers.
