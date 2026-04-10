# Parsing Documents with Databricks

Compare different PDF parsing techniques on Databricks and find the best one for your documents.

## What This Repo Does

Runs the same set of PDFs through multiple parsers, measures quality and timing,
and helps you pick the right parser. All results use a standard output schema
so comparison is automatic.

## Quick Start

```
00_setup/01_environment_setup  →  00_setup/02_prepare_documents
                                         │
                    ┌────────────┬────────┼────────┬───────────┐
                    ▼            ▼        ▼        ▼           ▼
              01_pymupdf  02_ai_parse  03_ai_  04_docling  advanced/
              (CPU,free)  _document    query   (CPU,free)  self_hosted
                          (SQL)       _vlm                 _vlm (GPU)
                    │            │        │        │
                    └────────────┴────────┴────────┘
                                  ▼
                    02_compare/01_quality_comparison
                                  ▼
                    02_compare/02_choose_and_export
```

See [tutorials/README.md](tutorials/README.md) for the full structure and decision matrix.

## Parser Options

| Parser | Speed | Scanned PDFs | Tables | Best For |
|--------|-------|-------------|--------|----------|
| PyMuPDF | Fast | No | No | Clean digital PDFs |
| ai_parse_document | Medium | Yes | Yes | General purpose, zero setup |
| ai_query VLM | Slow | Yes | Partial | Complex layouts, highest accuracy |
| Docling | Slow | Yes | Yes | Open-source with OCR + tables |
| Self-hosted VLM | Fast at scale | Yes | Yes | High volume |

## Requirements

- Databricks workspace with Unity Catalog
- DBR 17.1+ (for `ai_parse_document`)
- CPU cluster for PyMuPDF and Docling
- FMAPI access for ai_query VLM (e.g., Claude Sonnet endpoint)
- GPU cluster only for advanced/self_hosted_vlm

## Repo Structure

```
tutorials/           # The main tutorial notebooks (see tutorials/README.md)
apps/viewer_app/     # Streamlit PDF viewer app (deployed as Databricks App)
docs/plans/          # Architecture and planning documents
src/                 # DAB workflow definitions
```

## Viewer App

`apps/viewer_app/` contains a Streamlit app for browsing parsed documents
and viewing PDF pages side-by-side with extracted text.
