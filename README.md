# Parsing Documents with Databricks

A developer guide to building document parsing pipelines for RAG on Databricks.
Covers the full flow: ingestion → parsing → indexing → retrieval agent.

## Quick Start

Choose your parsing path based on your documents and infrastructure:

| Path | Best For | Requires |
|------|----------|----------|
| **PyMuPDF4LLM** | Text-layer PDFs (Word exports, LaTeX) | CPU cluster |
| **ai_parse_document** | General-purpose, zero-setup | DBR 17.1+, Unity Catalog |
| **ai_query + VLM** | Custom prompts, scanned docs, figures | FMAPI endpoint (no GPU) |
| **Self-hosted VLM** | High volume, cost-sensitive | GPU cluster |

All paths share the same ingestion and downstream steps (indexing, agent).

## Prerequisites

- Databricks workspace with Unity Catalog
- DBR 15.4+ (DBR 17.1+ for `ai_parse_document`)
- For VLM paths: a Foundation Model API endpoint with vision support (e.g. Claude Sonnet)
- For self-hosted VLM: GPU cluster (single A10/T4 for dev, multi-GPU for production)

## Setup

```bash
# Clone and deploy with Databricks Asset Bundles
databricks bundle deploy -t <your_target>
```

Or run the notebooks interactively — each one is self-contained.

## Tutorial Sequence

### Step 1: Ingest Documents

`interactive_examples/Create Document Database`

Recursively scans a Unity Catalog Volume for PDF/DOC files, reads binary content,
and stores everything in a Delta table. This is the starting point for all paths.

### Step 2: Extract Text (Choose One)

**Option A: PyMuPDF4LLM (fastest, CPU-only)**

`interactive_examples/Parsing Documents`

Converts PDFs to markdown using PyMuPDF4LLM. Works great for digitally-created PDFs
with an embedded text layer. No ML models, no GPU. Start here.

**Option B: ai_parse_document (zero-setup, Databricks-native)**

`interactive_examples/Parsing w ai_parse_document`

One SQL call parses entire documents — tables, figures, layout, bounding boxes.
No libraries to install, no model to manage. Requires DBR 17.1+ and Unity Catalog.
Best option if you want to get started fast and don't need custom parsing prompts.

**Option C: ai_query + Vision LLM (custom prompts, no GPU infra)**

`interactive_examples/Split Documents` → `interactive_examples/Parsing w ai_query`

First splits PDFs into page images, then sends each page to a vision-capable
model (Claude Sonnet/Opus) via the Foundation Model API. Higher quality than
PyMuPDF4LLM for scanned docs, complex tables, and figures. Full control over
the extraction prompt. Pay-per-token cost.

**Option D: Self-hosted VLM (best throughput, GPU required)**

`interactive_examples/Split Documents` → `interactive_examples/Deploy LLM Server` → `interactive_examples/Parsing w OpenAI API`

Stands up a vLLM server with a dedicated OCR model and calls it via the
OpenAI-compatible API. Best for large batches where you want to control cost.
Includes adaptive concurrency and backpressure handling.

To scale across a multi-GPU cluster:

`interactive_examples/Parsing w Ray`

Uses Ray Data + vLLM for distributed inference across multiple GPUs.

### Note: Output Normalization

Each parsing path produces a different output schema:
- **PyMuPDF4LLM** → `document_markdown` (one row per document, `markdown_content` column)
- **ai_parse_document** → `document_ai_parsed` (VARIANT with structured elements)
- **ai_query** → `document_vlm_parsed` (one row per page, `parsed_markdown` column)
- **Self-hosted VLM** → `document_store_ocr` (one row per page, OCR text)

The downstream steps (summaries, indexing, agent) expect text content. If you're switching
between parsing paths, you may need to create a view or table that normalizes the output
into a common schema (e.g., `file_name`, `page_number`, `text_content`). Each parsing
notebook's "Next Steps" section notes what format its output is in.

### Step 3: Summarize Documents

`interactive_examples/Create Document Summaries`

Uses `ai_query()` with an LLM to generate document-level summaries and analysis.
Creates a two-tier strategy: full analysis for small docs, truncated for large ones.

### Step 4: Create Vector Indexes

`interactive_examples/Create Document Summary Index` — index on document summaries
`interactive_examples/Create Page Index` — index on page-level OCR text

Both use Databricks Vector Search with GTE embeddings. The agent uses both indexes
for two-stage retrieval (find relevant docs → search within those docs).

### Step 5: Build and Deploy the RAG Agent

`interactive_examples/Building A Compound Chain`

A LangGraph agent with two-stage search: summary index for document discovery,
OCR index for detailed content retrieval. Includes temporal awareness and
dynamic system prompt injection.

`interactive_examples/Deploy A Compound Chain`

Logs the agent to MLflow, runs evaluation with Mosaic AI judges, registers in
Unity Catalog, and deploys via Model Serving.

## Architecture

```
UC Volume (PDFs)
    │
    ▼
┌───────────────────────┐
│  Create Document DB    │  ← Step 1: Ingest
└─────────┬─────────────┘
          │
    ┌─────┼────────────┬──────────────────┐
    ▼     ▼            ▼                  ▼
 PyMuPDF  ai_parse_  Split Pages      Split Pages
 4LLM     document   + ai_query       + vLLM/Ray
 (CPU)    (SQL)      (managed VLM)    (self-hosted)
    │     │            │                  │
    └─────┼────────────┼──────────────────┘
          ▼
┌───────────────────────┐
│  Document Summaries    │  ← Step 3
└─────────┬─────────────┘
          ▼
┌───────────────────────┐
│  Vector Indexes        │  ← Step 4
│  (summary + page)      │
└─────────┬─────────────┘
          ▼
┌───────────────────────┐
│  LangGraph Agent       │  ← Step 5
│  (two-stage search)    │
└───────────────────────┘
```

## Data Model

| Table | Created By | Contents |
|-------|-----------|----------|
| `document_store` | Create Document Database | File metadata + binary content |
| `document_markdown` | Parsing Documents | PyMuPDF4LLM markdown extraction |
| `document_ai_parsed` | Parsing w ai_parse_document | Native Databricks parsed output |
| `document_page_docs` | Split Documents | Per-page PNG images |
| `document_vlm_parsed` | Parsing w ai_query | VLM-extracted markdown per page |
| `document_store_ocr` | Parsing w OpenAI API | Self-hosted VLM OCR text per page |
| `parsed_markdown_pages` | Parsing w Ray | Ray-processed VLM OCR text |
| `document_analysis_simple` | Create Document Summaries | AI-generated document summaries |
| `flattened_documents` | Create Document Summary Index | Flattened summaries for vector search |
| `document_ocr_vector_ready` | Create Page Index | Cleaned OCR text for vector search |

## Databricks Asset Bundles

The repo includes DAB configuration for automated workflows:

- `src/workflows/parsing_workflow.yml` — CPU parsing pipeline (ingest → PyMuPDF4LLM)
- `src/workflows/ocr_parsing_workflow.yml` — OCR pipeline (ingest → split → images)

Deploy with: `databricks bundle deploy -t mg_dev`

## Legacy Notebooks

These notebooks are kept for reference but have been superseded:

- `Parsing Documents - OCR.py` — Tesseract OCR (replaced by VLM approaches)
- `Parsing Documents - VLM.py` — Early VLM exploration (incomplete)
- `Exploring LLM Parsing in DBX.py` — LLM parsing experiments

## Viewer App

`apps/viewer_app/` contains a Streamlit app for browsing documents, viewing
PDF pages side-by-side with extracted markdown, and running vector search queries.
