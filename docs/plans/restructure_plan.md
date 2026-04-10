# Document Parsing Tutorial — Restructure Plan (v2)

## Focus
Parsing comparison and validation. The repo answers one question:
**"Which parsing technique should I use for my documents?"**

RAG, agents, and deployment are out of scope for this iteration.

## Design Principles
1. "Pick your parser" — don't force users through every option
2. Deterministic metrics first, LLM-as-judge second (hybrid evaluation)
3. Widget-based config everywhere (no .env files)
4. ai_parse_document as recommended default parser
5. Standard output schema so comparison notebook works with any parser
6. Demo mode widget — process 2 pages quickly, then scale to full corpus
7. Self-hosted VLM is advanced/optional (requires GPU cluster)

## Proposed Structure

```
tutorials/
├── README.md                          # Overview, decision matrix, getting started
├── 00_setup/
│   ├── 01_environment_setup.py        # UC catalog, schema, volumes, sample PDFs
│   └── 02_prepare_documents.py        # READ_FILES → Delta table with binary content
│
├── 01_parse/
│   ├── 01_pymupdf.py                  # Open source, CPU-only, zero API cost
│   ├── 02_ai_parse_document.py        # Native SQL, zero deps, VARIANT output
│   ├── 03_ai_query_vlm.py            # Managed VLMs via FMAPI (Claude, Gemini, etc.)
│   └── 04_docling.py                  # Open source with OCR + table detection
│
├── 02_compare/
│   ├── 01_quality_comparison.py       # Deterministic metrics + selective LLM judge
│   └── 02_choose_and_export.py        # Pick best parser, export for downstream use
│
├── advanced/
│   └── self_hosted_vlm/
│       ├── 01_deploy_vllm_server.py   # vLLM + Nanonets/Qwen on GPU cluster
│       ├── 02_query_vlm_server.py     # OpenAI-compatible API queries
│       └── 03_distributed_batch_ray.py # Ray Data for batch processing
│
├── apps/
│   └── viewer_app/                    # Existing Streamlit PDF/parsing viewer
│       ├── app.yaml
│       ├── pdf_viewer.py
│       └── requirements.txt
│
└── docs/
    └── plans/
        └── restructure_plan.md        # This file
```

## Happy Path (7 notebooks)

```
00_setup/01 → 00_setup/02 → 01_parse/01 + 01_parse/02 + 01_parse/03 → 02_compare/01 → 02_compare/02
```

Core parsers: PyMuPDF, ai_parse_document, ai_query VLM (3 methods).
Docling (01_parse/04) is optional — newer, needs validation.
Self-hosted VLM in advanced/ for power users.

## What Each Notebook Does

### 00_setup/01_environment_setup.py
- Create UC catalog (auto-derived from username or widget)
- Create schema, volumes
- Download sample PDF documents into volume
- Preflight checks: verify FMAPI endpoints exist, DBR version, permissions
- Demo mode widget: `demo_mode = True` processes 2-3 sample pages only

### 00_setup/02_prepare_documents.py
- READ_FILES from volume → Delta table with binary content + file metadata
- This is the shared input for all parsers
- Split PDFs to page images (prerequisite for VLM-based parsers)
- Store both binary PDFs and page images

### 01_parse/01_pymupdf.py
- PyMuPDF4LLM for text-based PDF extraction
- UDF for distributed processing across files
- Tracks: wall-clock time, pages processed, estimated cost ($0)
- Writes to standard output schema
- Explains: best for clean digital PDFs, no API cost, CPU-only

### 01_parse/02_ai_parse_document.py
- Native SQL `ai_parse_document()` on binary content
- VARIANT output handling (CAST patterns for arrays)
- Structured element extraction (tables, headers, paragraphs)
- Tracks: wall-clock time, pages processed, estimated cost
- Writes to standard output schema
- Explains: native Databricks, zero Python deps, structured output

### 01_parse/03_ai_query_vlm.py
- `ai_query()` with vision models on page images
- Configurable model selection (Claude Sonnet, Gemini, etc.)
- failOnError handling, FMAPI billing explanation
- Tracks: wall-clock time, pages processed, estimated cost per model
- Writes to standard output schema
- Explains: best for complex layouts, scanned docs, highest accuracy ceiling

### 01_parse/04_docling.py
- Docling with built-in OCR + table structure detection
- Open source, runs on CPU (slower than PyMuPDF but handles more)
- Tracks: wall-clock time, pages processed, estimated cost ($0)
- Writes to standard output schema
- Explains: good middle ground — open source but handles tables/OCR

### 02_compare/01_quality_comparison.py
*Centrepiece notebook.* Two-tier evaluation:

**Tier 1 — Deterministic metrics (always run):**
- Text length ratio (parsed vs source page count)
- Token/word recall against reference
- Table detection: did the parser find tables? Row/column count accuracy
- Structural element count (headers, lists, code blocks)
- Wall-clock time per page
- Estimated cost per page

**Tier 2 — LLM-as-judge (selective, on hard pages):**
- Send original page image + parsed text to vision model via ai_query
- Score: completeness (1-5), accuracy (1-5), table quality (1-5)
- Only run on pages flagged as "interesting" (contain tables, images, mixed layouts)
- Cross-parser comparison scorecard

**Output:** Summary dashboard table comparing all parsers across all metrics.

### 02_compare/02_choose_and_export.py
- Review comparison results
- Select preferred parser (or blend — e.g., PyMuPDF for text pages, VLM for table pages)
- Export final parsed corpus to a clean Delta table
- Standard schema ready for downstream use (chunking, indexing, RAG, etc.)

### advanced/self_hosted_vlm/
- 01: Deploy vLLM server with Nanonets/Qwen on GPU cluster
- 02: Query the server with OpenAI-compatible API
- 03: Distributed batch with Ray Data
- For maximum control / cost optimization at scale
- Writes to same standard output schema for comparison compatibility

## Standard Output Schema

All parsers write to the same schema:

```sql
parsed_documents (
  source_file STRING,           -- full path to source file
  file_name STRING,             -- just the filename
  page_number INT,              -- NULL for whole-doc parsers
  parsed_text STRING,           -- extracted text/markdown
  contains_tables BOOLEAN,      -- detected table content
  parse_method STRING,          -- 'pymupdf', 'ai_parse_document', 'ai_query_vlm', etc.
  parse_duration_seconds FLOAT, -- wall-clock time for this page/doc
  estimated_cost_usd FLOAT,     -- API cost estimate (0.0 for open-source)
  parsed_at TIMESTAMP
)
```

## What Gets Dropped/Archived

- "Exploring LLM Parsing in DBX" → absorbed into 03_ai_query_vlm
- "Parsing Documents - OCR" → absorbed into advanced/self_hosted_vlm
- "Parsing Documents - VLM" → absorbed into 03_ai_query_vlm
- "Deploy LLM Server" → folded into advanced/01
- All RAG/agent/deploy notebooks → out of scope for this iteration
- Two-stage search logic → out of scope
- Summary generation → out of scope

## Source Material (existing notebooks to draw from)

**Current branch (feat/doc-overhaul-and-new-notebooks):**
- `Parsing Documents.py` → PyMuPDF4LLM (ACTIVE, tested)
- `Parsing w ai_parse_document.py` → ai_parse_document (ACTIVE, tested on DBR 18.0)
- `Parsing w ai_query.py` → ai_query VLM (ACTIVE, tested on DBR 18.0)
- `Split Documents.py` → PDF to page images (ACTIVE)
- `Parsing w OpenAI API.py` → Self-hosted VLM (ACTIVE)
- `Parsing_w_ray.py` → Ray distributed (EXPERIMENTAL)

**Dev branch (origin/dev):**
- `tutorials/02_advanced_parsing/02_parsing_w_docling.py` → Docling (NEW)
- Enhanced versions of setup and foundations notebooks

## Git Branch Strategy

Target state: `main` (stable) + `dev` (active work).
1. Merge current feature branch work into a clean dev branch
2. Cherry-pick useful content from origin/dev (especially Docling notebook)
3. Restructure into tutorials/ folder layout on dev
4. main stays stable until tutorials are tested and ready
