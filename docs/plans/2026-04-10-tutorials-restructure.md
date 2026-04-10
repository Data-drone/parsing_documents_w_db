# Tutorials Restructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reshape the tutorials/ folder to match the v2 parsing-focused plan — 7 happy-path notebooks covering 3 core parsers + comparison, with Docling optional and self-hosted VLM in advanced/.

**Architecture:** Reuse existing notebook code from both `main` (tested ai_parse_document, ai_query) and current `dev` (Docling, setup, split-to-images, self-hosted VLM). Each parser notebook writes to a standard output schema. New comparison notebook aggregates results with deterministic metrics + selective LLM judge.

**Tech Stack:** Databricks notebooks (Python), PySpark, PyMuPDF4LLM, Docling, ai_parse_document SQL, ai_query SQL, vLLM

---

## Source Mapping

| Target file | Source | Action |
|---|---|---|
| `00_setup/01_environment_setup.py` | `tutorials/00_setup/01_environment_setup.py` (dev) | Edit: strip vector search/embedding refs, add FMAPI preflight, add demo_mode widget |
| `00_setup/02_prepare_documents.py` | `tutorials/00_setup/02_create_document_store.py` (dev) + `tutorials/02_advanced_parsing/01_split_to_images.py` (dev) | Merge: doc store creation + PDF-to-image splitting in one notebook |
| `01_parse/01_pymupdf.py` | `tutorials/01_foundations/01_basic_pdf_parsing.py` (dev) | Edit: adapt output to standard schema, add timing/cost tracking |
| `01_parse/02_ai_parse_document.py` | `main:interactive_examples/Parsing w ai_parse_document.py` | Copy from main, edit: adapt output to standard schema |
| `01_parse/03_ai_query_vlm.py` | `main:interactive_examples/Parsing w ai_query.py` | Copy from main, edit: adapt output to standard schema |
| `01_parse/04_docling.py` | `tutorials/02_advanced_parsing/02_parsing_w_docling.py` (dev) | Edit: adapt output to standard schema, add timing/cost tracking |
| `02_compare/01_quality_comparison.py` | NEW | Write from scratch: deterministic metrics + LLM judge |
| `02_compare/02_choose_and_export.py` | NEW | Write from scratch: pick parser, export clean table |
| `advanced/self_hosted_vlm/01_deploy_vllm_server.py` | `tutorials/02_advanced_parsing/03a_custom_vlm_server.py` (dev) | Move as-is |
| `advanced/self_hosted_vlm/02_query_vlm_server.py` | `tutorials/02_advanced_parsing/03b_querying_custom_vlm_server.py` (dev) | Move as-is |
| `advanced/self_hosted_vlm/03_distributed_batch_ray.py` | `tutorials/02_advanced_parsing/04_distributed_batch_w_custom_vlm.py` (dev) | Move as-is |

## Standard Output Schema

All parsers MUST write to this schema (one table per parser, same columns):

```sql
CREATE TABLE {catalog}.{schema}.parsed_{method} (
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

## Shared Config Pattern

Every notebook starts with this widget block (copy-paste across all):

```python
# Get current user for catalog naming
current_user = spark.sql("SELECT current_user()").first()[0]
username = current_user.split('@')[0].replace('.', '_')

dbutils.widgets.text("catalog_name", f"{username}_document_parsing", "Catalog")
dbutils.widgets.text("schema_name", "tutorials", "Schema")
dbutils.widgets.text("volume_name", "sample_docs", "Volume")
dbutils.widgets.dropdown("demo_mode", "true", ["true", "false"], "Demo Mode (2 pages)")

CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA  = dbutils.widgets.get("schema_name")
VOLUME  = dbutils.widgets.get("volume_name")
DEMO_MODE = dbutils.widgets.get("demo_mode") == "true"
```

No `.env` files. No `dotenv` imports.

---

### Task 1: Clean out old folder structure

**Files:**
- Delete: `tutorials/01_foundations/` (entire directory)
- Delete: `tutorials/03_vector_search/` (entire directory)
- Delete: `tutorials/04_production_examples/` (entire directory)
- Delete: `tutorials/00_setup/README.md`
- Delete: `tutorials/02_advanced_parsing/` (entire directory — contents will be reorganized)

**Step 1: Remove out-of-scope directories**

```bash
cd /workspace/group/repo
git rm -r tutorials/01_foundations/
git rm -r tutorials/03_vector_search/
git rm -r tutorials/04_production_examples/
git rm tutorials/00_setup/README.md
```

**Step 2: Stash the advanced_parsing files we need before deleting**

```bash
# Copy files we want to keep to a temp location
mkdir -p /tmp/restructure_stash
cp tutorials/02_advanced_parsing/02_parsing_w_docling.py /tmp/restructure_stash/
cp tutorials/02_advanced_parsing/03a_custom_vlm_server.py /tmp/restructure_stash/
cp tutorials/02_advanced_parsing/03b_querying_custom_vlm_server.py /tmp/restructure_stash/
cp tutorials/02_advanced_parsing/04_distributed_batch_w_custom_vlm.py /tmp/restructure_stash/

git rm -r tutorials/02_advanced_parsing/
```

**Step 3: Create new directory structure**

```bash
mkdir -p tutorials/01_parse
mkdir -p tutorials/02_compare
mkdir -p tutorials/advanced/self_hosted_vlm
```

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: clear old tutorial structure for v2 parsing-focused layout

Remove foundations, vector_search, production_examples modules.
Remove advanced_parsing (will be reorganized in next commits).
Keep 00_setup as starting point."
```

---

### Task 2: Edit 00_setup/01_environment_setup.py

**Files:**
- Modify: `tutorials/00_setup/01_environment_setup.py`

**Changes:**
1. Remove dotenv imports and usage
2. Replace env-var-based config with widget-only config using the shared pattern above
3. Remove vector search endpoint widget and verification
4. Remove embedding model widget
5. Remove LLM model widget (parser notebooks handle their own models)
6. Add `demo_mode` dropdown widget
7. Add FMAPI preflight check: try `SELECT ai_query('databricks-meta-llama-3-3-70b-instruct', 'ping')` and report
8. Simplify to: create catalog, create schema, create volume, copy sample PDFs, verify, preflight

The notebook should end with a summary of what was created and which parser notebooks to run next.

**Step: Commit**

```bash
git add tutorials/00_setup/01_environment_setup.py
git commit -m "refactor: simplify environment setup for v2 parsing tutorial

Widget-only config (no dotenv), removed vector search and embedding
refs, added demo_mode widget and FMAPI preflight check."
```

---

### Task 3: Create 00_setup/02_prepare_documents.py

**Files:**
- Rename: `tutorials/00_setup/02_create_document_store.py` → `tutorials/00_setup/02_prepare_documents.py`

**Changes:**
Merge the document store creation AND the split-to-images functionality into one notebook. This notebook:

1. Uses shared config pattern (widget-only, no dotenv)
2. READ_FILES from volume → `document_store` table (binary PDF content + metadata)
3. Split PDFs to page images → `document_page_images` table (one row per page with PNG binary)
4. If `demo_mode == true`, only process first 2 pages per document
5. Summary: "You now have X documents and Y page images ready for parsing"

The page-image splitting logic comes from `01_split_to_images.py` (stashed in /tmp). Simplify it — remove the excessive config widgets and use the shared pattern.

**Step: Commit**

```bash
git add tutorials/00_setup/02_prepare_documents.py
git commit -m "feat: combined prepare_documents notebook (doc store + page images)

Merges document store creation and PDF-to-image splitting into one
notebook. Supports demo_mode for quick iteration."
```

---

### Task 4: Create 01_parse/01_pymupdf.py

**Files:**
- Create: `tutorials/01_parse/01_pymupdf.py`

**Source:** `tutorials/01_foundations/01_basic_pdf_parsing.py` (already deleted but we have its content in memory / can reconstruct from the code we read)

**Changes from source:**
1. Use shared config pattern (no dotenv)
2. Read from `{CATALOG}.{SCHEMA}.document_store` table
3. Keep the pandas UDF approach for distributed processing
4. Track wall-clock time per document
5. Write to standard output schema: `{CATALOG}.{SCHEMA}.parsed_pymupdf`
6. Set `parse_method = 'pymupdf'`, `estimated_cost_usd = 0.0`
7. Set `contains_tables = False` (PyMuPDF doesn't detect tables)
8. Remove the visual comparison section (nice but not needed for the parser notebook)
9. If demo_mode, limit to first 2 documents

Structure:
- Cmd 1: Title markdown
- Cmd 2: pip install + restart
- Cmd 3: Config widgets
- Cmd 4: Load document store
- Cmd 5: Define extraction UDF
- Cmd 6: Process documents with timing
- Cmd 7: Write to standard schema table
- Cmd 8: Display results summary

**Step: Commit**

```bash
git add tutorials/01_parse/01_pymupdf.py
git commit -m "feat: add PyMuPDF parser notebook with standard output schema"
```

---

### Task 5: Create 01_parse/02_ai_parse_document.py

**Files:**
- Create: `tutorials/01_parse/02_ai_parse_document.py`

**Source:** `main:interactive_examples/Parsing w ai_parse_document.py`

**Changes from source:**
1. Use shared config pattern (no dotenv)
2. Read files from volume (same as source)
3. Keep the core `ai_parse_document()` SQL logic
4. Track wall-clock time
5. Write to standard output schema: `{CATALOG}.{SCHEMA}.parsed_ai_parse_document`
6. Set `parse_method = 'ai_parse_document'`
7. Set `contains_tables` based on whether VARIANT output contains table elements
8. Estimate cost from Databricks pricing (placeholder comment — actual pricing varies)
9. If demo_mode, limit files processed

**Step: Commit**

```bash
git add tutorials/01_parse/02_ai_parse_document.py
git commit -m "feat: add ai_parse_document parser notebook with standard output schema"
```

---

### Task 6: Create 01_parse/03_ai_query_vlm.py

**Files:**
- Create: `tutorials/01_parse/03_ai_query_vlm.py`

**Source:** `main:interactive_examples/Parsing w ai_query.py`

**Changes from source:**
1. Use shared config pattern (no dotenv)
2. Add `vlm_endpoint` widget defaulting to `databricks-claude-sonnet-4`
3. Read page images from `{CATALOG}.{SCHEMA}.document_page_images`
4. Keep the core `ai_query()` logic for vision models
5. Track wall-clock time per page
6. Write to standard output schema: `{CATALOG}.{SCHEMA}.parsed_ai_query_vlm`
7. Set `parse_method = 'ai_query_vlm'`
8. Estimate cost from token usage (input ~1-2K tokens per page, output ~500-2K)
9. Set `contains_tables = False` (could be enhanced later with heuristic)
10. If demo_mode, limit to first 2 pages

**Step: Commit**

```bash
git add tutorials/01_parse/03_ai_query_vlm.py
git commit -m "feat: add ai_query VLM parser notebook with standard output schema"
```

---

### Task 7: Create 01_parse/04_docling.py

**Files:**
- Create: `tutorials/01_parse/04_docling.py`

**Source:** `/tmp/restructure_stash/02_parsing_w_docling.py`

**Changes from source:**
1. Use shared config pattern (no dotenv)
2. Read PDFs from volume path (same as source)
3. Keep the Docling converter logic (basic + advanced with OCR + tables)
4. Remove the configuration comparison section (too verbose for a parser notebook)
5. Track wall-clock time
6. Write to standard output schema: `{CATALOG}.{SCHEMA}.parsed_docling`
7. Set `parse_method = 'docling'`, `estimated_cost_usd = 0.0`
8. Set `contains_tables` based on Docling's table detection
9. If demo_mode, limit files processed

**Step: Commit**

```bash
git add tutorials/01_parse/04_docling.py
git commit -m "feat: add Docling parser notebook with standard output schema"
```

---

### Task 8: Move self-hosted VLM notebooks to advanced/

**Files:**
- Create: `tutorials/advanced/self_hosted_vlm/01_deploy_vllm_server.py` from stash
- Create: `tutorials/advanced/self_hosted_vlm/02_query_vlm_server.py` from stash
- Create: `tutorials/advanced/self_hosted_vlm/03_distributed_batch_ray.py` from stash

**Step 1: Copy from stash**

```bash
cp /tmp/restructure_stash/03a_custom_vlm_server.py tutorials/advanced/self_hosted_vlm/01_deploy_vllm_server.py
cp /tmp/restructure_stash/03b_querying_custom_vlm_server.py tutorials/advanced/self_hosted_vlm/02_query_vlm_server.py
cp /tmp/restructure_stash/04_distributed_batch_w_custom_vlm.py tutorials/advanced/self_hosted_vlm/03_distributed_batch_ray.py
```

**Step 2: Commit**

```bash
git add tutorials/advanced/
git commit -m "feat: move self-hosted VLM notebooks to advanced/"
```

---

### Task 9: Create 02_compare/01_quality_comparison.py

**Files:**
- Create: `tutorials/02_compare/01_quality_comparison.py`

**This is a NEW notebook — the centrepiece.**

Structure:
1. Config widgets (shared pattern + `judge_endpoint` widget for LLM-as-judge)
2. Load all parsed tables: `parsed_pymupdf`, `parsed_ai_parse_document`, `parsed_ai_query_vlm`, `parsed_docling` — handle missing tables gracefully
3. Union all into a single comparison DataFrame
4. **Tier 1 — Deterministic metrics** (always run):
   - `text_length`: character count of parsed_text
   - `word_count`: word count
   - `contains_tables`: boolean from parser output
   - `parse_duration_seconds`: from parser output
   - `estimated_cost_usd`: from parser output
   - Per-page and per-method aggregation
5. **Tier 2 — LLM-as-judge** (selective, on pages with tables or complex layouts):
   - Join page images from `document_page_images`
   - For flagged pages: send image + parsed text to `ai_query(judge_endpoint, ...)`
   - Score: completeness (1-5), accuracy (1-5), formatting (1-5)
   - Only run on a sample (e.g., first 3 "interesting" pages per parser)
6. Summary dashboard: comparison table with all metrics per parser
7. Save comparison results to `{CATALOG}.{SCHEMA}.parser_comparison`

**Step: Commit**

```bash
git add tutorials/02_compare/01_quality_comparison.py
git commit -m "feat: add quality comparison notebook (deterministic + LLM judge)"
```

---

### Task 10: Create 02_compare/02_choose_and_export.py

**Files:**
- Create: `tutorials/02_compare/02_choose_and_export.py`

Structure:
1. Config widgets (shared pattern + `preferred_parser` dropdown)
2. Load comparison results from `parser_comparison` table
3. Display summary from comparison notebook
4. Widget to select preferred parser (or "best_per_page" for blended approach)
5. Export selected parser's output to `{CATALOG}.{SCHEMA}.final_parsed_documents`
6. The export table is the "handoff point" — ready for chunking, indexing, RAG, etc.
7. Print summary: "Exported N pages from {parser} to final_parsed_documents"

**Step: Commit**

```bash
git add tutorials/02_compare/02_choose_and_export.py
git commit -m "feat: add choose-and-export notebook for parser selection"
```

---

### Task 11: Update tutorials/README.md

**Files:**
- Modify: `tutorials/README.md`

Replace entire contents with the v2 structure, decision matrix, and getting started guide. Keep it concise — no emojis overload.

**Step: Commit**

```bash
git add tutorials/README.md
git commit -m "docs: rewrite tutorials README for v2 parsing-focused structure"
```

---

### Task 12: Update root README.md

**Files:**
- Modify: `README.md`

Update to reflect the new parsing-focused scope. Remove references to RAG, agents, deployment. Point to tutorials/README.md for the full structure.

**Step: Commit**

```bash
git add README.md
git commit -m "docs: update root README for v2 parsing tutorial focus"
```

---

### Task 13: Final cleanup and verification

**Step 1: Verify file structure**

```bash
find tutorials/ -name "*.py" -o -name "*.md" | sort
```

Expected:
```
tutorials/00_setup/01_environment_setup.py
tutorials/00_setup/02_prepare_documents.py
tutorials/01_parse/01_pymupdf.py
tutorials/01_parse/02_ai_parse_document.py
tutorials/01_parse/03_ai_query_vlm.py
tutorials/01_parse/04_docling.py
tutorials/02_compare/01_quality_comparison.py
tutorials/02_compare/02_choose_and_export.py
tutorials/advanced/self_hosted_vlm/01_deploy_vllm_server.py
tutorials/advanced/self_hosted_vlm/02_query_vlm_server.py
tutorials/advanced/self_hosted_vlm/03_distributed_batch_ray.py
tutorials/README.md
```

**Step 2: Verify no stale references**

```bash
grep -r "01_foundations\|03_vector_search\|04_production\|dotenv\|load_dotenv" tutorials/ --include="*.py"
```

Should return empty (no dotenv, no old module references).

**Step 3: Check git status**

```bash
git status
git log --oneline -15
```

**Step 4: Clean up stash**

```bash
rm -rf /tmp/restructure_stash
```
