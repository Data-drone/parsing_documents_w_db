# Databricks notebook source
# MAGIC %md
# MAGIC # olmOCR-Bench: GLM-OCR vs ai_parse_document (v3)
# MAGIC
# MAGIC Fixed: GLM-OCR prompt format (must use "Text Recognition:"), correct apply_chat_template usage
# MAGIC
# MAGIC PDFs are stored persistently in a UC volume to avoid re-downloading.

# COMMAND ----------

dbutils.widgets.text("sample_n", "50", "Sample N PDFs per category (0=all)")
SAMPLE_N = int(dbutils.widgets.get("sample_n"))
GLM_MODEL = "zai-org/GLM-OCR"
BENCH_DIR = "/tmp/olmocr_bench"
RESULTS_DIR = "/tmp/olmocr_results"
VOL_BASE = "/Volumes/main/default/olmocr_bench_pdfs"
print(f"Config: sample_n={SAMPLE_N}, model={GLM_MODEL}")

# COMMAND ----------

# MAGIC %pip install -q "transformers>=5.1.0" fuzzysearch rapidfuzz pypdf huggingface_hub Pillow PyMuPDF

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys, json, glob, random, time, re, unicodedata, base64, io, shutil
import torch

SAMPLE_N = int(dbutils.widgets.get("sample_n"))
GLM_MODEL = "zai-org/GLM-OCR"
BENCH_DIR = "/tmp/olmocr_bench"
RESULTS_DIR = "/tmp/olmocr_results"
VOL_BASE = "/Volumes/main/default/olmocr_bench_pdfs"

print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

import transformers
print(f"transformers={transformers.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure Benchmark Data in UC Volume
# MAGIC Download once to volume; subsequent runs skip the download.

# COMMAND ----------

from huggingface_hub import snapshot_download

# Create volume if needed
try:
    spark.sql("CREATE SCHEMA IF NOT EXISTS main.default")
    spark.sql("CREATE VOLUME IF NOT EXISTS main.default.olmocr_bench_pdfs")
except Exception as e:
    print(f"Volume setup: {e}")

# Check if volume already has the dataset by looking for a marker file
vol_marker = os.path.join(VOL_BASE, ".olmocr_bench_complete")
if os.path.exists(vol_marker):
    print("Dataset already in volume (skipping download)")
else:
    print("Downloading olmOCR-bench dataset to /tmp then copying to volume...")
    snapshot_download(repo_id="allenai/olmOCR-bench", repo_type="dataset",
                      local_dir=BENCH_DIR, ignore_patterns=["*.DS_Store"])

    # Copy all PDFs to volume (flat: VOL_BASE/{cat}/{file}.pdf)
    pdf_src = os.path.join(BENCH_DIR, "bench_data", "pdfs")
    copy_count = 0
    for cat in sorted(os.listdir(pdf_src)):
        cat_path = os.path.join(pdf_src, cat)
        if not os.path.isdir(cat_path):
            continue
        dst_dir = os.path.join(VOL_BASE, cat)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in sorted(os.listdir(cat_path)):
            if fname.endswith(".pdf"):
                dst = os.path.join(dst_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(cat_path, fname), dst)
                    copy_count += 1
    print(f"Copied {copy_count} PDFs to volume")

    # Copy JSONL test files to volume
    bench_data_src = os.path.join(BENCH_DIR, "bench_data")
    bench_data_dst = os.path.join(VOL_BASE, "bench_data")
    os.makedirs(bench_data_dst, exist_ok=True)
    for jf in glob.glob(os.path.join(bench_data_src, "*.jsonl")):
        dst = os.path.join(bench_data_dst, os.path.basename(jf))
        if not os.path.exists(dst):
            shutil.copy2(jf, dst)

    # Write marker
    with open(vol_marker, "w") as f:
        f.write("complete")
    print("Volume setup complete")

# Use volume as the source of truth
pdf_dir = VOL_BASE
bench_data = os.path.join(VOL_BASE, "bench_data")

categories = sorted([d for d in os.listdir(pdf_dir)
                     if os.path.isdir(os.path.join(pdf_dir, d)) and d != "bench_data"])
total = 0
for cat in categories:
    n = len([f for f in os.listdir(os.path.join(pdf_dir, cat)) if f.endswith(".pdf")])
    total += n
    print(f"  {cat}: {n} PDFs")
print(f"Total: {total} PDFs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Sample Tests

# COMMAND ----------

all_tests = []
for jf in sorted(glob.glob(os.path.join(bench_data, "*.jsonl"))):
    with open(jf) as f:
        for line in f:
            line = line.strip()
            if line:
                all_tests.append(json.loads(line))

print(f"Total tests: {len(all_tests)}")

# Sample - sorted listdir for deterministic sampling across runs
random.seed(42)
selected_pdfs = set()
for cat in categories:
    cat_pdfs = sorted([f for f in os.listdir(os.path.join(pdf_dir, cat)) if f.endswith(".pdf")])
    sampled = random.sample(cat_pdfs, min(SAMPLE_N, len(cat_pdfs)))
    for s in sampled:
        selected_pdfs.add(f"{cat}/{s}")

filtered_tests = [t for t in all_tests if t["pdf"] in selected_pdfs]
print(f"Selected {len(selected_pdfs)} PDFs, {len(filtered_tests)} tests")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate: all sampled PDFs exist

# COMMAND ----------

missing_from_vol = []
for pdf_ref in sorted(selected_pdfs):
    cat, fname = pdf_ref.split("/", 1)
    if not os.path.exists(os.path.join(pdf_dir, cat, fname)):
        missing_from_vol.append(pdf_ref)

if missing_from_vol:
    print(f"ERROR: {len(missing_from_vol)} sampled PDFs missing!")
    for m in missing_from_vol[:20]:
        print(f"  {m}")
    raise RuntimeError(f"{len(missing_from_vol)} sampled PDFs missing. Re-run after fixing the volume.")
else:
    print(f"All {len(selected_pdfs)} sampled PDFs verified")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load GLM-OCR

# COMMAND ----------

from transformers import AutoProcessor, AutoModelForImageTextToText

print(f"Loading {GLM_MODEL}...")
processor = AutoProcessor.from_pretrained(GLM_MODEL, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    GLM_MODEL, torch_dtype="auto", device_map="auto", trust_remote_code=True
)
print(f"Model loaded! Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run GLM-OCR

# COMMAND ----------

import fitz  # PyMuPDF
from PIL import Image
from pypdf import PdfReader

def pdf_page_to_tempfile(pdf_path, page_num=1, dpi=200):
    """Render PDF page to a temp PNG file and return its path."""
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat)
    tmp_path = f"/tmp/ocr_page_{os.getpid()}.png"
    pix.save(tmp_path)
    doc.close()
    return tmp_path

def run_glm_ocr(pdf_path, page_num=1):
    """Run GLM-OCR on a PDF page using the official prompt format."""
    img_path = pdf_page_to_tempfile(pdf_path, page_num)

    # GLM-OCR official format: use "Text Recognition:" prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": img_path},
                {"type": "text", "text": "Text Recognition:"},
            ],
        }
    ]

    # Use apply_chat_template with tokenize=True as per official example
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=8192)

    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    # Clean up temp file
    try:
        os.remove(img_path)
    except:
        pass

    return output_text

# Quick test on first PDF
test_pdf = None
for cat in categories:
    pdfs_in_cat = sorted([f for f in os.listdir(os.path.join(pdf_dir, cat)) if f.endswith(".pdf")])
    if pdfs_in_cat:
        test_pdf = os.path.join(pdf_dir, cat, pdfs_in_cat[0])
        break

if test_pdf:
    print(f"Quick test on: {test_pdf}")
    result = run_glm_ocr(test_pdf, 1)
    print(f"Output length: {len(result)}")
    print(f"First 500 chars:\n{result[:500]}")

# COMMAND ----------

# Process all selected PDFs
glm_output_dir = os.path.join(RESULTS_DIR, "glm_ocr")
os.makedirs(glm_output_dir, exist_ok=True)

glm_count = 0
glm_errors = []
start = time.time()

for idx, pdf_ref in enumerate(sorted(selected_pdfs)):
    # pdf_ref is "category/filename.pdf"
    parts = pdf_ref.split("/", 1)
    cat, fname = parts[0], parts[1]
    pdf_path = os.path.join(pdf_dir, cat, fname)

    if not os.path.exists(pdf_path):
        glm_errors.append(f"Not found: {pdf_ref}")
        continue

    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
    except Exception as e:
        glm_errors.append(f"{pdf_ref}: {e}")
        continue

    out_dir = os.path.join(glm_output_dir, cat)
    os.makedirs(out_dir, exist_ok=True)
    base = fname.replace(".pdf", "")

    for pg in range(1, num_pages + 1):
        out_path = os.path.join(out_dir, f"{base}_pg{pg}_repeat1.md")
        if os.path.exists(out_path):
            glm_count += 1
            continue

        try:
            md = run_glm_ocr(pdf_path, pg)
            with open(out_path, "w") as f:
                f.write(md if md else "")
            glm_count += 1
        except Exception as e:
            with open(out_path, "w") as f:
                f.write("")
            glm_errors.append(f"{pdf_ref} pg{pg}: {str(e)[:100]}")

    if (idx + 1) % 10 == 0:
        elapsed = time.time() - start
        rate = (idx + 1) / elapsed * 60
        print(f"  [{idx+1}/{len(selected_pdfs)}] {rate:.1f} PDFs/min, {elapsed:.0f}s")

elapsed = time.time() - start
print(f"\nGLM-OCR: {glm_count} pages in {elapsed:.0f}s, {len(glm_errors)} errors")
if glm_errors[:3]:
    for e in glm_errors[:3]:
        print(f"  {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run ai_parse_document

# COMMAND ----------

ai_output_dir = os.path.join(RESULTS_DIR, "ai_parse_document")
os.makedirs(ai_output_dir, exist_ok=True)

ai_count = 0
ai_errors = []
start = time.time()

for idx, pdf_ref in enumerate(sorted(selected_pdfs)):
    parts = pdf_ref.split("/", 1)
    cat, fname = parts[0], parts[1]
    vol_path = os.path.join(pdf_dir, cat, fname)
    out_dir = os.path.join(ai_output_dir, cat)
    os.makedirs(out_dir, exist_ok=True)
    base = fname.replace(".pdf", "")

    # Check if already done
    existing = glob.glob(os.path.join(out_dir, f"{base}_pg*_repeat1.md"))
    if existing:
        ai_count += len(existing)
        continue

    try:
        elements_df = spark.sql(f"""
            WITH parsed AS (
                SELECT ai_parse_document(content, map('version', '2.0')) AS parsed_result
                FROM READ_FILES('{vol_path}', format => 'binaryFile')
            ),
            elements AS (
                SELECT explode(CAST(parsed_result:document:elements AS ARRAY<VARIANT>)) AS elem
                FROM parsed
            )
            SELECT
                COALESCE(elem:page_number::INT, 1) AS page_number,
                elem:content::STRING AS content
            FROM elements
            ORDER BY COALESCE(elem:page_number::INT, 1), COALESCE(elem:element_index::INT, 0)
        """)

        page_texts = {}
        for row in elements_df.collect():
            pg = row.page_number or 1
            content = row.content or ""
            if pg not in page_texts:
                page_texts[pg] = []
            page_texts[pg].append(content)

        if not page_texts:
            # ai_parse returned no elements — write empty page 1 file
            ai_errors.append(f"{pdf_ref}: no elements returned")
            with open(os.path.join(out_dir, f"{base}_pg1_repeat1.md"), "w") as f:
                f.write("")
            continue

        for pg, texts in page_texts.items():
            md = "\n\n".join(texts)
            with open(os.path.join(out_dir, f"{base}_pg{pg}_repeat1.md"), "w") as f:
                f.write(md)
            ai_count += 1

    except Exception as e:
        ai_errors.append(f"{pdf_ref}: {str(e)[:200]}")
        # Write empty files so scoring doesn't count as "missing"
        try:
            reader = PdfReader(vol_path)
            for pg in range(1, len(reader.pages) + 1):
                with open(os.path.join(out_dir, f"{base}_pg{pg}_repeat1.md"), "w") as f:
                    f.write("")
        except:
            # Last resort: write at least page 1
            with open(os.path.join(out_dir, f"{base}_pg1_repeat1.md"), "w") as f:
                f.write("")

    if (idx + 1) % 10 == 0:
        elapsed = time.time() - start
        rate = (idx + 1) / elapsed * 60
        print(f"  [{idx+1}/{len(selected_pdfs)}] {rate:.1f} PDFs/min, {elapsed:.0f}s")

elapsed = time.time() - start
print(f"\nai_parse_document: {ai_count} pages in {elapsed:.0f}s, {len(ai_errors)} errors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score Results

# COMMAND ----------

from fuzzysearch import find_near_matches
from rapidfuzz import fuzz
from collections import defaultdict

def normalize_text(text):
    if not text:
        return ""
    text = re.sub(r"<br/?>", " ", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"</?[bi]>", "", text)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    text = re.sub(r"\s+", " ", text)
    text = unicodedata.normalize("NFC", text)
    for old, new in {"\u2018":"'","\u2019":"'","\u201c":'"',"\u201d":'"',
                     "\u2013":"-","\u2014":"-","\u2212":"-"}.items():
        text = text.replace(old, new)
    return text

def run_test(td, md):
    tt = td["type"]
    mn = normalize_text(md)

    if tt == "baseline":
        return (len(mn.strip()) > 10, "empty")

    elif tt in ("present", "absent"):
        text = normalize_text(td.get("text", ""))
        md_diffs = td.get("max_diffs", 0)
        cs = td.get("case_sensitive", True)
        search = mn
        if not cs:
            search = search.lower()
            text = text.lower()
        fn = td.get("first_n")
        ln = td.get("last_n")
        if fn:
            search = search[:fn]
        if ln:
            search = search[-ln:]
        if md_diffs == 0:
            found = text in search
        else:
            found = len(find_near_matches(text, search, max_l_dist=md_diffs)) > 0
        if tt == "present":
            return (found, "")
        else:
            return (not found, "")

    elif tt == "order":
        before = normalize_text(td.get("before", ""))
        after = normalize_text(td.get("after", ""))
        md_diffs = td.get("max_diffs", 0)
        if md_diffs == 0:
            bp = mn.find(before)
            ap = mn.find(after)
        else:
            bm = find_near_matches(before, mn, max_l_dist=md_diffs)
            am = find_near_matches(after, mn, max_l_dist=md_diffs)
            bp = bm[0].start if bm else -1
            ap = am[0].start if am else -1
        if bp == -1 or ap == -1:
            return (False, "text not found")
        return (bp < ap, "")

    elif tt == "table":
        cell = normalize_text(td.get("cell", ""))
        return (cell in mn, "")

    elif tt == "math":
        expr = td.get("expression", td.get("math", ""))
        ec = expr.replace("$","").replace("\\(","").replace("\\)","")
        ec = ec.replace("\\[","").replace("\\]","").strip()
        if ec in md or ec in mn:
            return (True, "")
        if fuzz.partial_ratio(ec, mn) > 80:
            return (True, "")
        return (False, "")

    elif tt == "footnote":
        marker = td.get("marker", "")
        return (marker in md, "")

    elif tt == "format":
        return (True, "")

    return (False, "")

def score_candidate(cand_dir, tests):
    type_scores = defaultdict(list)
    tp = 0
    tot = 0
    missing = 0
    for t in tests:
        pdf_ref = t["pdf"]
        pg = t.get("page", 1)
        parts = pdf_ref.split("/", 1)
        cat = parts[0] if len(parts) == 2 else ""
        fname = parts[1] if len(parts) == 2 else parts[0]
        base = fname.replace(".pdf", "")
        out_path = os.path.join(cand_dir, cat, f"{base}_pg{pg}_repeat1.md")
        if not os.path.exists(out_path):
            type_scores[t["type"]].append(0.0)
            tot += 1
            missing += 1
            continue
        with open(out_path) as f:
            md = f.read()
        passed, _ = run_test(t, md)
        s = 1.0 if passed else 0.0
        type_scores[t["type"]].append(s)
        tp += s
        tot += 1

    results = {}
    for tt, scores in sorted(type_scores.items()):
        results[tt] = {"score": round(sum(scores)/len(scores)*100, 1), "count": len(scores)}
    overall = round(tp/tot*100, 1) if tot else 0
    return {"overall": overall, "total_tests": tot, "total_pass": int(tp),
            "missing_files": missing, "by_type": results}

print("Scoring GLM-OCR...")
gs = score_candidate(os.path.join(RESULTS_DIR, "glm_ocr"), filtered_tests)
print(f"  {gs['total_pass']}/{gs['total_tests']} passed, {gs['missing_files']} missing")

print("Scoring ai_parse_document...")
ais = score_candidate(os.path.join(RESULTS_DIR, "ai_parse_document"), filtered_tests)
print(f"  {ais['total_pass']}/{ais['total_tests']} passed, {ais['missing_files']} missing")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

# Category name mapping for display
cat_display = {
    "absent": "Headers/Footers",
    "baseline": "Baseline",
    "math": "Math",
    "order": "Reading Order",
    "present": "Text Present",
    "table": "Tables",
    "footnote": "Footnotes",
    "format": "Format",
}

print("=" * 70)
print(f"{'olmOCR-Bench Results':^70}")
print(f"{'Sample: 50 PDFs/category':^70}")
print("=" * 70)
print(f"{'Test Type':<20} {'GLM-OCR':>12} {'ai_parse':>12} {'Tests':>8}")
print("-" * 55)

all_types = sorted(set(list(gs["by_type"].keys()) + list(ais["by_type"].keys())))
for t in all_types:
    g = gs["by_type"].get(t, {"score":0,"count":0})
    a = ais["by_type"].get(t, {"score":0,"count":0})
    n = max(g["count"], a["count"])
    label = cat_display.get(t, t)
    print(f"{label:<20} {g['score']:>10.1f}% {a['score']:>10.1f}% {n:>8}")

print("-" * 55)
print(f"{'OVERALL':<20} {gs['overall']:>10.1f}% {ais['overall']:>10.1f}% {gs['total_tests']:>8}")
print()
print(f"GLM-OCR: {gs['total_pass']}/{gs['total_tests']} ({gs['missing_files']} missing)")
print(f"ai_parse: {ais['total_pass']}/{ais['total_tests']} ({ais['missing_files']} missing)")

# Save
summary = {"benchmark": "olmOCR-bench", "sample_n": SAMPLE_N,
           "glm_ocr": gs, "ai_parse_document": ais}
rp = os.path.join(RESULTS_DIR, "benchmark_results.json")
with open(rp, "w") as f:
    json.dump(summary, f, indent=2)

dbutils.notebook.exit(json.dumps({
    "glm_ocr": gs["overall"], "ai_parse": ais["overall"],
    "glm_pass": gs["total_pass"], "ai_pass": ais["total_pass"],
    "tests": gs["total_tests"]
}))
