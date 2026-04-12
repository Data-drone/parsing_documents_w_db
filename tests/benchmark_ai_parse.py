# Databricks notebook source
# MAGIC %md
# MAGIC # olmOCR-Bench: ai_parse_document only
# MAGIC Runs in parallel with GLM-OCR on GPU cluster.
# MAGIC
# MAGIC PDFs are stored persistently in a UC volume to avoid re-downloading.

# COMMAND ----------

dbutils.widgets.text("sample_n", "50", "Sample N PDFs per category (0=all)")
SAMPLE_N = int(dbutils.widgets.get("sample_n"))
BENCH_DIR = "/tmp/olmocr_bench"
RESULTS_DIR = "/tmp/olmocr_results"
VOL_BASE = "/Volumes/main/default/olmocr_bench_pdfs"
print(f"Config: sample_n={SAMPLE_N}")

# COMMAND ----------

# MAGIC %pip install -q pypdf huggingface_hub fuzzysearch rapidfuzz

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, json, glob, random, time, re, unicodedata, shutil
SAMPLE_N = int(dbutils.widgets.get("sample_n"))
BENCH_DIR = "/tmp/olmocr_bench"
RESULTS_DIR = "/tmp/olmocr_results"
VOL_BASE = "/Volumes/main/default/olmocr_bench_pdfs"

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
# MAGIC ## Sample Tests (same seed as GLM-OCR notebook)

# COMMAND ----------

all_tests = []
for jf in sorted(glob.glob(os.path.join(bench_data, "*.jsonl"))):
    with open(jf) as f:
        for line in f:
            line = line.strip()
            if line:
                all_tests.append(json.loads(line))

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
# MAGIC ## Validate: all sampled PDFs exist in volume

# COMMAND ----------

missing_from_vol = []
for pdf_ref in sorted(selected_pdfs):
    cat, fname = pdf_ref.split("/", 1)
    vol_path = os.path.join(pdf_dir, cat, fname)
    if not os.path.exists(vol_path):
        missing_from_vol.append(pdf_ref)

if missing_from_vol:
    print(f"ERROR: {len(missing_from_vol)} sampled PDFs missing from volume!")
    for m in missing_from_vol[:20]:
        print(f"  {m}")
    raise RuntimeError(f"{len(missing_from_vol)} sampled PDFs missing from volume. Re-run after fixing the volume.")
else:
    print(f"All {len(selected_pdfs)} sampled PDFs verified in volume")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run ai_parse_document

# COMMAND ----------

from pypdf import PdfReader

ai_output_dir = os.path.join(RESULTS_DIR, "ai_parse_document")
os.makedirs(ai_output_dir, exist_ok=True)

ai_count = 0
ai_errors = []
start = time.time()

for idx, pdf_ref in enumerate(sorted(selected_pdfs)):
    cat, fname = pdf_ref.split("/", 1)
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
            ai_errors.append(f"{pdf_ref}: no elements returned")

        # Always use PdfReader to get total page count, then write a file
        # for every page — empty if ai_parse didn't return elements for it.
        try:
            reader = PdfReader(vol_path)
            total_pages = len(reader.pages)
        except Exception:
            total_pages = max(page_texts.keys()) if page_texts else 1

        for pg in range(1, total_pages + 1):
            if pg in page_texts:
                md = "\n\n".join(page_texts[pg])
            else:
                md = ""
            with open(os.path.join(out_dir, f"{base}_pg{pg}_repeat1.md"), "w") as f:
                f.write(md)
            ai_count += 1

    except Exception as e:
        ai_errors.append(f"{pdf_ref}: {str(e)[:200]}")
        # Write empty files for all pages so scoring doesn't count as "missing"
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
if ai_errors:
    for e in ai_errors[:10]:
        print(f"  {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score ai_parse_document

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
        return (len(mn.strip()) > 10, "")
    elif tt in ("present", "absent"):
        text = normalize_text(td.get("text", ""))
        md_diffs = td.get("max_diffs", 0)
        cs = td.get("case_sensitive", True)
        search = mn
        if not cs:
            search = search.lower()
            text = text.lower()
        if td.get("first_n"):
            search = search[:td["first_n"]]
        if td.get("last_n"):
            search = search[-td["last_n"]:]
        if md_diffs == 0:
            found = text in search
        else:
            found = len(find_near_matches(text, search, max_l_dist=md_diffs)) > 0
        return (found if tt == "present" else not found, "")
    elif tt == "order":
        before = normalize_text(td.get("before", ""))
        after = normalize_text(td.get("after", ""))
        md_diffs = td.get("max_diffs", 0)
        if md_diffs == 0:
            bp, ap = mn.find(before), mn.find(after)
        else:
            bm = find_near_matches(before, mn, max_l_dist=md_diffs)
            am = find_near_matches(after, mn, max_l_dist=md_diffs)
            bp = bm[0].start if bm else -1
            ap = am[0].start if am else -1
        if bp == -1 or ap == -1:
            return (False, "")
        return (bp < ap, "")
    elif tt == "table":
        cell = normalize_text(td.get("cell", ""))
        return (cell in mn, "")
    elif tt == "math":
        expr = td.get("expression", td.get("math", ""))
        ec = expr.replace("$","").replace("\\(","").replace("\\)","").replace("\\[","").replace("\\]","").strip()
        if ec in md or ec in mn:
            return (True, "")
        return (fuzz.partial_ratio(ec, mn) > 80, "")
    elif tt == "footnote":
        return (td.get("marker", "") in md, "")
    elif tt == "format":
        return (True, "")
    return (False, "")

def score_candidate(cand_dir, tests):
    type_scores = defaultdict(list)
    cat_scores = defaultdict(list)
    tp, tot, missing = 0, 0, 0
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
            cat_scores[cat].append(0.0)
            tot += 1
            missing += 1
            continue
        with open(out_path) as f:
            md = f.read()
        passed, _ = run_test(t, md)
        s = 1.0 if passed else 0.0
        type_scores[t["type"]].append(s)
        cat_scores[cat].append(s)
        tp += s
        tot += 1
    by_type = {}
    for tt, scores in sorted(type_scores.items()):
        by_type[tt] = {"score": round(sum(scores)/len(scores)*100, 1), "count": len(scores)}
    by_category = {}
    for cc, scores in sorted(cat_scores.items()):
        by_category[cc] = {"score": round(sum(scores)/len(scores)*100, 1), "count": len(scores)}
    overall = round(tp/tot*100, 1) if tot else 0
    return {"overall": overall, "total_tests": tot, "total_pass": int(tp), "missing_files": missing, "by_type": by_type, "by_category": by_category}

print("Scoring ai_parse_document...")
ais = score_candidate(ai_output_dir, filtered_tests)
print(f"  Overall: {ais['overall']}%")
print(f"  {ais['total_pass']}/{ais['total_tests']} passed, {ais['missing_files']} missing")

print("\n  By test type:")
for tt, data in sorted(ais["by_type"].items()):
    print(f"    {tt}: {data['score']}% ({data['count']} tests)")

print("\n  By PDF category:")
for cc, data in sorted(ais["by_category"].items()):
    print(f"    {cc}: {data['score']}% ({data['count']} tests)")

# Save
with open(os.path.join(RESULTS_DIR, "ai_parse_results.json"), "w") as f:
    json.dump(ais, f, indent=2)

dbutils.notebook.exit(json.dumps({"ai_parse": ais["overall"], "pass": ais["total_pass"],
                                   "tests": ais["total_tests"], "missing": ais["missing_files"],
                                   "by_category": {c: d["score"] for c, d in ais["by_category"].items()}}))
