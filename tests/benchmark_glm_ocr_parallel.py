# Databricks notebook source
# MAGIC %md
# MAGIC # GLM-OCR Benchmark (Parallel, Persistent)
# MAGIC
# MAGIC Runs GLM-OCR across multiple GPUs in parallel using Spark.
# MAGIC Results are persisted to a UC volume immediately — safe to cancel/restart.
# MAGIC
# MAGIC **Cluster requirements:** Multi-node GPU cluster (e.g. 4x A10G or similar).
# MAGIC Each worker loads GLM-OCR onto its GPU independently.

# COMMAND ----------

dbutils.widgets.text("sample_n", "50", "Sample N PDFs per category (0=all)")
dbutils.widgets.text("partitions", "0", "Num partitions (0=auto, matches num GPUs)")
SAMPLE_N = int(dbutils.widgets.get("sample_n"))
NUM_PARTITIONS = int(dbutils.widgets.get("partitions"))

VOL_BASE = "/Volumes/main/default/olmocr_bench_pdfs"
RESULTS_VOL = "/Volumes/main/default/olmocr_bench_results"
GLM_MODEL = "zai-org/GLM-OCR"

print(f"Config: sample_n={SAMPLE_N}, partitions={NUM_PARTITIONS}, model={GLM_MODEL}")

# COMMAND ----------

# MAGIC %pip install -q "transformers>=5.1.0" fuzzysearch rapidfuzz pypdf huggingface_hub Pillow PyMuPDF

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, json, glob, random, time, re, unicodedata, shutil
SAMPLE_N = int(dbutils.widgets.get("sample_n"))
NUM_PARTITIONS = int(dbutils.widgets.get("partitions"))
VOL_BASE = "/Volumes/main/default/olmocr_bench_pdfs"
RESULTS_VOL = "/Volumes/main/default/olmocr_bench_results"
GLM_MODEL = "zai-org/GLM-OCR"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Results Volume

# COMMAND ----------

try:
    spark.sql("CREATE VOLUME IF NOT EXISTS main.default.olmocr_bench_results")
except Exception as e:
    print(f"Volume setup: {e}")

glm_results_dir = os.path.join(RESULTS_VOL, "glm_ocr")
os.makedirs(glm_results_dir, exist_ok=True)
print(f"Results will persist to: {glm_results_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure Benchmark Data in UC Volume

# COMMAND ----------

from huggingface_hub import snapshot_download

try:
    spark.sql("CREATE VOLUME IF NOT EXISTS main.default.olmocr_bench_pdfs")
except Exception as e:
    print(f"Volume setup: {e}")

vol_marker = os.path.join(VOL_BASE, ".olmocr_bench_complete")
if os.path.exists(vol_marker):
    print("Dataset already in volume (skipping download)")
else:
    print("Downloading olmOCR-bench dataset...")
    bench_dir = "/tmp/olmocr_bench"
    snapshot_download(repo_id="allenai/olmOCR-bench", repo_type="dataset",
                      local_dir=bench_dir, ignore_patterns=["*.DS_Store"])
    pdf_src = os.path.join(bench_dir, "bench_data", "pdfs")
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

    bench_data_src = os.path.join(bench_dir, "bench_data")
    bench_data_dst = os.path.join(VOL_BASE, "bench_data")
    os.makedirs(bench_data_dst, exist_ok=True)
    for jf in glob.glob(os.path.join(bench_data_src, "*.jsonl")):
        dst = os.path.join(bench_data_dst, os.path.basename(jf))
        if not os.path.exists(dst):
            shutil.copy2(jf, dst)

    with open(vol_marker, "w") as f:
        f.write("complete")
    print("Volume setup complete")

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
# MAGIC ## Sample Tests & Build Work List

# COMMAND ----------

all_tests = []
for jf in sorted(glob.glob(os.path.join(bench_data, "*.jsonl"))):
    with open(jf) as f:
        for line in f:
            line = line.strip()
            if line:
                all_tests.append(json.loads(line))

print(f"Total tests: {len(all_tests)}")

# Deterministic sampling (same seed as other benchmarks)
random.seed(42)
selected_pdfs = set()
for cat in categories:
    cat_pdfs = sorted([f for f in os.listdir(os.path.join(pdf_dir, cat)) if f.endswith(".pdf")])
    if SAMPLE_N > 0:
        sampled = random.sample(cat_pdfs, min(SAMPLE_N, len(cat_pdfs)))
    else:
        sampled = cat_pdfs
    for s in sampled:
        selected_pdfs.add(f"{cat}/{s}")

filtered_tests = [t for t in all_tests if t["pdf"] in selected_pdfs]
print(f"Selected {len(selected_pdfs)} PDFs, {len(filtered_tests)} tests")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Page-Level Work Items (skip already done)

# COMMAND ----------

from pypdf import PdfReader

work_items = []  # list of (pdf_ref, page_num, pdf_path, out_path)
skipped = 0

for pdf_ref in sorted(selected_pdfs):
    cat, fname = pdf_ref.split("/", 1)
    pdf_path = os.path.join(pdf_dir, cat, fname)
    base = fname.replace(".pdf", "")
    out_dir = os.path.join(glm_results_dir, cat)

    if not os.path.exists(pdf_path):
        print(f"WARNING: missing {pdf_ref}")
        continue

    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
    except Exception as e:
        print(f"WARNING: can't read {pdf_ref}: {e}")
        continue

    for pg in range(1, num_pages + 1):
        out_path = os.path.join(out_dir, f"{base}_pg{pg}_repeat1.md")
        if os.path.exists(out_path):
            skipped += 1
        else:
            work_items.append({
                "pdf_ref": pdf_ref,
                "page_num": pg,
                "pdf_path": pdf_path,
                "out_path": out_path
            })

print(f"Work items: {len(work_items)} pages to process, {skipped} already done")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick Test: Verify GLM-OCR loads on driver

# COMMAND ----------

import torch
print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Driver GPU: {torch.cuda.get_device_name(0)}")

# Quick model load test on driver (will be repeated on each worker)
from transformers import AutoProcessor, AutoModelForImageTextToText
print(f"Testing model load for {GLM_MODEL}...")
_test_proc = AutoProcessor.from_pretrained(GLM_MODEL, trust_remote_code=True)
_test_model = AutoModelForImageTextToText.from_pretrained(
    GLM_MODEL, torch_dtype="auto", device_map="auto", trust_remote_code=True
)
print(f"Model loaded on driver! VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# Clean up driver model — workers will load their own
del _test_model, _test_proc
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parallel Processing with Spark
# MAGIC
# MAGIC Distributes page-level work items across GPU workers.
# MAGIC Each worker loads the model once and processes its partition.
# MAGIC Results are written to the UC volume immediately.

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, BooleanType

# Create Spark DataFrame of work items
work_df = spark.createDataFrame(work_items)

# Auto-detect number of GPUs if not specified
if NUM_PARTITIONS <= 0:
    try:
        sc = spark.sparkContext
        num_executors = max(1, len(sc._jsc.sc().getExecutorMemoryStatus()) - 1)
        NUM_PARTITIONS = num_executors
    except Exception:
        NUM_PARTITIONS = 4  # fallback
print(f"Using {NUM_PARTITIONS} partitions (1 per GPU worker)")

# Repartition so each worker gets a roughly equal share
work_df = work_df.repartition(NUM_PARTITIONS)

print(f"Distributing {work_df.count()} pages across {NUM_PARTITIONS} workers")

# COMMAND ----------

result_schema = StructType([
    StructField("pdf_ref", StringType()),
    StructField("page_num", IntegerType()),
    StructField("out_path", StringType()),
    StructField("output_len", IntegerType()),
    StructField("success", BooleanType()),
    StructField("error", StringType()),
    StructField("elapsed_s", FloatType()),
])

def process_partition(iterator):
    """
    Each Spark worker calls this once per partition.
    Loads GLM-OCR on the local GPU, then processes all pages in the partition.
    Writes each result to UC volume immediately.
    """
    import os, time, torch, fitz
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from pypdf import PdfReader

    model_name = "zai-org/GLM-OCR"

    # Load model on this worker's GPU
    proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )

    def ocr_page(pdf_path, page_num):
        """Render page to image and run GLM-OCR."""
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        mat = fitz.Matrix(200/72, 200/72)  # 200 DPI
        pix = page.get_pixmap(matrix=mat)
        tmp_path = f"/tmp/ocr_{os.getpid()}_{page_num}.png"
        pix.save(tmp_path)
        doc.close()

        messages = [{"role": "user", "content": [
            {"type": "image", "url": tmp_path},
            {"type": "text", "text": "Text Recognition:"},
        ]}]

        inputs = proc.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(mdl.device)
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            gen_ids = mdl.generate(**inputs, max_new_tokens=8192)
        text = proc.decode(gen_ids[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True)
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return text

    results = []
    for row in iterator:
        pdf_ref = row["pdf_ref"]
        page_num = row["page_num"]
        pdf_path = row["pdf_path"]
        out_path = row["out_path"]

        # Skip if already exists (race condition guard)
        if os.path.exists(out_path):
            results.append({
                "pdf_ref": pdf_ref, "page_num": page_num,
                "out_path": out_path, "output_len": -1,
                "success": True, "error": "already_done", "elapsed_s": 0.0
            })
            continue

        t0 = time.time()
        try:
            md_text = ocr_page(pdf_path, page_num)
            # Persist to UC volume immediately
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                f.write(md_text if md_text else "")
            elapsed = time.time() - t0
            results.append({
                "pdf_ref": pdf_ref, "page_num": page_num,
                "out_path": out_path, "output_len": len(md_text or ""),
                "success": True, "error": "", "elapsed_s": round(elapsed, 2)
            })
        except Exception as e:
            elapsed = time.time() - t0
            # Write empty file so scoring doesn't count as "missing"
            try:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w") as f:
                    f.write("")
            except OSError:
                pass
            results.append({
                "pdf_ref": pdf_ref, "page_num": page_num,
                "out_path": out_path, "output_len": 0,
                "success": False, "error": str(e)[:200], "elapsed_s": round(elapsed, 2)
            })

    # Yield all results at end of partition
    import pandas as pd
    yield pd.DataFrame(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute Parallel OCR

# COMMAND ----------

print(f"Starting parallel GLM-OCR processing...")
start_time = time.time()

results_df = work_df.mapInPandas(process_partition, schema=result_schema)

# Force execution and collect stats
results_pdf = results_df.toPandas()

total_elapsed = time.time() - start_time
success_count = results_pdf["success"].sum()
error_count = (~results_pdf["success"]).sum()
total_chars = results_pdf.loc[results_pdf["success"], "output_len"].sum()
avg_time = results_pdf.loc[results_pdf["success"] & (results_pdf["elapsed_s"] > 0), "elapsed_s"].mean()

print(f"\nCompleted in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
print(f"  Success: {success_count}, Errors: {error_count}")
print(f"  Total chars output: {total_chars:,}")
print(f"  Avg time per page: {avg_time:.1f}s")
print(f"  Pages already done: {(results_pdf['error'] == 'already_done').sum()}")

if error_count > 0:
    print(f"\nErrors:")
    for _, row in results_pdf[~results_pdf["success"]].head(10).iterrows():
        print(f"  {row['pdf_ref']} pg{row['page_num']}: {row['error']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Persisted Results

# COMMAND ----------

persisted = 0
for cat in categories:
    cat_dir = os.path.join(glm_results_dir, cat)
    if os.path.isdir(cat_dir):
        n = len([f for f in os.listdir(cat_dir) if f.endswith(".md")])
        persisted += n
        print(f"  {cat}: {n} pages")

print(f"\nTotal persisted: {persisted} pages in {glm_results_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score GLM-OCR Results

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
    return {"overall": overall, "total_tests": tot, "total_pass": int(tp),
            "missing_files": missing, "by_type": by_type, "by_category": by_category}

print("Scoring GLM-OCR from UC volume...")
gs = score_candidate(glm_results_dir, filtered_tests)

print(f"\n{'='*60}")
print(f"{'GLM-OCR olmOCR-Bench Results':^60}")
print(f"{'='*60}")
print(f"Overall: {gs['overall']}%  ({gs['total_pass']}/{gs['total_tests']}, {gs['missing_files']} missing)")
print(f"\nBy test type:")
for tt, data in sorted(gs["by_type"].items()):
    print(f"  {tt}: {data['score']}% ({data['count']} tests)")
print(f"\nBy PDF category:")
for cc, data in sorted(gs["by_category"].items()):
    print(f"  {cc}: {data['score']}% ({data['count']} tests)")

# Save results JSON to volume
results_json = os.path.join(RESULTS_VOL, "glm_ocr_results.json")
with open(results_json, "w") as f:
    json.dump(gs, f, indent=2)
print(f"\nResults saved to {results_json}")

dbutils.notebook.exit(json.dumps({
    "glm_ocr": gs["overall"],
    "pass": gs["total_pass"],
    "tests": gs["total_tests"],
    "missing": gs["missing_files"],
    "by_category": {c: d["score"] for c, d in gs["by_category"].items()}
}))
