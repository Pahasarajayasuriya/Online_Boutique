# Online Boutique Pipeline Run Guide

Running the GraphCodeBERT semantic clone detection pipeline against  
**Google's Online Boutique** (`GoogleCloudPlatform/microservices-demo`).

---

## What you'll get

| Service | Language | Functions expected |
|---------|----------|--------------------|
| `adservice` | Java | ~15–25 |
| `emailservice` | Python | ~5–10 |
| `recommendationservice` | Python | ~5–10 |
| `currencyservice` | JavaScript (Node.js) | ~5–10 |
| `paymentservice` | JavaScript (Node.js) | ~5–10 |

**Total: ~40–60 functions, ~800–1500 cross-service pairs**  
Runtime: ~5–15 minutes (CPU), ~1–3 minutes (GPU)

---

## Step 1 — Place all scripts in one folder

Put these three files in the **same directory** (e.g. `~/online-boutique-pipeline/`):

```
extract_functions_and_embed.py   ← updated pipeline (v2, with CLI args)
plot_pr_curve.py                 ← PR curve generator
run_online_boutique.sh           ← one-shot setup + run script
```

---

## Step 2 — Install dependencies

```bash
pip install transformers torch tree-sitter==0.20.4 tree-sitter-languages matplotlib
```

> If you already ran the pipeline on the synthetic corpus, these are already installed.

---

## Step 3 — Run

```bash
cd ~/online-boutique-pipeline/
chmod +x run_online_boutique.sh
./run_online_boutique.sh
```

The script will:
1. Clone Online Boutique into `./microservices-demo/` (shallow, ~30 seconds)
2. Copy the 5 supported services into `./ob_workspace/`
3. Run the pipeline — outputs go to `./ob_run/`
4. Run `plot_pr_curve.py` on the result

---

## Step 4 — Verify the workspace (optional sanity check)

After the clone step, your workspace should look like this:

```
ob_workspace/
├── adservice/
│   └── *.java
├── emailservice/
│   ├── email_server.py
│   └── ...
├── recommendationservice/
│   ├── recommendation_server.py
│   └── ...
├── currencyservice/
│   ├── server.js
│   └── ...
└── paymentservice/
    ├── index.js
    └── ...
```

If any folder is missing, the shell script will warn you. You can also clone
manually and copy the folders yourself.

---

## Step 5 — Manual run (if you prefer not to use the shell script)

```bash
# 1. Clone
git clone --depth 1 https://github.com/GoogleCloudPlatform/microservices-demo.git

# 2. Build workspace manually
mkdir -p ob_workspace
cp -r microservices-demo/src/adservice/src/main/java/hipstershop  ob_workspace/adservice
cp -r microservices-demo/src/emailservice                          ob_workspace/emailservice
cp -r microservices-demo/src/recommendationservice                 ob_workspace/recommendationservice
cp -r microservices-demo/src/currencyservice                       ob_workspace/currencyservice
cp -r microservices-demo/src/paymentservice                        ob_workspace/paymentservice

# 3. Run pipeline
mkdir -p ob_run
python3 extract_functions_and_embed.py \
    --services-root ob_workspace \
    --output-dir    ob_run/pipeline_output \
    --embeddings    ob_run/embeddings.pkl \
    --report        ob_run/clone_report.json \
    --threshold     0.85

# 4. Generate figures
python3 plot_pr_curve.py ob_run/clone_report.json
```

> **Note on threshold:** Use `--threshold 0.85` for Online Boutique.  
> The synthetic corpus used 0.90, but real-world code has more variation —  
> 0.85 is a better starting point. You can always lower to 0.80 to see more candidates.

---

## What to look for in the results

### Expected findings (real-world clones that likely exist)

Online Boutique is a realistic polyglot system built by Google engineers.
Known patterns that your pipeline should surface:

| Pattern | Where to look | Why it's likely |
|---------|--------------|-----------------|
| Money/currency arithmetic | `currencyservice` (JS) ↔ `paymentservice` (JS) | Both handle monetary amounts |
| Input validation | `paymentservice` ↔ `recommendationservice` | Standard field-checking boilerplate |
| Error handling / retry | `adservice` (Java) ↔ any service | gRPC retry patterns are shared |
| List pagination / slicing | `recommendationservice` ↔ `currencyservice` | Both return paginated/truncated lists |

### How to interpret the report

```
clone_report.json → "clone_pairs" array
```

Each entry has:
- `score` — cosine similarity (0–1). Above threshold = flagged as clone.
- `function_1`, `function_2` — `service/file::functionName`
- `lang_1`, `lang_2` — language pair (e.g. `java` ↔ `javascript`)
- `confidence` — `HIGH` (≥0.95) or `MEDIUM` (≥threshold)

Cross-language pairs (`java ↔ javascript`, `python ↔ java`) are the most
interesting — these are the hardest for syntactic tools to find and the
strongest evidence for your research contribution.

### Red flags (false positives to explain in the paper)

- Same-language pairs with very high scores (≥0.95): likely structural similarity
  (both are simple getter-style functions or trivial wrappers)
- `adservice` Java pairs: adservice has many protobuf-generated boilerplate
  methods that will cluster together — these are worth filtering or noting

---

## Troubleshooting

**`adservice` source path not found**  
The Java source lives at `src/adservice/src/main/java/hipstershop/`.  
Check this path exists in your clone: `ls microservices-demo/src/adservice/src/`

**Too few functions extracted**  
Some services have thin source files (only 1–2 functions). This is fine —
Online Boutique is a demo, not an enterprise system. Report the actual counts.

**Pipeline runs out of memory**  
GraphCodeBERT is ~500MB. On machines with <8GB RAM, use:
```bash
# Process one service at a time by temporarily removing others from ob_workspace
```

**Slow runtime on CPU**  
Expected. Each function takes ~2–5 seconds on CPU.  
For ~50 functions: 2–4 minutes. This is fine for a research run.

---

## After the run — what to write in Section 6.3

Your paper needs to report:

1. **Corpus statistics** — how many services, files, functions extracted
2. **Clone candidates found** — count at threshold 0.85, broken down by language pair
3. **Qualitative analysis** — pick 2–3 flagged pairs, show the code side-by-side,
   explain whether they are genuine semantic clones or false positives
4. **Comparison to synthetic corpus** — does precision/recall improve or worsen on
   real-world code? (Expected: slightly worse — real code is noisier)

Template sentence for the paper:
> *"To evaluate the tool on a real-world system, we applied the pipeline to the
> Online Boutique microservices-demo (Google, 2023), a polyglot application
> comprising 11 services. Of the 5 services implemented in languages supported
> by our pipeline (Java, Python, JavaScript), we extracted N functions across
> M cross-service pairs. At threshold τ=0.85, the pipeline identified K candidate
> clone pairs, of which X were confirmed as semantic clones upon manual inspection."*