# Running with Git LFS — Contributor Guide

This repository uses **Git LFS (Large File Storage)** to manage model weights and large binary results. If you are pulling this repository for the first time, follow these steps to ensure the model weights are correctly downloaded and the pipeline runs without errors.

---

## 1. Install Git LFS

You must have Git LFS installed on your system before cloning or pulling.

- **Ubuntu/Debian**: `sudo apt install git-lfs`
- **MacOS**: `brew install git-lfs`
- **Windows**: Download from [git-lfs.github.com](https://git-lfs.github.com)

Once installed, initialize LFS globally:
```bash
git lfs install
```

---

## 2. Pull the Repository

When you clone the repo, Git LFS should automatically download the large files. If you have already cloned it and find small "pointer" files instead of the actual weights, run:

```bash
git lfs pull
```

---

## 3. Set Up the Python Environment

The pipeline requires specific versions of `transformers` and `peft` to avoid architecture and tensor dimension errors.

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install PINNED dependencies (Critical)
pip install "transformers==4.40.2" "peft==0.10.0" \
            torch tree-sitter==0.20.4 tree-sitter-languages matplotlib
```

> [!IMPORTANT]
> **Why pinned versions?**
> - `transformers 4.40.2` is required to avoid attention mask dimension errors (introduced in 4.41.0+ for GraphCodeBERT).
> - `peft 0.10.0` is required for compatibility with transformers 4.40.2.

---

## 4. Run the Pipeline

### Step 1: Prep the Online Boutique workspace
If you haven't already, clone the microservices demo and copy the source services:

```bash
# Clone the target repo
git clone --depth 1 https://github.com/GoogleCloudPlatform/microservices-demo.git

# Prepare the workspace folder
mkdir -p ob_workspace
cp -r microservices-demo/src/adservice/src/main/java/hipstershop  ob_workspace/adservice
cp -r microservices-demo/src/emailservice                          ob_workspace/emailservice
cp -r microservices-demo/src/recommendationservice                 ob_workspace/recommendationservice
cp -r microservices-demo/src/currencyservice                       ob_workspace/currencyservice
cp -r microservices-demo/src/paymentservice                        ob_workspace/paymentservice
```

### Step 2: Run Extraction & Embedding
```bash
mkdir -p ob_run
python3 extract_functions_and_embed.py \
    --services-root ob_workspace \
    --output-dir    ob_run/pipeline_output \
    --embeddings    ob_run/embeddings.pkl \
    --report        ob_run/clone_report.json \
    --threshold     0.85
```

### Step 3: Generate Plots
```bash
python3 plot_pr_curve.py ob_run/clone_report.json
```

---

## 5. Troubleshooting

**"ModuleNotFoundError: No module named 'peft'"**
Ensure you have activated your `venv` or use the full path: `./venv/bin/python3 ...`

**"Traceback ... numbers of sizes provided (4) must be greater or equal ..."**
This means you are using a `transformers` version > 4.40.2. Uninstall and reinstall the pinned version:
`pip install "transformers==4.40.2"`
