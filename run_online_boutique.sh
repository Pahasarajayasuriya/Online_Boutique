#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  run_online_boutique.sh
#  ─────────────────────
#  One-shot setup + run script for the GraphCodeBERT clone detection pipeline
#  on Google's Online Boutique microservices-demo repository.
#
#  What this script does:
#    1. Clones the Online Boutique repo (shallow, no history)
#    2. Builds a flat workspace with one folder per supported service
#       (Python / JavaScript / Java only — Go and C# are skipped)
#    3. Runs extract_functions_and_embed.py against that workspace
#    4. Runs plot_pr_curve.py on the resulting clone_report.json
#
#  Supported services extracted:
#    adservice          → Java
#    emailservice       → Python
#    recommendationservice → Python
#    currencyservice    → JavaScript (Node.js)
#    paymentservice     → JavaScript (Node.js)
#
#  Skipped (unsupported languages):
#    frontend, cartservice, checkoutservice,
#    productcatalogservice, shippingservice  → Go / C#
#
#  Usage:
#    chmod +x run_online_boutique.sh
#    ./run_online_boutique.sh
#
#  Output (written to ./ob_run/):
#    clone_report.json          ← main result
#    embeddings.pkl             ← serialised embeddings (reusable)
#    pipeline_output/           ← per-function inspection JSONs
#    fig_combined.png           ← PR curve + distribution figure
#    fig1_pr_curve.png
#    fig2_score_distribution.png
#    fig3_f1_vs_threshold.png
#    pipeline.log               ← full stdout log
#
#  Requirements:
#    pip install transformers torch tree-sitter==0.20.4 tree-sitter-languages matplotlib
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="$SCRIPT_DIR/extract_functions_and_embed.py"
PR_SCRIPT="$SCRIPT_DIR/plot_pr_curve.py"
REPO_DIR="$SCRIPT_DIR/microservices-demo"
WORKSPACE="$SCRIPT_DIR/ob_workspace"   # flattened input for the pipeline
RUN_DIR="$SCRIPT_DIR/ob_run"           # all pipeline outputs go here

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Pre-flight checks ─────────────────────────────────────────────────────────
info "Pre-flight checks..."

[[ -f "$PIPELINE_SCRIPT" ]] || \
    error "extract_functions_and_embed.py not found at $PIPELINE_SCRIPT"
[[ -f "$PR_SCRIPT" ]] || \
    warn "plot_pr_curve.py not found — PR figures will be skipped"

python3 -c "import transformers, torch, tree_sitter_languages, matplotlib" 2>/dev/null || \
    error "Missing Python packages. Run:
    pip install transformers torch tree-sitter==0.20.4 tree-sitter-languages matplotlib"

# ── Step 1: Clone Online Boutique ─────────────────────────────────────────────
if [[ -d "$REPO_DIR" ]]; then
    info "Repo already cloned at $REPO_DIR — skipping clone."
else
    info "Cloning GoogleCloudPlatform/microservices-demo (shallow)..."
    git clone --depth 1 \
        https://github.com/GoogleCloudPlatform/microservices-demo.git \
        "$REPO_DIR"
    info "Clone complete."
fi

# ── Step 2: Build flat workspace ─────────────────────────────────────────────
info "Building pipeline workspace at $WORKSPACE..."
rm -rf "$WORKSPACE"
mkdir -p "$WORKSPACE"

# Services to include and their source subdirectories within the repo
# Format: "workspace_folder_name:repo_src_subpath"
declare -A SERVICES=(
    ["adservice"]="src/adservice/src/main/java/hipstershop"
    ["emailservice"]="src/emailservice"
    ["recommendationservice"]="src/recommendationservice"
    ["currencyservice"]="src/currencyservice"
    ["paymentservice"]="src/paymentservice"
)

for svc in "${!SERVICES[@]}"; do
    src_path="$REPO_DIR/${SERVICES[$svc]}"
    dst_path="$WORKSPACE/$svc"
    if [[ -d "$src_path" ]]; then
        cp -r "$src_path" "$dst_path"
        info "  ✓ $svc  ←  ${SERVICES[$svc]}"
    else
        warn "  ✗ $svc — source path not found: $src_path (skipping)"
    fi
done

# ── Step 3: Run the pipeline ──────────────────────────────────────────────────
info "Creating run directory at $RUN_DIR..."
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"

info "Running GraphCodeBERT pipeline..."
info "This will take several minutes (model inference on every function)."
echo ""

# Run from inside the run directory so all outputs land there
cd "$RUN_DIR"
python3 "$PIPELINE_SCRIPT" \
    --services-root "$WORKSPACE" \
    --output-dir    "$RUN_DIR/pipeline_output" \
    --embeddings    "$RUN_DIR/embeddings.pkl" \
    --report        "$RUN_DIR/clone_report.json" \
    --threshold     0.85 \
    2>&1 | tee "$RUN_DIR/pipeline.log"

info "Pipeline complete."
echo ""

# ── Step 4: Generate PR figures ───────────────────────────────────────────────
if [[ -f "$PR_SCRIPT" && -f "$RUN_DIR/clone_report.json" ]]; then
    info "Generating PR curve figures..."
    cd "$RUN_DIR"
    python3 "$PR_SCRIPT" "$RUN_DIR/clone_report.json" 2>&1 | tee -a "$RUN_DIR/pipeline.log"
else
    warn "Skipping PR figures (missing script or report file)."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
info "════════════════════════════════════════════════════════"
info "  Run complete. Results in: $RUN_DIR/"
info "  Key files:"
info "    clone_report.json       ← clone detection results"
info "    embeddings.pkl          ← function embeddings"
info "    fig_combined.png        ← PR curve figure"
info "    pipeline.log            ← full log"
info "════════════════════════════════════════════════════════"