"""
extract_functions_and_embed.py
──────────────────────────────
Pipeline (Phase 1):
  Raw source code (Java / JS / Python)
    → function-level extraction       (tree-sitter)
    → DFG extraction per function     (tree-sitter DFG)
    → Strategy A: line-based DFG slice
        Keep only lines containing DFG variable nodes
        + DFG_CONTEXT_LINES lines above/below for context.
        Inspired by: srcClone (Alomari & Stephan, 2022)
                     LLMxCPG  (Lekssays et al., 2025)
    → GraphCodeBERT embeddings        (microsoft/graphcodebert-base)
    → Cosine similarity clone report  (brute-force, HNSW added later)

Fixes applied (v2):
  Fix 1 — BPE-aware slicing trigger:
      Raw token count is multiplied by BPE_EXPANSION_FACTOR (1.4) before
      comparing to the token limit. This ensures functions that exceed the
      model's 512-position budget after BPE sub-word expansion are sliced
      BEFORE silent truncation occurs inside the model.
  Fix 2 — Minimum stub token filter (MIN_STUB_TOKENS = 15):
      Functions with fewer than 15 raw tokens are skipped entirely.
      Single-line stubs produce [CLS]-dominated embeddings with near-zero
      variance, causing artificially high cosine scores with all other stubs
      (primary false-positive source in zero-shot mode).
  Fix 3 — Raised CLONE_THRESHOLD from 0.80 → 0.90:
      At 0.80 the zero-shot model flags structural similarity (same language,
      same algorithmic shape) as clones. 0.90 is the empirically determined
      lower bound for cross-language semantic clones in this corpus.

Install:
    pip install transformers torch tree-sitter==0.20.4 tree-sitter-languages

Usage:
    python extract_functions_and_embed.py
"""

import os, json, pickle, time, argparse
import torch
import numpy as np
import torch.nn.functional as F
from itertools import combinations
from transformers import (
    AutoTokenizer,
    RobertaModel,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
SERVICES_ROOT       = "."
OUTPUT_JSON_DIR     = "pipeline_output"
EMBEDDINGS_FILE     = "embeddings.pkl"
REPORT_FILE         = "clone_report2.json"

MODEL_NAME          = "microsoft/graphcodebert-base"
LORA_ADAPTER_DIR    = "model"   # subdirectory containing the BCB fine-tuned adapter
CODE_LENGTH         = 256    # official GraphCodeBERT value
DATA_FLOW_LENGTH    = 64     # official GraphCodeBERT value
CLONE_THRESHOLD     = 0.90   # cosine similarity threshold for clone detection
                             # 0.90 chosen for zero-shot GraphCodeBERT:
                             # below this, structural similarity (same language,
                             # same algorithmic shape) dominates over semantics.

# Strategy A config
# Number of lines above and below each DFG line to keep as context.
# 1 = keep immediate neighbours (recommended for first phase)
DFG_CONTEXT_LINES   = 1

# Minimum raw token count for a function to be included in the pipeline.
# Single-line stub methods (e.g. `void send() { queue.add(msg); }`) produce
# embeddings dominated by the [CLS] token with near-zero variance, causing
# artificially high cosine similarity with all other short functions (false
# positives). 15 raw tokens ≈ a 2–3 line function with at least one branch
# or assignment — the smallest unit with meaningful DFG content.
MIN_STUB_TOKENS     = 15

IGNORE_DIRS = {
    "joern_output", "gcb_input", "pipeline_output", "workspace",
    ".vscode", ".metals", "node_modules", "__pycache__",
    "tree_sitter_grammars", "codebert-base-local"
}

# Files to skip entirely — auto-generated, test harness, or non-business-logic.
# Protobuf-generated *_pb2.py and *_pb2_grpc.py files contain only boilerplate
# gRPC stubs; they are identical across services and produce thousands of
# false-positive clone pairs.
IGNORE_FILES = {
    "demo_pb2.py", "demo_pb2_grpc.py",   # protobuf auto-generated (Python gRPC)
    "logger.py", "logger.js",             # trivial shared logging boilerplate
    "client.js", "email_client.py", "client.py",  # gRPC client/test harness
    "index.js",                           # service entry-point wrapper (no logic)
}

EXTENSION_TO_LANG = {
    ".py": "python", ".js": "javascript",
    ".ts": "javascript", ".java": "java",
}
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  PART 1 — FUNCTION EXTRACTION
#  Uses tree-sitter to find every function/method in a source file
#  and return its source text + location metadata.
# ══════════════════════════════════════════════════════════════════════════════

FUNCTION_NODE_TYPES = {
    "python":     ["function_definition", "async_function_definition"],
    "javascript": ["function_declaration", "function_expression",
                   "arrow_function", "generator_function_declaration",
                   "method_definition"],
    "java":       ["method_declaration", "constructor_declaration"],
}


def get_function_name(node):
    """Extract function/method name from a tree-sitter node."""
    for child in node.children:
        if child.type == "identifier":
            return child.text.decode("utf-8", errors="replace")
    return "<anonymous>"


def extract_functions_from_file(code: str, lang: str, parsers: dict) -> list:
    """
    Parse a source file and return every function as a separate unit.

    Each entry:
        {
          "name":       function name string,
          "code":       full source text of this function,
          "start_line": 1-based start line in the original file,
          "end_line":   1-based end line in the original file,
        }

    Why function-level?
        File-level embedding produces ONE vector per file, hiding
        which specific function is the clone. Function-level gives
        one vector per function — precise clone identification.
    """
    parser, _ = parsers[lang]
    tree       = parser.parse(bytes(code, "utf-8"))
    code_lines = code.splitlines()
    functions  = []

    target_types = FUNCTION_NODE_TYPES.get(lang, [])

    def walk(node):
        if node.type in target_types:
            start     = node.start_point[0]          # 0-based
            end       = node.end_point[0]             # 0-based
            func_code = "\n".join(code_lines[start : end + 1])
            name      = get_function_name(node)
            functions.append({
                "name":       name,
                "code":       func_code,
                "start_line": start + 1,
                "end_line":   end   + 1,
            })
            # Do not recurse — avoids double-counting nested functions
            return
        for child in node.children:
            walk(child)

    walk(tree.root_node)

    # Fallback: if no functions detected, treat whole file as one unit
    if not functions:
        functions.append({
            "name":       "<file>",
            "code":       code,
            "start_line": 1,
            "end_line":   len(code_lines),
        })

    return functions


# ══════════════════════════════════════════════════════════════════════════════
#  PART 2 — DFG EXTRACTION
#  Ported from official GraphCodeBERT/clonedetection/parser/DFG.py
#
#  states  →  variable_name : (last_token_idx, last_var_name)
#  DFG entry: (var, idx, 'computedFrom', [src_var_names], [src_token_idxs])
# ══════════════════════════════════════════════════════════════════════════════

def get_src(var, states):
    """Return (src_name, src_idx) for a variable from the states dict."""
    if var in states:
        return states[var][1], states[var][0]
    return var, -1


def DFG_python(node, index_to_code, states):
    DFG, t = [], node.type
    if t in ["assignment", "augmented_assignment"]:
        eq = [c for c in node.children if c.type == "="]
        if not eq:
            lefts, rights = [node.children[0]], [node.children[-1]]
        else:
            i = node.children.index(eq[0])
            lefts, rights = node.children[:i], node.children[i+1:]
        for n in rights:
            _, _, states, d = DFG_python(n, index_to_code, states)
            DFG += d
        for n in lefts:
            if n.type == "identifier" and n.start_byte in index_to_code:
                idx = index_to_code[n.start_byte][0]
                var = index_to_code[n.start_byte][1]
                sn, si = get_src(var, states)
                DFG.append((var, idx, "computedFrom", [sn], [si]))
                states[var] = (idx, var)
        return None, None, states, DFG
    elif t == "for_in_clause":
        for n in [node.children[-1]]:
            _, _, states, d = DFG_python(n, index_to_code, states)
            DFG += d
        for n in [node.children[1]]:
            if n.type == "identifier" and n.start_byte in index_to_code:
                idx = index_to_code[n.start_byte][0]
                var = index_to_code[n.start_byte][1]
                sn, si = get_src(var, states)
                DFG.append((var, idx, "computedFrom", [sn], [si]))
                states[var] = (idx, var)
        return None, None, states, DFG
    elif t == "identifier":
        if node.start_byte in index_to_code:
            idx = index_to_code[node.start_byte][0]
            var = index_to_code[node.start_byte][1]
            sn, si = get_src(var, states)
            DFG.append((var, idx, "computedFrom", [sn], [si]))
        return None, None, states, DFG
    elif t in ["function_definition", "async_function_definition",
               "class_definition", "decorated_definition", "lambda"]:
        return None, None, states, DFG   # skip nested scopes
    else:
        for c in node.children:
            _, _, states, d = DFG_python(c, index_to_code, states)
            DFG += d
        return None, None, states, DFG


def DFG_javascript(node, index_to_code, states):
    DFG, t = [], node.type
    if t in ["lexical_declaration", "variable_declaration"]:
        for c in node.children:
            _, _, states, d = DFG_javascript(c, index_to_code, states)
            DFG += d
        return None, None, states, DFG
    elif t == "variable_declarator":
        if len(node.children) >= 3:
            for n in [node.children[2]]:
                _, _, states, d = DFG_javascript(n, index_to_code, states)
                DFG += d
            for n in [node.children[0]]:
                if n.type == "identifier" and n.start_byte in index_to_code:
                    idx = index_to_code[n.start_byte][0]
                    var = index_to_code[n.start_byte][1]
                    sn, si = get_src(var, states)
                    DFG.append((var, idx, "computedFrom", [sn], [si]))
                    states[var] = (idx, var)
        return None, None, states, DFG
    elif t in ["assignment_expression", "augmented_assignment_expression"]:
        for n in [node.children[-1]]:
            _, _, states, d = DFG_javascript(n, index_to_code, states)
            DFG += d
        for n in [node.children[0]]:
            if n.type == "identifier" and n.start_byte in index_to_code:
                idx = index_to_code[n.start_byte][0]
                var = index_to_code[n.start_byte][1]
                sn, si = get_src(var, states)
                DFG.append((var, idx, "computedFrom", [sn], [si]))
                states[var] = (idx, var)
        return None, None, states, DFG
    elif t == "identifier":
        if node.start_byte in index_to_code:
            idx = index_to_code[node.start_byte][0]
            var = index_to_code[node.start_byte][1]
            sn, si = get_src(var, states)
            DFG.append((var, idx, "computedFrom", [sn], [si]))
        return None, None, states, DFG
    elif t in ["function_declaration", "function_expression",
               "generator_function_declaration"]:
        for c in node.children:
            if c.type == "statement_block":
                _, _, states, d = DFG_javascript(c, index_to_code, states)
                DFG += d
        return None, None, states, DFG
    elif t in ["arrow_function", "function", "class_declaration",
               "class_expression", "method_definition"]:
        return None, None, states, DFG   # skip nested scopes
    else:
        for c in node.children:
            _, _, states, d = DFG_javascript(c, index_to_code, states)
            DFG += d
        return None, None, states, DFG


def DFG_java(node, index_to_code, states):
    DFG, t = [], node.type
    if t == "local_variable_declaration":
        for c in node.children:
            _, _, states, d = DFG_java(c, index_to_code, states)
            DFG += d
        return None, None, states, DFG
    elif t == "variable_declarator":
        if len(node.children) >= 3:
            for n in [node.children[2]]:
                _, _, states, d = DFG_java(n, index_to_code, states)
                DFG += d
            for n in [node.children[0]]:
                if n.type == "identifier" and n.start_byte in index_to_code:
                    idx = index_to_code[n.start_byte][0]
                    var = index_to_code[n.start_byte][1]
                    sn, si = get_src(var, states)
                    DFG.append((var, idx, "computedFrom", [sn], [si]))
                    states[var] = (idx, var)
        return None, None, states, DFG
    elif t == "assignment_expression":
        for n in [node.children[-1]]:
            _, _, states, d = DFG_java(n, index_to_code, states)
            DFG += d
        for n in [node.children[0]]:
            if n.type == "identifier" and n.start_byte in index_to_code:
                idx = index_to_code[n.start_byte][0]
                var = index_to_code[n.start_byte][1]
                sn, si = get_src(var, states)
                DFG.append((var, idx, "computedFrom", [sn], [si]))
                states[var] = (idx, var)
        return None, None, states, DFG
    elif t == "identifier":
        if node.start_byte in index_to_code:
            idx = index_to_code[node.start_byte][0]
            var = index_to_code[node.start_byte][1]
            sn, si = get_src(var, states)
            DFG.append((var, idx, "computedFrom", [sn], [si]))
        return None, None, states, DFG
    elif t == "method_declaration":
        for c in node.children:
            if c.type == "block":
                _, _, states, d = DFG_java(c, index_to_code, states)
                DFG += d
        return None, None, states, DFG
    elif t in ["class_declaration", "constructor_declaration",
               "lambda_expression"]:
        return None, None, states, DFG   # skip nested scopes
    else:
        for c in node.children:
            _, _, states, d = DFG_java(c, index_to_code, states)
            DFG += d
        return None, None, states, DFG


DFG_FUNCTIONS = {
    "python": DFG_python,
    "javascript": DFG_javascript,
    "java": DFG_java,
}

FUNCTION_WRAPPERS = {
    "python":     ["function_definition", "async_function_definition"],
    "javascript": ["function_declaration", "function_expression",
                   "generator_function_declaration", "arrow_function"],
    "java":       ["method_declaration", "constructor_declaration"],
}


def find_function_body(root_node, lang):
    """Navigate from file root into the function body node."""
    wrappers   = FUNCTION_WRAPPERS.get(lang, [])
    body_types = {
        "python":     ["block"],
        "javascript": ["statement_block"],
        "java":       ["block"],
    }
    def find_w(node, depth=0):
        if node.type in wrappers:
            return node
        if depth < 3:
            for c in node.children:
                r = find_w(c, depth + 1)
                if r: return r
        return None
    wrapper = find_w(root_node)
    if not wrapper:
        return root_node
    for c in wrapper.children:
        if c.type in body_types.get(lang, []):
            return c
    return wrapper


def extract_tokens_and_dfg(code: str, lang: str, parsers: dict):
    """
    Extract code tokens and DFG from a single function's source code.
    Navigates into the function body before running DFG
    (matches official GraphCodeBERT extract_dataflow() behaviour).
    """
    parser, dfg_func = parsers[lang]
    tree             = parser.parse(bytes(code, "utf-8"))
    file_root        = tree.root_node

    # Collect all leaf tokens across the whole function (skipping comments)
    tokens_index = []
    def collect(node):
        if "comment" in node.type:
            return
        if node.child_count == 0 and node.start_byte != node.end_byte:
            tokens_index.append((node.start_byte, node.end_byte))
        for c in node.children:
            collect(c)
    collect(file_root)

    code_bytes    = code.encode("utf-8")
    code_tokens   = []
    index_to_code = {}
    for i, (s, e) in enumerate(tokens_index):
        tok = code_bytes[s:e].decode("utf-8", errors="replace")
        code_tokens.append(tok)
        index_to_code[s] = (i, tok)

    # Run DFG on function BODY only (not the wrapper node)
    body_node    = find_function_body(file_root, lang)
    _, _, _, dfg = dfg_func(body_node, index_to_code, {})

    # Remove self-loops and deduplicate
    dfg = [d for d in dfg if d[1] not in d[4]]
    seen, dfg_clean = set(), []
    for d in dfg:
        k = (d[1], tuple(d[4]))
        if k not in seen:
            seen.add(k)
            dfg_clean.append(d)

    return code_tokens, dfg_clean


# ══════════════════════════════════════════════════════════════════════════════
#  PART 3 — STRATEGY A: LINE-BASED DFG SLICE
#
#  Motivation (srcClone, Alomari & Stephan 2022):
#    Program slicing produces a smaller, semantically equivalent
#    representation of a function by keeping only the code elements
#    that affect the variables of interest.
#
#  Our adaptation for clone detection:
#    Criterion = lines containing DFG variable definition/use nodes.
#    These are the lines that define the function's LOGIC.
#    Lines with only logging, comments, or boilerplate have no DFG
#    nodes and are removed.
#
#  Steps:
#    1. Map each DFG token index → source line number
#    2. Collect all lines referenced by DFG nodes
#    3. Expand with ±DFG_CONTEXT_LINES for syntactic context
#    4. Reconstruct source from kept lines
#    5. Re-tokenize the slice (clean token sequence for model)
# ══════════════════════════════════════════════════════════════════════════════

def strategy_a_slice(code: str, dfg: list,
                     code_tokens: list, lang: str,
                     parsers: dict):
    """
    Strategy A: Line-based DFG-guided slicing.

    Returns:
        slice_tokens  — tokens of the compressed slice
        slice_dfg     — DFG of the compressed slice
        slice_code    — human-readable source of the slice
        kept_lines    — list of kept line numbers (1-based)
        stats         — dict with reduction metrics
    """
    code_lines = code.splitlines()
    n_lines    = len(code_lines)
    n_tokens   = len(code_tokens)

    # ── Case: no DFG or already within token limit ─────────────────
    # BPE-aware trigger: GraphCodeBERT's BPE tokenizer expands raw tokens
    # (e.g. camelCase identifiers split into multiple sub-words). Empirically,
    # BPE output is ~1.4× the raw token count. We apply this multiplier so
    # that functions which would exceed the model's 512-position limit after
    # BPE expansion are sliced BEFORE silent truncation occurs inside the model.
    token_limit = CODE_LENGTH + DATA_FLOW_LENGTH
    BPE_EXPANSION_FACTOR = 1.4
    if not dfg or (n_tokens * BPE_EXPANSION_FACTOR) <= token_limit:
        kept = list(range(1, n_lines + 1))
        return code_tokens, dfg, code, kept, {
            "original_tokens": n_tokens,
            "slice_tokens":    n_tokens,
            "original_lines":  n_lines,
            "kept_lines":      n_lines,
            "dfg_lines_found": 0,
            "reduction_pct":   0.0,
            "slicing_applied": False,
        }

    # ── Step 1: build token_index → line_number map ────────────────
    parser, _ = parsers[lang]
    tree       = parser.parse(bytes(code, "utf-8"))

    token_line_map = {}   # token_index (int) → line_number (1-based int)
    idx = [0]
    def collect_lines(node):
        if "comment" in node.type:
            return
        if node.child_count == 0 and node.start_byte != node.end_byte:
            token_line_map[idx[0]] = node.start_point[0] + 1
            idx[0] += 1
        for c in node.children:
            collect_lines(c)
    collect_lines(tree.root_node)

    # ── Step 2: collect lines referenced by DFG nodes ──────────────
    dfg_lines = set()
    for var, tok_idx, _, src_vars, src_idxs in dfg:
        if tok_idx in token_line_map:
            dfg_lines.add(token_line_map[tok_idx])
        for si in src_idxs:
            if si >= 0 and si in token_line_map:
                dfg_lines.add(token_line_map[si])

    if not dfg_lines:
        # DFG exists but tokens not mappable — return original
        kept = list(range(1, n_lines + 1))
        return code_tokens, dfg, code, kept, {
            "original_tokens": n_tokens,
            "slice_tokens":    n_tokens,
            "original_lines":  n_lines,
            "kept_lines":      n_lines,
            "dfg_lines_found": 0,
            "reduction_pct":   0.0,
            "slicing_applied": False,
        }

    # ── Step 3: expand with ±context lines ─────────────────────────
    kept_set = set()
    for ln in dfg_lines:
        for offset in range(-DFG_CONTEXT_LINES, DFG_CONTEXT_LINES + 1):
            nb = ln + offset
            if 1 <= nb <= n_lines:
                kept_set.add(nb)

    kept_lines = sorted(kept_set)

    # ── Step 4: reconstruct slice source ───────────────────────────
    slice_code = "\n".join(code_lines[ln - 1] for ln in kept_lines)

    # ── Step 5: re-tokenize + re-extract DFG from the slice ────────
    try:
        slice_tokens, slice_dfg = extract_tokens_and_dfg(
            slice_code, lang, parsers)
    except Exception:
        # Slice may be syntactically incomplete — fallback to original
        slice_tokens, slice_dfg = code_tokens, dfg

    reduction = (1 - len(slice_tokens) / max(1, n_tokens)) * 100

    stats = {
        "original_tokens": n_tokens,
        "slice_tokens":    len(slice_tokens),
        "original_lines":  n_lines,
        "kept_lines":      len(kept_lines),
        "dfg_lines_found": len(dfg_lines),
        "reduction_pct":   round(reduction, 1),
        "slicing_applied": True,
    }

    return slice_tokens, slice_dfg, slice_code, kept_lines, stats


# ══════════════════════════════════════════════════════════════════════════════
#  PART 4 — GRAPHCODEBERT EMBEDDING
#  Official input format: code tokens + DFG → 768-dim vector
# ══════════════════════════════════════════════════════════════════════════════

def resolve_lora_adapter_dir(adapter_dir: str | None = None):
    """
    Return a directory that looks like a PEFT LoRA checkpoint, or None.

    We prefer the Hugging Face PEFT format:
      - adapter_config.json
      - adapter_model.safetensors (or adapter_model.bin)

    Search order:
      1. Explicit adapter_dir argument
      2. LORA_ADAPTER_DIR config value (default: "model" subdir)
      3. <script_dir>/model/            ← preferred: BCB fine-tuned checkpoint
      4. Current working directory
      5. Script directory (fallback to any adapter found nearby)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = []
    if adapter_dir:
        candidates.append(adapter_dir)
        # Also try an explicit adapter_dir/model sub-path in case caller passes root
        candidates.append(os.path.join(adapter_dir, "model"))
    # Resolve LORA_ADAPTER_DIR as a relative path from script dir
    lora_rel = os.path.join(script_dir, LORA_ADAPTER_DIR)
    candidates.extend([
        lora_rel,
        os.path.join(script_dir, "model"),
        os.path.join(os.getcwd(), "model"),
        os.getcwd(),
        script_dir,
    ])

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(candidate)
        if candidate in seen or not os.path.isdir(candidate):
            continue
        seen.add(candidate)

        has_config = os.path.exists(os.path.join(candidate, "adapter_config.json"))
        has_weights = (
            os.path.exists(os.path.join(candidate, "adapter_model.safetensors")) or
            os.path.exists(os.path.join(candidate, "adapter_model.bin"))
        )
        if has_config and has_weights:
            return candidate
    return None


def load_graphcodebert_model(adapter_dir: str | None = None):
    """
    Load GraphCodeBERT for embeddings, optionally with a local LoRA adapter.

    Architecture note:
      The BCB fine-tuned LoRA adapter was trained with RobertaModel as the
      base (task_type=FEATURE_EXTRACTION), NOT RobertaForSequenceClassification.
      Keys in adapter_model.safetensors are keyed as:
          base_model.model.encoder.layer.X.attention.self.{query,value}.lora_*
      Loading into RobertaForSequenceClassification produces a path mismatch
      (.roberta.encoder vs .encoder) so the adapter weights are never applied.
      We therefore load the plain RobertaModel to match training exactly.

    For function embeddings we extract the [CLS] hidden state — the same
    representation used during training to compute clone similarity.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Use plain RobertaModel — matches the training architecture exactly
    base_model = RobertaModel.from_pretrained(
        MODEL_NAME,
        add_pooling_layer=False,
    )

    resolved_adapter_dir = resolve_lora_adapter_dir(adapter_dir)
    if not resolved_adapter_dir:
        return tokenizer, base_model.eval(), None

    try:
        from peft import PeftModel
    except ImportError as e:
        raise ImportError(
            "LoRA adapter files were found, but PEFT is not installed. "
            "Run: pip install peft"
        ) from e

    model = PeftModel.from_pretrained(base_model, resolved_adapter_dir)
    return tokenizer, model.eval(), resolved_adapter_dir


def get_encoder_hidden_states(model, input_ids, position_ids, attention_mask):
    """
    Return token hidden states from the GraphCodeBERT encoder regardless of
    whether we loaded the plain RobertaModel or a LoRA-wrapped PeftModel.

    With base = RobertaModel (BCB training architecture):
      - PeftModel delegates __getattr__ to RobertaModel, which has no .roberta
      - We call model() directly; RobertaModel returns BaseModelOutput with
        .last_hidden_state → (batch, seq_len, hidden_size)

    With base = RobertaForSequenceClassification (legacy fallback):
      - model.roberta resolves to the inner RobertaModel via __getattr__
      - We call roberta() directly to skip the classification head
    """
    # Try .roberta sub-model (RobertaForSequenceClassification path)
    roberta = getattr(model, "roberta", None)
    if roberta is not None and callable(roberta):
        return roberta(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )[0]

    # Direct call — works for PeftModel(RobertaModel) and plain RobertaModel
    out = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    # Final fallback: out is a tuple, first element is last_hidden_state
    return out[0]


def build_graphcodebert_inputs(code_tokens: list, dfg: list, tokenizer):
    """
    Convert code tokens + DFG into GraphCodeBERT tensor inputs.

    Input sequence layout:
        [CLS] <bpe_code_tokens> [SEP] <dfg_var_nodes> [PAD...]

    Attention: graph-guided mask — DFG nodes attend to their
    corresponding code token positions (where variables appear).
    """
    # BPE sub-word tokenization
    bpe_tokens, orig_indices = [], []
    for i, tok in enumerate(code_tokens):
        sub = tokenizer.tokenize(tok)
        if not sub:
            continue
        bpe_tokens.extend(sub)
        orig_indices.extend([i] * len(sub))

    # Truncate to fit model limits
    max_dfg  = min(len(dfg), DATA_FLOW_LENGTH)
    max_code = min(CODE_LENGTH + DATA_FLOW_LENGTH - 3 - max_dfg, 512 - 3)
    bpe_tokens   = bpe_tokens[:max_code]
    orig_indices = orig_indices[:max_code]

    # Build [CLS] code [SEP] sequence
    src_tokens = [tokenizer.cls_token] + bpe_tokens + [tokenizer.sep_token]
    src_ids    = tokenizer.convert_tokens_to_ids(src_tokens)
    pos_idx    = [i + tokenizer.pad_token_id + 1
                  for i in range(len(src_tokens))]

    # Append DFG variable nodes (one token per DFG entry)
    dfg_trunc = dfg[:CODE_LENGTH + DATA_FLOW_LENGTH - len(src_tokens)]
    for var_name, *_ in dfg_trunc:
        sub = tokenizer.tokenize(var_name) or [tokenizer.unk_token]
        src_tokens.append(sub[0])
        src_ids.append(tokenizer.convert_tokens_to_ids(sub[0]))
        pos_idx.append(1)

    # Pad to max_len
    max_len = CODE_LENGTH + DATA_FLOW_LENGTH
    pad_len = max_len - len(src_ids)
    src_ids += [tokenizer.pad_token_id] * pad_len
    pos_idx += [tokenizer.pad_token_id] * pad_len

    # Graph-guided attention mask (max_len × max_len)
    attn     = np.zeros((max_len, max_len), dtype=np.bool_)
    code_end = len(src_tokens) - len(dfg_trunc)
    attn[:code_end, :code_end] = True   # code tokens attend to all code

    for di, (_, tok_idx, _, src_vars, src_idxs) in enumerate(dfg_trunc):
        sp = code_end + di
        if sp >= max_len:
            break
        attn[sp, :code_end] = True      # DFG node attends to all code
        for si in src_idxs:             # DFG node attends to source tokens
            if si >= 0:
                for sub_pos, oi in enumerate(orig_indices):
                    if oi == si:
                        attn[sp, sub_pos + 1] = True
        for dj, (_, _, _, _, other_src) in enumerate(dfg_trunc):
            op = code_end + dj          # DFG nodes with shared sources
            if op < max_len and set(src_idxs) & set(other_src):
                attn[sp, op] = True

    input_ids = torch.tensor(src_ids, dtype=torch.long)
    pos_ids   = torch.tensor(pos_idx, dtype=torch.long)
    attn_t    = torch.from_numpy(attn)
    return input_ids, pos_ids, attn_t


def to_embedding(code_tokens: list, dfg: list, tokenizer, model):
    """
    Convert code tokens + DFG to a 768-dim GraphCodeBERT embedding.
    """
    input_ids, pos_ids, attn_t = build_graphcodebert_inputs(
        code_tokens,
        dfg,
        tokenizer,
    )

    with torch.no_grad():
        out = get_encoder_hidden_states(
            model,
            input_ids=input_ids.unsqueeze(0),
            position_ids=pos_ids.unsqueeze(0),
            attention_mask=attn_t.unsqueeze(0),
        )

    # [CLS] token embedding = function semantic fingerprint
    return out[0, 0, :].cpu()


def score_clone_similarity(record_a: dict, record_b: dict, model):
    """
    Score a function pair using cosine similarity between the fine-tuned CLS
    representations.
    """
    ids = torch.stack([record_a["input_ids"], record_b["input_ids"]], dim=0)
    pos = torch.stack([record_a["position_ids"], record_b["position_ids"]], dim=0)
    attn = torch.stack([record_a["attention_mask"], record_b["attention_mask"]], dim=0)

    with torch.no_grad():
        hidden = get_encoder_hidden_states(
            model,
            input_ids=ids,
            position_ids=pos,
            attention_mask=attn,
        )

    cls_a = hidden[0, 0, :]
    cls_b = hidden[1, 0, :]
    cosine = torch.nn.functional.cosine_similarity(
        cls_a.unsqueeze(0),
        cls_b.unsqueeze(0),
    ).item()
    return cosine


# ══════════════════════════════════════════════════════════════════════════════
#  PART 5 — SERVICE DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def find_service_files(root: str) -> list:
    """Walk service folders and return all supported source files."""
    files = []
    for svc in sorted(os.listdir(root)):
        svc_path = os.path.join(root, svc)
        if not os.path.isdir(svc_path):
            continue
        if svc in IGNORE_DIRS or svc.startswith("."):
            continue
        for r, dirs, fnames in os.walk(svc_path):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for fname in fnames:
                if fname in IGNORE_FILES:
                    print(f"    [SKIP file] {svc}/{fname}  (in IGNORE_FILES)")
                    continue
                ext  = os.path.splitext(fname)[1].lower()
                lang = EXTENSION_TO_LANG.get(ext)
                if lang and lang in DFG_FUNCTIONS:
                    files.append((svc, os.path.join(r, fname), lang))
    return files


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global SERVICES_ROOT, OUTPUT_JSON_DIR, EMBEDDINGS_FILE, REPORT_FILE
    global CLONE_THRESHOLD, LORA_ADAPTER_DIR

    # ── CLI arguments (all optional — fall back to CONFIG defaults) ────────
    parser = argparse.ArgumentParser(
        description="GraphCodeBERT cross-service clone detection pipeline")
    parser.add_argument("--services-root", default=None,
        help="Root folder containing one sub-folder per service "
             "(default: current working directory)")
    parser.add_argument("--output-dir", default=None,
        help="Directory for per-function JSON inspection files "
             "(default: ./pipeline_output)")
    parser.add_argument("--embeddings", default=None,
        help="Path to save/load embeddings .pkl file "
             "(default: ./embeddings.pkl)")
    parser.add_argument("--report", default=None,
        help="Path for the clone_report2.json output "
             "(default: ./clone_report2.json)")
    parser.add_argument("--threshold", type=float, default=None,
        help=f"Cosine similarity threshold for clone detection "
             f"(default: {CLONE_THRESHOLD})")
    parser.add_argument("--lora-adapter-dir", default=None,
        help="Directory containing PEFT adapter files "
             "(adapter_config.json + adapter_model.safetensors). "
             "If omitted, auto-detect from the repo root/current directory.")
    args = parser.parse_args()

    # Apply CLI overrides
    if args.services_root:         SERVICES_ROOT   = args.services_root
    if args.output_dir:            OUTPUT_JSON_DIR = args.output_dir
    if args.embeddings:            EMBEDDINGS_FILE = args.embeddings
    if args.report:                REPORT_FILE     = args.report
    if args.threshold is not None: CLONE_THRESHOLD = args.threshold
    if args.lora_adapter_dir:      LORA_ADAPTER_DIR = args.lora_adapter_dir

    print("=" * 65)
    print("  GraphCodeBERT Pipeline — Phase 1")
    print("  Strategy A: Line-based DFG-guided slicing")
    print("=" * 65)
    print(f"  Services root : {SERVICES_ROOT}")
    print(f"  Output dir    : {OUTPUT_JSON_DIR}")
    print(f"  Embeddings    : {EMBEDDINGS_FILE}")
    print(f"  Report        : {REPORT_FILE}")
    print(f"  Threshold     : {CLONE_THRESHOLD}")
    print(f"  LoRA adapter  : {LORA_ADAPTER_DIR}")

    # ── [1/5] Build parsers ────────────────────────────────────────
    print("\n[1/5] Building tree-sitter parsers...")
    try:
        from tree_sitter_languages import get_parser
    except ImportError:
        raise ImportError(
            "Run: pip install tree-sitter==0.20.4 tree-sitter-languages")

    parsers = {}
    for lang in DFG_FUNCTIONS:
        parsers[lang] = (get_parser(lang), DFG_FUNCTIONS[lang])
        print(f"  ✓ {lang}")

    # ── [2/5] Discover files ───────────────────────────────────────
    print("\n[2/5] Discovering service files...")
    all_files = find_service_files(SERVICES_ROOT)
    print(f"  Found {len(all_files)} source file(s):")
    for svc, path, lang in all_files:
        print(f"    {svc}/{os.path.basename(path)}  ({lang})")

    if not all_files:
        print("  No supported files found. Exiting.")
        return

    # ── [3/5] Extract functions → DFG → Strategy A slice ──────────
    print("\n[3/5] Extracting functions + DFG + Strategy A slicing...")
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    all_records = []
    needs_slice_count = 0

    for svc, file_path, lang in all_files:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()

        functions = extract_functions_from_file(code, lang, parsers)
        print(f"\n  ── {svc}/{os.path.basename(file_path)} "
              f"({lang}) → {len(functions)} function(s)")

        for fn in functions:
            fname  = fn["name"]
            fcode  = fn["code"]
            fstart = fn["start_line"]
            fend   = fn["end_line"]

            try:
                # Step 1: raw tokens + DFG
                tokens_raw, dfg_raw = extract_tokens_and_dfg(
                    fcode, lang, parsers)

                # ── Fix 2: skip trivial stub functions ─────────────────────
                # Functions under MIN_STUB_TOKENS raw tokens are single-line
                # stubs whose embeddings have near-zero variance (dominated by
                # the [CLS] token), causing high cosine similarity with every
                # other stub regardless of intent — a primary source of false
                # positives in zero-shot GraphCodeBERT.
                if len(tokens_raw) < MIN_STUB_TOKENS:
                    print(f"    [SKIP stub] {fname}()"
                          f"  lines {fstart}-{fend}"
                          f"  raw={len(tokens_raw)} tokens"
                          f"  (< MIN_STUB_TOKENS={MIN_STUB_TOKENS})")
                    continue

                # Step 2: Strategy A slice
                tokens_s, dfg_s, code_s, kept_lines, stats = strategy_a_slice(
                    fcode, dfg_raw, tokens_raw, lang, parsers)

                if stats["slicing_applied"]:
                    needs_slice_count += 1

                record = {
                    "service":       svc,
                    "file":          file_path,
                    "function_name": fname,
                    "lang":          lang,
                    "start_line":    fstart,
                    "end_line":      fend,
                    # Raw (before slicing)
                    "raw_tokens":    tokens_raw,
                    "raw_dfg":       dfg_raw,
                    # Slice (what goes to GraphCodeBERT)
                    "slice_tokens":  tokens_s,
                    "slice_dfg":     dfg_s,
                    "slice_code":    code_s,
                    "kept_lines":    kept_lines,
                    "stats":         stats,
                }
                all_records.append(record)

                # Console summary
                flag = "⚠ SLICED" if stats["slicing_applied"] else "✓ fits "
                print(f"    [{flag}] {fname}()"
                      f"  lines {fstart}-{fend}"
                      f"  raw={stats['original_tokens']} tokens"
                      f"  → slice={stats['slice_tokens']} tokens"
                      f"  reduction={stats['reduction_pct']}%"
                      f"  dfg={len(dfg_raw)}")

                # Save per-function inspection JSON
                safe = (f"{svc}_{fname}_{fstart}"
                        .replace("/","_").replace("\\","_") + ".json")
                with open(os.path.join(OUTPUT_JSON_DIR, safe),
                          "w", encoding="utf-8") as jf:
                    json.dump({
                        "service":        svc,
                        "file":           file_path,
                        "function":       fname,
                        "lang":           lang,
                        "lines":          f"{fstart}-{fend}",
                        "slicing_stats":  stats,
                        "kept_lines":     kept_lines,
                        "slice_preview":  code_s[:500],
                        "slice_tokens_preview": tokens_s[:20],
                        "slice_dfg_preview":    dfg_s[:5],
                    }, jf, indent=2)

            except Exception as e:
                print(f"    ✗ {fname}(): ERROR — {e}")

    # Slicing summary
    print(f"\n  Functions processed:     {len(all_records)}")
    print(f"  Functions that needed slicing: {needs_slice_count}")
    print(f"  Functions within limit:  {len(all_records) - needs_slice_count}")

    # ── [4/5] GraphCodeBERT embeddings ────────────────────────────
    print(f"\n[4/5] Generating GraphCodeBERT embeddings...")
    print(f"  Loading: {MODEL_NAME}")
    tokenizer, model, adapter_dir = load_graphcodebert_model(LORA_ADAPTER_DIR)
    if adapter_dir:
        print(f"  LoRA adapter: {adapter_dir}")
    else:
        print("  LoRA adapter: none found, using base GraphCodeBERT")
    print("  Model ready.\n")

    embedding_db = {}

    for rec in all_records:
        # Key: service/file::function_name  (unique per function)
        key = (f"{rec['service']}/{os.path.basename(rec['file'])}"
               f"::{rec['function_name']}")

        try:
            emb = to_embedding(
                rec["slice_tokens"], rec["slice_dfg"],
                tokenizer, model)

            embedding_db[key] = {
                "service":        rec["service"],
                "file":           rec["file"],
                "function":       rec["function_name"],
                "lang":           rec["lang"],
                "start_line":     rec["start_line"],
                "end_line":       rec["end_line"],
                "raw_tokens":     len(rec["raw_tokens"]),
                "slice_tokens":   len(rec["slice_tokens"]),
                "dfg_count":      len(rec["slice_dfg"]),
                "reduction_pct":  rec["stats"]["reduction_pct"],
                "sliced":         rec["stats"]["slicing_applied"],
                "embedding":      emb,
            }
            sliced_tag = "[sliced]" if rec["stats"]["slicing_applied"] else ""
            print(f"  ✓ {key} {sliced_tag}"
                  f"  tokens={len(rec['slice_tokens'])}"
                  f"  shape={emb.shape}")

        except Exception as e:
            print(f"  ✗ {key}: ERROR — {e}")

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embedding_db, f)
    print(f"\n  Saved {len(embedding_db)} embeddings → {EMBEDDINGS_FILE}")

    # ── [5/5] Clone detection via cosine similarity ────────────────
    print(f"\n[5/5] Computing cross-service cosine similarity...")
    print(f"  Threshold: {CLONE_THRESHOLD}")

    keys   = list(embedding_db.keys())
    pairs  = []

    for k1, k2 in combinations(keys, 2):
        svc1 = embedding_db[k1]["service"]
        svc2 = embedding_db[k2]["service"]
        if svc1 == svc2:
            continue   # skip same-service pairs

        e1    = embedding_db[k1]["embedding"].unsqueeze(0)
        e2    = embedding_db[k2]["embedding"].unsqueeze(0)
        score = F.cosine_similarity(e1, e2).item()

        pairs.append({
            "score":      round(score, 4),
            "function_1": k1,
            "service_1":  svc1,
            "lang_1":     embedding_db[k1]["lang"],
            "function_2": k2,
            "service_2":  svc2,
            "lang_2":     embedding_db[k2]["lang"],
            "is_clone":   score >= CLONE_THRESHOLD,
            "confidence": ("HIGH"   if score >= 0.95 else
                           "MEDIUM" if score >= CLONE_THRESHOLD else
                           "LOW"),
        })

    pairs.sort(key=lambda x: x["score"], reverse=True)
    clones = [p for p in pairs if p["is_clone"]]

    # Print report
    print(f"\n{'=' * 65}")
    print(f"  CLONE DETECTION REPORT")
    print(f"{'=' * 65}")

    if clones:
        print(f"\n  ⚠  {len(clones)} POTENTIAL CLONE(S) DETECTED\n")
        for p in clones:
            badge = "★ HIGH  " if p["confidence"] == "HIGH" else "~ MEDIUM"
            print(f"  [{badge}]  score={p['score']:.4f}"
                  f"  ({p['lang_1']} ↔ {p['lang_2']})")
            print(f"    {p['function_1']}")
            print(f"    {p['function_2']}")
            print()
    else:
        print(f"\n  ✓ No clones detected above threshold {CLONE_THRESHOLD}\n")

    print(f"  All cross-service pairs (ranked):")
    print(f"  {'Score':<8} {'Lang pair':<14} "
          f"{'Function 1':<35} Function 2")
    print(f"  {'-'*8} {'-'*14} {'-'*35} {'-'*35}")
    for p in pairs:
        flag     = " ← CLONE" if p["is_clone"] else ""
        langpair = f"{p['lang_1'][:2]}↔{p['lang_2'][:2]}"
        print(f"  {p['score']:.4f}   {langpair:<14} "
              f"{p['function_1']:<35} {p['function_2']}{flag}")

    print(f"\n{'=' * 65}")
    print(f"  Total cross-service pairs: {len(pairs)}")
    print(f"  Clones detected:           {len(clones)}")
    print(f"{'=' * 65}")

    # Save full report
    report = {
        "config": {
            "model":           MODEL_NAME,
            "threshold":       CLONE_THRESHOLD,
            "code_length":     CODE_LENGTH,
            "dfg_length":      DATA_FLOW_LENGTH,
            "context_lines":   DFG_CONTEXT_LINES,
            "slicing_strategy": "A — line-based DFG-guided",
        },
        "summary": {
            "functions_processed":       len(all_records),
            "functions_sliced":          needs_slice_count,
            "embeddings_generated":      len(embedding_db),
            "cross_service_pairs":       len(pairs),
            "clones_detected":           len(clones),
        },
        "slicing_details": [
            {
                "service":       r["service"],
                "function":      r["function_name"],
                "lang":          r["lang"],
                "lines":         f"{r['start_line']}-{r['end_line']}",
                "raw_tokens":    r["stats"]["original_tokens"],
                "slice_tokens":  r["stats"]["slice_tokens"],
                "reduction_pct": r["stats"]["reduction_pct"],
                "sliced":        r["stats"]["slicing_applied"],
            }
            for r in all_records
        ],
        "clone_pairs": clones,
        "all_pairs":   pairs,
    }
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Inspection JSONs → {OUTPUT_JSON_DIR}/")
    print(f"  Embeddings       → {EMBEDDINGS_FILE}")
    print(f"  Clone report     → {REPORT_FILE}")


if __name__ == "__main__":
    main()
