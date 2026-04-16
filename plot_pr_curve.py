"""
Online Boutique — Generate 4 separate plot files from clone_report.json
Usage: python3 plot_ob2_separate.py [path/to/clone_report.json]

Outputs:
  ob2_fig1_score_distribution.png
  ob2_fig2_top15_clones.png
  ob2_fig3_language_pairs.png
  ob2_fig4_threshold_sweep.png
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── Load data ─────────────────────────────────────────────────────────────────
report_path = sys.argv[1] if len(sys.argv) > 1 else 'clone_report.json'
r = json.load(open(report_path))
all_pairs   = r['all_pairs']
clone_pairs = r['clone_pairs']

clone_scores = [p['score'] for p in all_pairs if p['is_clone']]
non_scores   = [p['score'] for p in all_pairs if not p['is_clone']]

# Ground truth: manually confirmed true clones
TRUE_CLONES = {
    ('emailservice/email_server.py::Check',
     'recommendationservice/recommendation_server.py::Check'),
    ('emailservice/email_server.py::Watch',
     'recommendationservice/recommendation_server.py::Watch'),
    ('emailservice/email_server.py::initStackdriverProfiling',
     'recommendationservice/recommendation_server.py::initStackdriverProfiling'),
}

def is_true(p):
    key  = (p['function_1'], p['function_2'])
    key2 = (p['function_2'], p['function_1'])
    return key in TRUE_CLONES or key2 in TRUE_CLONES

# Language colours
LANG_COLOURS = {
    'python↔python':          '#4CAF50',
    'java↔javascript':        '#FF9800',
    'javascript↔javascript':  '#2196F3',
    'java↔python':            '#9C27B0',
    'javascript↔python':      '#F44336',
    'python↔java':            '#9C27B0',
    'python↔javascript':      '#F44336',
}
def lang_pair(p):
    return f"{p['lang_1']}↔{p['lang_2']}"
def lang_colour(p):
    return LANG_COLOURS.get(lang_pair(p), '#607D8B')

GRID_KW = dict(alpha=0.3, linestyle='--', linewidth=0.7)

# ── Threshold sweep data ───────────────────────────────────────────────────────
thresholds = np.arange(0.70, 1.001, 0.005)
precisions, recalls, f1s = [], [], []
for t in thresholds:
    tp = sum(1 for p in all_pairs if p['score'] >= t and is_true(p))
    fp = sum(1 for p in all_pairs if p['score'] >= t and not is_true(p))
    fn = sum(1 for p in all_pairs if p['score'] <  t and is_true(p))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)

best_idx  = int(np.argmax(f1s))
best_t    = thresholds[best_idx]
best_f1   = f1s[best_idx]
best_prec = precisions[best_idx]
best_rec  = recalls[best_idx]

# ═════════════════════════════════════════════════════════════════════════════
# Figure 1 — Score Distribution
# ═════════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(8, 5))
fig1.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#F8F9FA')

bins = np.linspace(0.0, 1.01, 51)  # 0.0 to 1.0 with 0.02 width bins
ax.hist(non_scores,   bins=bins, alpha=0.55, color='#EF5350',
        label=f'Non-clone ({len(non_scores):,})', edgecolor='white', linewidth=0.4)
ax.hist(clone_scores, bins=bins, alpha=0.75, color='#42A5F5',
        label=f'Detected clone ({len(clone_scores):,})', edgecolor='white', linewidth=0.4)
ax.axvline(0.85, color='#FFA000', linewidth=2, linestyle='--', label='τ = 0.85')
ax.axvline(np.mean(clone_scores), color='#1565C0', linewidth=1.5, linestyle=':',
           label=f'Clone mean = {np.mean(clone_scores):.3f}')
ax.axvline(np.mean(non_scores), color='#B71C1C', linewidth=1.5, linestyle=':',
           label=f'Non-clone mean = {np.mean(non_scores):.3f}')

ax.set_xlabel('Cosine Similarity Score', fontsize=12)
ax.set_ylabel('Number of Pairs', fontsize=12)
ax.set_title('Score Distribution — Clone vs Non-Clone Pairs\n'
             'Online Boutique · Zero-Shot GraphCodeBERT',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=9)
ax.grid(**GRID_KW)

fig1.tight_layout()
fig1.savefig('ob2_fig1_score_distribution.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print('Saved → ob2_fig1_score_distribution.png')
plt.close(fig1)

# ═════════════════════════════════════════════════════════════════════════════
# Figure 2 — Top 15 Clone Pairs Bar Chart
# ═════════════════════════════════════════════════════════════════════════════
top15 = sorted(clone_pairs, key=lambda x: x['score'], reverse=True)[:15]

def short_name(fn):
    parts = fn.split('::')
    svc   = parts[0].split('/')[0][:14]
    func  = parts[1][:20] if len(parts) > 1 else ''
    return f"{svc}::{func}"

labels     = [f"{short_name(p['function_1'])}  ↔  {short_name(p['function_2'])}" for p in top15]
scores_t15 = [p['score'] for p in top15]
colours    = [lang_colour(p) for p in top15]
true_marks = ['★ CONFIRMED TRUE CLONE' if is_true(p) else '' for p in top15]

fig2, ax = plt.subplots(figsize=(12, 7))
fig2.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#F8F9FA')

bars = ax.barh(range(len(top15)), scores_t15, color=colours,
               edgecolor='white', linewidth=0.7, height=0.72)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(labels, fontsize=8.5)
ax.invert_yaxis()
ax.set_xlim(0.83, 1.04)
ax.set_xlabel('Cosine Similarity Score', fontsize=12)
ax.set_title('Top 15 Detected Clone Pairs\n'
             'Online Boutique · Zero-Shot GraphCodeBERT',
             fontsize=13, fontweight='bold', pad=12)
ax.axvline(0.85, color='#FFA000', linewidth=2, linestyle='--', alpha=0.9, label='τ = 0.85')
ax.axvline(0.997, color='#2E7D32', linewidth=1.5, linestyle=':', alpha=0.9,
           label='Confirmed true clone boundary (≥0.997)')
ax.grid(axis='x', **GRID_KW)

for i, (score, mark) in enumerate(zip(scores_t15, true_marks)):
    ax.text(score + 0.001, i, f'{score:.4f}  {mark}',
            va='center', fontsize=8,
            color='#2E7D32' if mark else '#333333',
            fontweight='bold' if mark else 'normal')

legend_patches = [mpatches.Patch(color=c, label=l)
                  for l, c in LANG_COLOURS.items()
                  if any(lang_pair(p) == l for p in top15)]
legend_patches += [
    plt.Line2D([0], [0], color='#FFA000', linewidth=2, linestyle='--', label='τ = 0.85'),
    plt.Line2D([0], [0], color='#2E7D32', linewidth=1.5, linestyle=':', label='True clone boundary'),
]
ax.legend(handles=legend_patches, fontsize=8.5, loc='lower right',
          title='Language Pair', title_fontsize=9)

fig2.tight_layout()
fig2.savefig('ob2_fig2_top15_clones.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print('Saved → ob2_fig2_top15_clones.png')
plt.close(fig2)

# ═════════════════════════════════════════════════════════════════════════════
# Figure 3 — Language Pair Distribution Pie
# ═════════════════════════════════════════════════════════════════════════════
lp_counts = defaultdict(int)
for p in clone_pairs:
    lp_counts[lang_pair(p)] += 1

lp_labels = list(lp_counts.keys())
lp_vals   = [lp_counts[l] for l in lp_labels]
lp_cols   = [LANG_COLOURS.get(l, '#607D8B') for l in lp_labels]

fig3, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
fig3.patch.set_facecolor('#FAFAFA')

# Pie
wedges, texts, autotexts = ax_pie.pie(
    lp_vals, labels=lp_labels, colors=lp_cols,
    autopct=lambda p: f'{p:.0f}%\n({int(round(p*sum(lp_vals)/100))})',
    startangle=140,
    textprops={'fontsize': 9},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
for at in autotexts:
    at.set_fontsize(8.5)
    at.set_fontweight('bold')
ax_pie.set_title('Clone Pairs by Language Pair\n(31 total detected)',
                 fontsize=12, fontweight='bold', pad=10)

# Bar (same data)
ax_bar.set_facecolor('#F8F9FA')
sorted_pairs = sorted(zip(lp_vals, lp_labels, lp_cols), reverse=True)
vals_s, labs_s, cols_s = zip(*sorted_pairs)
bars = ax_bar.bar(range(len(labs_s)), vals_s, color=cols_s,
                  edgecolor='white', linewidth=0.7, width=0.6)
ax_bar.set_xticks(range(len(labs_s)))
ax_bar.set_xticklabels(labs_s, rotation=25, ha='right', fontsize=9)
ax_bar.set_ylabel('Number of Clone Pairs', fontsize=11)
ax_bar.set_title('Clone Count by Language Pair', fontsize=12, fontweight='bold', pad=10)
ax_bar.grid(axis='y', **GRID_KW)
for bar, val in zip(bars, vals_s):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')

fig3.suptitle('Language Distribution of Detected Clone Pairs\nOnline Boutique · Zero-Shot GraphCodeBERT',
              fontsize=13, fontweight='bold', y=1.01)
fig3.tight_layout()
fig3.savefig('ob2_fig3_language_pairs.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print('Saved → ob2_fig3_language_pairs.png')
plt.close(fig3)

# ═════════════════════════════════════════════════════════════════════════════
# Figure 4 — Precision / Recall / F1 vs Threshold
# ═════════════════════════════════════════════════════════════════════════════
fig4, ax = plt.subplots(figsize=(9, 5))
fig4.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#F8F9FA')

ax.plot(thresholds, precisions, color='#1565C0', linewidth=2.2, label='Precision')
ax.plot(thresholds, recalls,    color='#C62828', linewidth=2.2, label='Recall')
ax.plot(thresholds, f1s,        color='#2E7D32', linewidth=2.8, label='F1', zorder=5)

ax.axvline(0.85,   color='#FFA000', linewidth=2,   linestyle='--', alpha=0.9, label='Current τ = 0.85')
ax.axvline(best_t, color='#6A1B9A', linewidth=1.8, linestyle=':',  alpha=0.9,
           label=f'Best F1 τ = {best_t:.3f}')
ax.scatter([best_t], [best_f1], color='#6A1B9A', s=100, zorder=6)
ax.annotate(
    f'Best F1 = {best_f1:.3f}\nPrecision = {best_prec:.3f}\nRecall = {best_rec:.3f}',
    xy=(best_t, best_f1),
    xytext=(best_t - 0.12, best_f1 - 0.25),
    fontsize=9, color='#6A1B9A',
    arrowprops=dict(arrowstyle='->', color='#6A1B9A', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3E5F5', edgecolor='#6A1B9A', alpha=0.8)
)

ax.set_xlabel('Threshold τ', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Precision / Recall / F1 vs Threshold\n'
             'Ground truth = 3 confirmed true clones · Online Boutique',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=9.5)
ax.grid(**GRID_KW)
ax.set_xlim(0.70, 1.0)
ax.set_ylim(-0.05, 1.05)

fig4.tight_layout()
fig4.savefig('ob2_fig4_threshold_sweep.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print('Saved → ob2_fig4_threshold_sweep.png')
plt.close(fig4)

print('\nAll 4 figures saved successfully.')