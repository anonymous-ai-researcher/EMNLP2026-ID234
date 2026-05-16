"""
Figure 1: Pipeline overview — From specification to causal verification.
Four-step flow using subject-verb agreement as running example.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'figure.dpi': 1200,
    'savefig.dpi': 1200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
})

C_BLUE = '#1f78b4'
C_RED  = '#e31a1c'
C_GREEN = '#238B45'
C_GRAY = '#969696'
C_LIGHTBLUE = '#d0e4f5'
C_LIGHTGREEN = '#d5ecd4'
C_LIGHTYELLOW = '#fef9e7'
C_LIGHTRED = '#fde0dc'

fig, ax = plt.subplots(1, 1, figsize=(7.0, 1.85))
ax.set_xlim(0, 10)
ax.set_ylim(0, 2.4)
ax.axis('off')

positions = [1.1, 3.5, 5.9, 8.3]
box_w, box_h = 1.8, 1.85
cy = 1.2

def draw_box(ax, cx, cy, w, h, color, title):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle="round,pad=0.08", linewidth=0.8,
                         edgecolor='#999999', facecolor=color)
    ax.add_patch(box)
    ax.text(cx, cy + h/2 - 0.20, title, ha='center', va='center',
            fontsize=6.5, fontweight='bold', color='#333333')

# ── Box 1: Phenomenon ──
draw_box(ax, positions[0], cy, box_w, box_h, C_LIGHTYELLOW, 'Phenomenon')
lines_phen = [
    ('The', '#777777', 6, 'normal'),
    ('keys', C_BLUE, 7, 'bold'),
    ('to the cabinet', C_GRAY, 5.5, 'italic'),
    ('are', C_GREEN, 7, 'bold'),
]
y0 = cy + 0.30
for k, (txt, col, fs, wt) in enumerate(lines_phen):
    yy = y0 - k * 0.27
    sty = 'italic' if wt == 'italic' else 'normal'
    fw = 'bold' if wt == 'bold' else 'normal'
    props = {}
    if wt == 'bold' and col == C_BLUE:
        props = dict(bbox=dict(boxstyle='round,pad=0.12', fc=C_LIGHTBLUE, ec=C_BLUE, lw=0.5))
    elif wt == 'bold' and col == C_GREEN:
        props = dict(bbox=dict(boxstyle='round,pad=0.12', fc=C_LIGHTGREEN, ec=C_GREEN, lw=0.5))
    ax.text(positions[0], yy, txt, ha='center', va='center',
            fontsize=fs, fontweight=fw, fontstyle=sty, color=col, **props)

# ── Box 2: B-RASP Program ──
draw_box(ax, positions[1], cy, box_w, box_h, C_LIGHTBLUE, 'B-RASP Program')
code = [
    ('SUBJ_NUM(i) :=',              C_BLUE,    'bold',   0.32),
    ('  $\\triangleleft_j$[j<i, noun]',     '#333333', 'normal', 0.12),
    ('  $Q_{\\mathrm{pl}}$(j) : 0',         '#333333', 'normal', -0.04),
    ('VERB_FORM(i) :=',             C_BLUE,    'bold',   -0.22),
    ('  SUBJ_NUM $\\leftrightarrow$ $Q_v$', '#333333', 'normal', -0.38),
]
for txt, col, fw, dy in code:
    ax.text(positions[1] - 0.78, cy + dy, txt, ha='left', va='center',
            fontsize=5.5, fontweight=fw, color=col, family='monospace')

# ── Box 3: Compiled SCM ──
draw_box(ax, positions[2], cy, box_w, box_h, C_LIGHTGREEN, 'Compiled SCM')
nx_sn, ny_sn = positions[2], cy - 0.03
nx_vf, ny_vf = positions[2], cy - 0.42

# Input nodes - symmetric about center
ax.text(positions[2] - 0.45, cy + 0.30, '$w$', ha='center', va='center', fontsize=7,
        color=C_GRAY,
        bbox=dict(boxstyle='circle,pad=0.12', fc='white', ec=C_GRAY, lw=0.5))
ax.text(positions[2] + 0.45, cy + 0.30, '$Q_v$', ha='center', va='center', fontsize=6,
        color=C_GRAY,
        bbox=dict(boxstyle='circle,pad=0.10', fc='white', ec=C_GRAY, lw=0.5))

# Endogenous nodes - centered vertically
ax.text(nx_sn, ny_sn, 'S_N', ha='center', va='center', fontsize=6,
        fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.16', fc=C_BLUE, ec=C_BLUE, lw=0.6))
ax.text(nx_vf, ny_vf, 'V_F', ha='center', va='center', fontsize=6,
        fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.16', fc=C_RED, ec=C_RED, lw=0.6))

# DAG arrows
ax.annotate('', xy=(nx_sn - 0.05, ny_sn + 0.18),
            xytext=(positions[2] - 0.43, cy + 0.30 - 0.16),
            arrowprops=dict(arrowstyle='->', color='#666666', lw=0.7))
ax.annotate('', xy=(nx_vf, ny_vf + 0.18),
            xytext=(nx_sn, ny_sn - 0.18),
            arrowprops=dict(arrowstyle='->', color='#666666', lw=0.7))
ax.annotate('', xy=(nx_vf + 0.05, ny_vf + 0.18),
            xytext=(positions[2] + 0.43, cy + 0.30 - 0.16),
            arrowprops=dict(arrowstyle='->', color='#666666', lw=0.7))

# ── Box 4: DAS Verification ──
draw_box(ax, positions[3], cy, box_w, box_h, C_LIGHTRED, 'DAS Verification')
for k, (lbl, col, var) in enumerate([('Layer 1', C_BLUE, 'S_N'),
                                       ('Layer 2', C_RED,  'V_F')]):
    yy = cy + 0.25 - k * 0.38
    rect = FancyBboxPatch((positions[3] - 0.70, yy - 0.12), 1.40, 0.24,
                          boxstyle="round,pad=0.04", linewidth=0.5,
                          edgecolor='#bbbbbb', facecolor='white')
    ax.add_patch(rect)
    ax.text(positions[3] - 0.55, yy, lbl, ha='left', va='center', fontsize=5.5, color='#555555')
    ax.plot(positions[3] + 0.25, yy, 's', color=col, markersize=5, zorder=5)
    ax.text(positions[3] + 0.42, yy, var, ha='left', va='center', fontsize=5.5,
            fontweight='bold', color=col)

# IIA score
ax.text(positions[3], cy - 0.50, 'IIA = .98 / .96', ha='center', va='center',
        fontsize=6.5, fontweight='bold', color=C_GREEN,
        bbox=dict(boxstyle='round,pad=0.12', fc='white', ec=C_GREEN, lw=0.6))

# ── Arrows between boxes ──
labels = ['formalize', 'compile', 'verify']
for i in range(3):
    x1 = positions[i] + box_w / 2 + 0.02
    x2 = positions[i + 1] - box_w / 2 - 0.02
    ax.annotate('', xy=(x2, cy), xytext=(x1, cy),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2))
    ax.text((x1 + x2) / 2, cy + 0.14, labels[i], ha='center', va='center',
            fontsize=5.5, color='#555555', fontstyle='italic')

plt.savefig('/home/claude/figures/fig_pipeline.pdf', format='pdf')
plt.close()
print("Figure 1 (pipeline) saved.")
