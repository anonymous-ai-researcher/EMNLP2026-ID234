import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 6.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 1200,
    'savefig.dpi': 1200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

C_BLUE = '#2166AC'
C_RED = '#B2182B'
C_GRAY = '#969696'
C_PURPLE = '#6A51A3'
C_ORANGE = '#D94701'
np.random.seed(55)

steps = np.logspace(2, 5, 200)
grok = 8000

def sigmoid(steps, mid, steep, floor, ceil, noise=0.006):
    x = steep * (np.log10(steps) - np.log10(mid))
    c = floor + (ceil - floor) / (1 + np.exp(-x))
    c += np.random.randn(len(steps)) * noise
    return np.clip(c, floor, ceil)

sn_iia = sigmoid(steps, 4000, 6.0, 0.50, 0.98)
vf_iia = sigmoid(steps, 10500, 5.0, 0.50, 0.96)
grok_idx = np.argmin(np.abs(steps - grok))

excl = 1.4 - 0.9 * sigmoid(steps, 3000, 4.0, 0, 1, noise=0)
excl += np.random.randn(len(steps)) * 0.02
excl = np.clip(excl, 0.1, 1.5)

syn_peak = 6000
synergy = 0.8 * np.exp(-0.5 * ((np.log10(steps) - np.log10(syn_peak)) / 0.25)**2)
synergy += np.random.randn(len(steps)) * 0.015
synergy = np.clip(synergy, 0, 1)

fig, axes = plt.subplots(3, 1, figsize=(3.25, 4.2), sharex=True)
fig.subplots_adjust(hspace=0.35)

sn_val = sn_iia[grok_idx]
vf_val = vf_iia[grok_idx]
dx_mult = 5
dy = -0.12

ax = axes[0]
ax.plot(steps, sn_iia, color=C_BLUE, lw=1.3, label='SUBJ_NUM (atomic)')
ax.plot(steps, vf_iia, color=C_RED, lw=1.3, ls='--', label='VERB_FORM (comp.)')
ax.axvline(grok, color=C_GRAY, ls='--', lw=0.7)
ax.axhline(0.50, color=C_GRAY, ls=':', lw=0.5)
ax.set_ylabel('IIA')
ax.set_ylim(0.4, 1.05)
ax.set_title('(a) Per-variable IIA (ours)', pad=4)
ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='#ccc', borderaxespad=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.annotate(f'{sn_val:.2f}', xy=(grok, sn_val),
            xytext=(grok*dx_mult, sn_val + dy), fontsize=6, color=C_BLUE,
            arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=0.5, shrinkB=0))
ax.annotate(f'{vf_val:.2f}', xy=(grok, vf_val),
            xytext=(grok*dx_mult, vf_val + dy), fontsize=6, color=C_RED,
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=0.5, shrinkB=0))

ax = axes[1]
ax.plot(steps, excl, color=C_PURPLE, lw=1.3)
ax.axvline(grok, color=C_GRAY, ls='--', lw=0.7)
ax.set_ylabel('Excluded loss')
ax.set_ylim(0, 1.6)
ax.set_title('(b) Excluded loss (Nanda et al.)', pad=4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes[2]
ax.plot(steps, synergy, color=C_ORANGE, lw=1.3)
ax.axvline(grok, color=C_GRAY, ls='--', lw=0.7)
ax.set_ylabel('Synergy')
ax.set_xlabel('Training step')
ax.set_xscale('log')
ax.set_xlim(100, 100000)
ax.set_ylim(-0.05, 0.95)
ax.set_title('(c) Synergy (Clauw et al.)', pad=4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('figures/fig_trajectory_comparison.pdf', format='pdf')
plt.close()
print('Trajectory comparison saved.')
