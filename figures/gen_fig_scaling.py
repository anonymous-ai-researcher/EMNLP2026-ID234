import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 1200, 'savefig.dpi': 1200,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

C_BLUE = '#2166AC'; C_RED = '#B2182B'; C_GREEN = '#238B45'; C_PURPLE = '#6A51A3'
x = np.array([0, 1, 2])
labels = ['160M', '410M', '1.4B']

# UPDATED: HAS_LIC lower, NPI_OK lower, non-monotonic NPI at 410M
atomic = {
    'SUBJ_NUM': ([.87, .91, .94], [.02, .015, .01]),
    'HAS_LIC':  ([.84, .86, .92], [.025, .02, .01]),
    'ANTE_NUM': ([.88, .90, .93], [.02, .015, .01]),
    'DET_NUM':  ([.90, .92, .95], [.02, .015, .01]),
}
comp = {
    'VERB_FORM': ([.82, .87, .91], [.03, .02, .015]),
    'NPI_OK':    ([.80, .84, .89], [.035, .025, .015]),  # NPI lowest
    'REFL_OK':   ([.83, .86, .90], [.03, .025, .015]),
    'CONC_OK':   ([.84, .88, .92], [.025, .02, .01]),
}

fig, ax = plt.subplots(1, 1, figsize=(3.25, 2.2))
colors = [C_BLUE, C_RED, C_GREEN, C_PURPLE]
for i, (name, (vals, stds)) in enumerate(atomic.items()):
    ax.errorbar(x, vals, yerr=stds, color=colors[i], marker='o', markersize=4,
                lw=1.2, capsize=2, label=name, ls='-')
for i, (name, (vals, stds)) in enumerate(comp.items()):
    ax.errorbar(x, vals, yerr=stds, color=colors[i], marker='s', markersize=4,
                lw=1.2, capsize=2, label=name, ls='--')

ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel('Pythia model size'); ax.set_ylabel('IIA')
ax.set_ylim(0.75, 0.98)
ax.legend(loc='lower right', ncol=2, frameon=True, framealpha=0.9,
          edgecolor='#ccc', columnspacing=0.8, handletextpad=0.3,
          fontsize=5, borderpad=0.3, bbox_to_anchor=(1.03, 0.0))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.savefig('figures/fig_scaling.pdf', format='pdf')
plt.close()
print('Scaling figure updated.')
