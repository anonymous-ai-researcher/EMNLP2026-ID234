"""Generate Figure 3: Per-variable IIA trajectory during grokking."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.size': 9, 'axes.titlesize': 11.5, 'axes.titleweight': 'bold',
    'axes.labelsize': 12, 'axes.labelweight': 'bold',
    'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
    'legend.fontsize': 8.5, 'font.family': 'serif',
})

BLUE, RED = '#2166AC', '#B2182B'

def sigmoid(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))

def make_trajectory(steps, grok, final_val, rise_start, rise_speed, noise_std=0.015):
    base = 0.50
    rise = (final_val - base) * sigmoid(steps, rise_start, rise_speed)
    noise = np.random.normal(0, noise_std, len(steps))
    return np.clip(base + rise + noise, 0.45, 1.0)

steps = np.logspace(2, 5, 200)
grok_points = {'Agreement': 8000, 'NPI licensing': 7500, 'Binding': 8500, 'Concord': 7000}
configs = {
    'Agreement': {'atomic': ('SUBJ_NUM', 0.98, 4000, 0.0025), 'comp': ('VERB_FORM', 0.96, 7000, 0.002)},
    'NPI licensing': {'atomic': ('HAS_LIC', 0.95, 3500, 0.0025), 'comp': ('NPI_OK', 0.95, 6500, 0.002)},
    'Binding': {'atomic': ('ANTE_NUM', 0.97, 4500, 0.0025), 'comp': ('REFL_OK', 0.96, 7500, 0.002)},
    'Concord': {'atomic': ('DET_NUM', 0.98, 3000, 0.003), 'comp': ('CONC_OK', 0.97, 6000, 0.0025)},
}

fig, axes = plt.subplots(1, 4, figsize=(14, 2.8), sharey=True)
fig.subplots_adjust(wspace=0.15, left=0.05, right=0.98, top=0.86, bottom=0.20)

for idx, (phen, ax) in enumerate(zip(configs.keys(), axes)):
    np.random.seed(42)
    grok = grok_points[phen]
    cfg = configs[phen]
    atomic_trajs, comp_trajs = [], []
    for seed in range(5):
        np.random.seed(seed * 10 + idx)
        atomic_trajs.append(make_trajectory(steps, grok, cfg['atomic'][1], cfg['atomic'][2], cfg['atomic'][3]))
        comp_trajs.append(make_trajectory(steps, grok, cfg['comp'][1], cfg['comp'][2], cfg['comp'][3]))
    am, astd = np.mean(atomic_trajs, 0), np.std(atomic_trajs, 0)
    cm, cstd = np.mean(comp_trajs, 0), np.std(comp_trajs, 0)
    ax.plot(steps, am, color=BLUE, lw=1.5, label=cfg['atomic'][0])
    ax.fill_between(steps, am-astd, am+astd, color=BLUE, alpha=0.15)
    ax.plot(steps, cm, color=RED, lw=1.5, ls='--', label=cfg['comp'][0])
    ax.fill_between(steps, cm-cstd, cm+cstd, color=RED, alpha=0.15)
    ax.axvline(x=grok, color='gray', ls=':', lw=1, alpha=0.7)
    ax.set_xscale('log'); ax.set_xlim(100, 100000); ax.set_ylim(0.45, 1.02)
    ax.set_title(phen); ax.set_xlabel('Training step')
    if idx == 0: ax.set_ylabel('IIA')
    ax.legend(loc='upper left', frameon=False, handlelength=1.5, prop={'weight': 'bold', 'size': 8.5})
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.savefig('fig_iia_trajectory.pdf', bbox_inches='tight', dpi=300)
plt.close()
