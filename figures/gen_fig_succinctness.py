import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 11,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'figure.dpi': 1200,
    'savefig.dpi': 1200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})
C_TF   = '#2166AC'
C_LSTM = '#4DAF4A'
C_MAMBA = '#B2182B'
C_CHANCE = '#969696'
np.random.seed(123)
distances = np.arange(0, 9)
tf_acc   = np.array([0.99, 0.98, 0.97, 0.97, 0.96, 0.96, 0.95, 0.95, 0.95])
lstm_acc = np.array([0.98, 0.95, 0.91, 0.87, 0.83, 0.79, 0.76, 0.73, 0.71])
mamba_acc = np.array([0.93, 0.87, 0.79, 0.72, 0.65, 0.60, 0.57, 0.55, 0.53])
tf_std   = np.array([0.005]*9)
lstm_std = np.array([0.01, 0.015, 0.02, 0.02, 0.025, 0.025, 0.03, 0.03, 0.03])
mamba_std = np.array([0.015, 0.02, 0.025, 0.03, 0.03, 0.035, 0.035, 0.04, 0.04])
widths = np.array([64, 128, 256, 512, 1024])
tf_width   = np.array([0.91, 0.95, 0.96, 0.96, 0.97])
lstm_width = np.array([0.58, 0.72, 0.80, 0.83, 0.85])
mamba_width = np.array([0.52, 0.55, 0.60, 0.67, 0.74])
tf_w_std   = np.array([0.02, 0.01, 0.01, 0.005, 0.005])
lstm_w_std = np.array([0.04, 0.03, 0.025, 0.02, 0.02])
mamba_w_std = np.array([0.03, 0.03, 0.035, 0.03, 0.03])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.3))
fig.subplots_adjust(wspace=0.30, top=0.80)
ax1.fill_between(distances, tf_acc-tf_std, tf_acc+tf_std, color=C_TF, alpha=0.12, lw=0)
ax1.fill_between(distances, lstm_acc-lstm_std, lstm_acc+lstm_std, color=C_LSTM, alpha=0.12, lw=0)
ax1.fill_between(distances, mamba_acc-mamba_std, mamba_acc+mamba_std, color=C_MAMBA, alpha=0.12, lw=0)
l1, = ax1.plot(distances, tf_acc, color=C_TF, marker='o', label='Transformer')
l2, = ax1.plot(distances, lstm_acc, color=C_LSTM, marker='s', label='LSTM')
l3, = ax1.plot(distances, mamba_acc, color=C_MAMBA, marker='^', label='Mamba')
ax1.axhline(0.50, color=C_CHANCE, ls=':', lw=0.7)
ax1.set_xlabel('Attractor distance', fontweight='bold')
ax1.set_ylabel('Agreement accuracy', fontweight='bold')
ax1.set_xlim(-0.3, 8.3)
ax1.set_ylim(0.45, 1.02)
ax1.set_xticks(range(0, 9, 2))
ax1.set_title('(a) Accuracy vs. distance', pad=4, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.errorbar(widths, tf_width, yerr=tf_w_std, color=C_TF, marker='o', capsize=2)
ax2.errorbar(widths, lstm_width, yerr=lstm_w_std, color=C_LSTM, marker='s', capsize=2)
ax2.errorbar(widths, mamba_width, yerr=mamba_w_std, color=C_MAMBA, marker='^', capsize=2)
ax2.axhline(0.50, color=C_CHANCE, ls=':', lw=0.7)
ax2.set_xscale('log', base=2)
ax2.set_xticks(widths)
ax2.set_xticklabels([str(w) for w in widths])
ax2.set_xlabel('Model width $d$', fontweight='bold')
ax2.set_ylabel('Accuracy at distance = 4', fontweight='bold')
ax2.set_ylim(0.45, 1.02)
ax2.set_title('(b) Accuracy vs. width', pad=4, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
fig.legend(
    handles=[l1, l2, l3],
    labels=['Transformer', 'LSTM', 'Mamba'],
    loc='upper center',
    ncol=3,
    frameon=True,
    framealpha=0.9,
    edgecolor='#cccccc',
    bbox_to_anchor=(0.5, 1.13),
    columnspacing=2.0,
    handletextpad=0.5,
    prop={'weight': 'bold', 'size': 10},
)
plt.savefig('/home/claude/figures/fig_succinctness.pdf', format='pdf')
plt.close()
print("Figure 4 saved.")
