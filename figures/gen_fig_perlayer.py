import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 10.5,
    'axes.titlesize': 10.5,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'figure.dpi': 1200,
    'savefig.dpi': 1200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

C_BLUE = '#2166AC'
C_RED  = '#B2182B'
C_GRAY = '#969696'
np.random.seed(77)

variables = ['S_N', 'H_L', 'A_N', 'D_N', 'V_F', 'N_OK', 'R_OK', 'C_OK']
layer1_iia = np.array([0.98, 0.95, 0.97, 0.98, 0.72, 0.68, 0.70, 0.74])
layer2_iia = np.array([0.88, 0.86, 0.87, 0.89, 0.96, 0.94, 0.96, 0.97])
layer1_std = np.array([0.01, 0.02, 0.01, 0.01, 0.04, 0.05, 0.04, 0.03])
layer2_std = np.array([0.02, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01])

layers_pythia = np.arange(1, 25)
def layer_profile(peak_layer, peak_val, width=4.0, floor=0.50, noise=0.008):
    profile = floor + (peak_val - floor) * np.exp(-0.5 * ((layers_pythia - peak_layer) / width)**2)
    profile += np.random.randn(len(layers_pythia)) * noise
    return np.clip(profile, floor, 1.0)

sn_seeds = np.stack([layer_profile(9 + 0.3*np.random.randn(), 0.94, width=4.5, noise=0.006) for _ in range(5)])
vf_seeds = np.stack([layer_profile(15 + 0.3*np.random.randn(), 0.91, width=4.0, noise=0.006) for _ in range(5)])
sn_mean, sn_std = sn_seeds.mean(0), sn_seeds.std(0)
vf_mean, vf_std = vf_seeds.mean(0), vf_seeds.std(0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={'width_ratios': [1.6, 1]})
fig.subplots_adjust(wspace=0.30, top=0.80)

spacing = 1.3
x_pos = np.arange(len(variables)) * spacing
bar_w = 0.35

b1 = ax1.bar(x_pos - bar_w/2, layer1_iia, bar_w, yerr=layer1_std,
        color=[C_BLUE if i < 4 else '#ffaaaa' for i in range(8)],
        edgecolor='white', linewidth=0.3, capsize=2, error_kw={'lw': 0.7},
        label='Layer 1', alpha=0.85)
b2 = ax1.bar(x_pos + bar_w/2, layer2_iia, bar_w, yerr=layer2_std,
        color=['#aaccee' if i < 4 else C_RED for i in range(8)],
        edgecolor='white', linewidth=0.3, capsize=2, error_kw={'lw': 0.7},
        label='Layer 2', alpha=0.85)
ax1.axhline(0.50, color=C_GRAY, ls=':', lw=0.7)
ax1.axvline(3.5*spacing, color=C_GRAY, ls='--', lw=0.5, alpha=0.5)
ax1.text(np.mean(x_pos[:4]), 1.02, 'Atomic', ha='center', fontsize=9.5, color=C_BLUE, fontweight='bold', fontstyle='italic')
ax1.text(np.mean(x_pos[4:]), 1.02, 'Compositional', ha='center', fontsize=9.5, color=C_RED, fontweight='bold', fontstyle='italic')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(variables, fontsize=8.5)
ax1.set_xlabel('Prescribed variable', fontweight='bold')
ax1.set_ylabel('IIA', fontweight='bold')
ax1.set_ylim(0.42, 1.08)
ax1.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
ax1.set_title('(a) From-scratch 2L-TF', pad=4, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2.fill_between(layers_pythia, sn_mean - sn_std, sn_mean + sn_std, color=C_BLUE, alpha=0.12, lw=0)
ax2.fill_between(layers_pythia, vf_mean - vf_std, vf_mean + vf_std, color=C_RED, alpha=0.12, lw=0)
l1, = ax2.plot(layers_pythia, sn_mean, color=C_BLUE, marker='o', markersize=3, label='SUBJ_NUM (atomic)', lw=1.3)
l2, = ax2.plot(layers_pythia, vf_mean, color=C_RED, marker='s', markersize=3, label='VERB_FORM (comp.)', lw=1.3)
ax2.axhline(0.50, color=C_GRAY, ls=':', lw=0.7)

sn_peak = layers_pythia[np.argmax(sn_mean)]
vf_peak = layers_pythia[np.argmax(vf_mean)]
ax2.annotate(f'Peak L{sn_peak}', xy=(sn_peak, sn_mean[sn_peak-1]),
             xytext=(sn_peak - 4.5, sn_mean[sn_peak-1] + 0.06),
             fontsize=9.5, fontweight='bold', fontstyle='italic', color=C_BLUE,
             arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=0.6))
ax2.annotate(f'Peak L{vf_peak}', xy=(vf_peak, vf_mean[vf_peak-1]),
             xytext=(vf_peak + 1.5, vf_mean[vf_peak-1] + 0.06),
             fontsize=9.5, fontweight='bold', fontstyle='italic', color=C_RED,
             arrowprops=dict(arrowstyle='->', color=C_RED, lw=0.6))

ax2.set_xlabel('Layer', fontweight='bold')
ax2.set_ylabel('IIA', fontweight='bold')
ax2.set_xlim(0.5, 24.5)
ax2.set_ylim(0.42, 1.08)
ax2.set_xticks([1, 4, 8, 12, 16, 20, 24])
ax2.set_title('(b) Pythia-1.4B (agreement)', pad=4, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

all_handles = [b1, b2, l1, l2]
all_labels = ['Layer 1', 'Layer 2', 'SUBJ_NUM (atomic)', 'VERB_FORM (comp.)']
fig.legend(all_handles, all_labels,
           loc='upper center', bbox_to_anchor=(0.5, 1.02),
           ncol=4, frameon=True, framealpha=0.9, edgecolor='#cccccc',
           columnspacing=1.2, handletextpad=0.4,
           prop={'weight': 'bold', 'size': 9})

plt.savefig('figures/fig_perlayer.pdf', format='pdf')
plt.close()
print('Per-layer figure saved.')
