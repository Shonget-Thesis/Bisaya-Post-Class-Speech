"""Export presentation figures from the existing acoustic dataset, without retraining."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'audio_outputs' / 'presentation_figures'
OUT.mkdir(exist_ok=True)
with (ROOT / 'audio_outputs/tables/02_acoustic_feature_dataset.csv').open(encoding='utf-8-sig', newline='') as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 40
assert len({r['Participant Code'] for r in rows}) == len(rows)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15,
                     'axes.titlesize': 21, 'axes.labelsize': 16,
                     'xtick.labelsize': 15, 'ytick.labelsize': 13,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#B4BFCA', 'text.color': '#192C40',
                     'axes.labelcolor': '#192C40', 'svg.fonttype': 'none'})
COLORS = ['#CE7959', '#9BA9BB', '#248C94']
rng = np.random.default_rng(42)

def boxes(ax, records, group_key, groups, feature, colors):
    values = [np.array([float(r[feature]) for r in records if r[group_key] == g]) for g in groups]
    assert all(len(v) and np.isfinite(v).all() for v in values)
    artists = ax.boxplot(values, positions=range(len(groups)), widths=.46,
                         patch_artist=True, showfliers=False,
                         medianprops={'color': '#192C40', 'linewidth': 2},
                         whiskerprops={'color': '#637386'}, capprops={'color': '#637386'})
    for idx, (patch, vals, color) in enumerate(zip(artists['boxes'], values, colors)):
        patch.set(facecolor=color, alpha=.23, edgecolor=color, linewidth=1.8)
        ax.scatter(idx + rng.uniform(-.13, .13, len(vals)), vals,
                   s=46, color=color, edgecolor='white', linewidth=.65, alpha=.9, zorder=3)
    ax.set_xticks(range(len(groups)), [f'{g}\n(n = {len(v)})' for g, v in zip(groups, values)])
    ax.grid(axis='y', color='#E3E8ED', linewidth=1)
    ax.set_axisbelow(True)
    return values

def save(fig, name):
    for extension in ['png', 'svg']:
        fig.savefig(OUT / f'{name}.{extension}', dpi=200, facecolor='white')
    plt.close(fig)

fig, ax = plt.subplots(figsize=(14, 7.875))
fig.subplots_adjust(left=.10, right=.96, top=.76, bottom=.23)
fig.text(.06, .92, 'Pitch across self-reported valence', fontsize=28, weight='bold')
fig.text(.06, .855, '40 post-class recordings  |  Dots show individual recordings', fontsize=16, color='#556577')
values = boxes(ax, rows, 'Valence_Label', ['Unpleasant', 'Neutral', 'Pleasant'], 'f0_mean', COLORS)
ax.set_ylabel('Mean pitch per recording (Hz)')
means = [float(v.mean()) for v in values]
ax.scatter(range(3), means, marker='D', s=95, color='#192C40', edgecolor='white', zorder=5)
for idx, mean in enumerate(means):
    ax.annotate(f'Mean {mean:.1f} Hz', (idx, mean), xytext=(20, 8),
                textcoords='offset points', fontsize=14, weight='bold')
fig.text(.06, .10, 'Higher group averages accompany pleasant ratings, with substantial overlap.', fontsize=17, weight='bold')
fig.text(.06, .054, 'Boxes: middle 50%; line: median; diamond: group mean. Pleasant includes ratings 4–5. Descriptive results only.', fontsize=12, color='#556577')
save(fig, '01_pitch_by_valence')

kapoy = [r for r in rows if r['Spoken Word'] == 'Kapoy']
assert len(kapoy) == 20
fig, axes = plt.subplots(1, 3, figsize=(16, 9))
fig.subplots_adjust(left=.075, right=.975, top=.74, bottom=.25, wspace=.36)
fig.text(.055, .92, 'Same word, different self-reports: “Kapoy”', fontsize=28, weight='bold')
fig.text(.055, .86, '20 recordings  |  12 Not Pleasant and 8 Pleasant  |  Dots show individual recordings', fontsize=16, color='#556577')
for ax, feature, title, ylabel in zip(axes, ['f0_mean', 'rms_mean', 'sc_mean'],
        ['Pitch', 'RMS energy', 'Spectral centroid'],
        ['Mean pitch (Hz)', 'Mean RMS amplitude (relative units)', 'Mean spectral centroid (Hz)']):
    boxes(ax, kapoy, 'Valence_Binary', ['Not Pleasant', 'Pleasant'], feature, [COLORS[0], COLORS[2]])
    ax.set_title(title, pad=20, weight='bold')
    ax.set_ylabel(ylabel)
    if feature == 'rms_mean':
        ax.set_ylim(bottom=0)
fig.text(.055, .145, 'Pleasant responses have higher median pitch and spectral centroid; energy overlaps.', fontsize=16, weight='bold')
fig.text(.055, .092, 'Not Pleasant includes neutral ratings (1–3). Pleasant = ratings 4–5. All observations, including extremes, are shown.', fontsize=12, color='#556577')
fig.text(.055, .05, 'Boxes: middle 50%; line: median; whiskers: values within 1.5 × IQR. Small samples and overlap limit interpretation.', fontsize=12, color='#556577')
save(fig, '02_kapoy_acoustic_comparison')
print('Pitch group means (Hz):', [round(m, 1) for m in means])
print('Exported two figures in PNG and SVG:', OUT)
