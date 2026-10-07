"""Plot existing cross-validation results without fitting or changing models."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / 'audio_outputs/tables'
OUT = ROOT / 'audio_outputs/presentation_figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14,
    'axes.titlesize': 20, 'axes.labelsize': 15, 'xtick.labelsize': 13,
    'ytick.labelsize': 14, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#BAC4CE', 'text.color': '#192C40', 'axes.labelcolor': '#192C40',
    'svg.fonttype': 'none'})

def read(name):
    with (TABLES / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

results = [read('11_model_comparison.csv'), read('15_arousal_model_comparison.csv')]
colors = ['#CE7959', '#248C94']

def save(fig, name):
    for extension in ['png', 'svg']:
        fig.savefig(OUT / f'{name}.{extension}', dpi=200, facecolor='white')
    plt.close(fig)

for metric, name, title in [
    ('Macro F1', '03_model_macro_f1', 'Model comparison: macro F1'),
    ('Accuracy', '04_model_accuracy', 'Model accuracy and majority-class baselines')]:
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.subplots_adjust(left=.19, right=.97, bottom=.24, top=.74, wspace=.72)
    fig.text(.05, .92, title, fontsize=28, weight='bold')
    fig.text(.05, .855, 'Five-fold participant-grouped cross-validation; 40 observations and 94 acoustic features', fontsize=16, color='#556577')
    for ax, rows, color, target in zip(axes, results, colors, ['Valence', 'Arousal']):
        labels = [r['Model'].replace('Dummy (Most Frequent)', 'Majority baseline') for r in rows]
        vals = np.array([float(r[f'{metric} Mean']) for r in rows])
        sd = np.array([float(r[f'{metric} SD']) for r in rows])
        baseline = next(float(r[f'{metric} Mean']) for r in rows if r['Model'].startswith('Dummy'))
        ax.barh(range(len(rows)), vals, color=[color if not r['Model'].startswith('Dummy') else '#A7B2BF' for r in rows], height=.57, alpha=.86)
        if metric == 'Macro F1':
            ax.errorbar(vals, range(len(rows)), xerr=sd, fmt='none', ecolor='#36495C', capsize=4, linewidth=1.4)
        ax.axvline(baseline, color='#64748B', linestyle='--', linewidth=1.4)
        for y, v in enumerate(vals):
            ax.text(1.10, y, f'{v:.3f}' if metric == 'Macro F1' else f'{v:.1%}', va='center', ha='right', fontsize=13, weight='bold')
        ax.set(yticks=range(len(rows)), yticklabels=labels, xlim=(0, 1.12), xticks=np.arange(0, 1.01, .2), xlabel=f'Mean {metric.lower()} across folds')
        ax.invert_yaxis()
        ax.set_title(target, loc='left', weight='bold', pad=20)
        ax.grid(axis='x', color='#E3E8ED')
        ax.set_axisbelow(True)
    if metric == 'Macro F1':
        takeaway = 'SVM leads valence; Logistic Regression leads arousal by mean macro F1.'
        note = 'Error bars: ±1 fold standard deviation, not confidence intervals. Dashed lines: target-specific majority baseline.'
    else:
        takeaway = 'Valence SVM: 55.0% vs 57.5% baseline. Arousal Logistic Regression: 72.5% vs 65.0%.'
        note = 'Models remain ordered by mean macro F1. Dashed lines: target-specific majority baseline. Accuracy bars show fold means.'
    fig.text(.05, .14, takeaway, fontsize=17, weight='bold')
    fig.text(.05, .085, note, fontsize=12, color='#556577')
    fig.text(.05, .045, 'Exploratory results: model selection uses the same folds; no independent external test set.', fontsize=12, color='#556577')
    save(fig, name)

fig, axes = plt.subplots(1, 2, figsize=(16, 9))
fig.subplots_adjust(left=.15, right=.94, bottom=.28, top=.74, wspace=.55)
fig.text(.05, .92, 'Classification errors on held-out folds', fontsize=28, weight='bold')
fig.text(.05, .855, 'Pooled out-of-fold predictions from the leading model for each target', fontsize=16, color='#556577')
for ax, filename, labels, title in zip(axes,
    ['12_confusion_matrix.csv', '16_arousal_confusion_matrix.csv'],
    [['Pleasant', 'Not Pleasant'], ['High', 'Low/Moderate']],
    ['Valence: RBF SVM', 'Arousal: Logistic Regression']):
    rows = read(filename)
    keys = list(rows[0])[1:]
    counts = np.array([[int(r[k]) for k in keys] for r in rows])
    assert counts.sum() == 40
    proportions = counts / counts.sum(axis=1, keepdims=True)
    ax.imshow(proportions, cmap='Blues', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{counts[i,j]}\n({proportions[i,j]:.1%})',
                    ha='center', va='center', fontsize=17,
                    color='white' if proportions[i,j] > .55 else '#192C40')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
           xlabel='Predicted class', ylabel='Actual class')
    ax.set_title(title, fontsize=20, weight='bold', pad=20)
    ax.tick_params(length=0, pad=12)
fig.text(.05, .145, 'Valence: 22/40 correct; 10/17 Pleasant responses missed. Arousal: 29/40 correct.', fontsize=17, weight='bold')
fig.text(.05, .088, 'Each cell shows a count and percentage within its actual class. Darker shading means a larger row percentage.', fontsize=12, color='#556577')
fig.text(.05, .045, 'Not Pleasant includes neutral valence; Low/Moderate includes arousal ratings 1–3. No independent external test set.', fontsize=12, color='#556577')
save(fig, '05_classification_confusion_matrices')
print('Exported 3 evaluation figures in PNG and SVG to', OUT)
