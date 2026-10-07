"""Run one fixed demonstration recording and export presentation output."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from inference.predict import predict_audio

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'audio_outputs/presentation_figures'
OUT.mkdir(exist_ok=True)
with (ROOT / 'bisaya_post_class_speech_metadata.csv').open(encoding='utf-8-sig', newline='') as stream:
    first = next(csv.DictReader(stream))
result = predict_audio(str(ROOT / 'Audios' / first['Audio Filename']))
summary = {k: result[k] for k in ['valence', 'arousal', 'acoustic_summary']}
summary['sample'] = 'Demo sample A (first metadata row; training recording)'
summary['model'] = 'Saved RBF SVM models for both targets'
summary['feature_count'] = len(result['raw_features'])
assert summary['feature_count'] == 94
summary['combined_prediction'] = result['valence']['prediction'] + ' + ' + result['arousal']['prediction'].replace('Low Arousal', 'Low/Moderate Arousal')
summary['limitation'] = 'Training-recording demonstration, not independent evaluation. Probabilities are model estimates, not accuracy or verified emotional certainty.'
(OUT / '06_prototype_sample_output.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
lines = ['PROTOTYPE SAMPLE OUTPUT', 'Sample: Demo sample A (training recording)',
         'Models: RBF SVM for valence and arousal', 'Acoustic features: 94',
         f"Trimmed duration: {result['acoustic_summary']['duration_sec']} seconds",
         f"Mean pitch: {result['acoustic_summary']['pitch_f0_mean_hz']} Hz"]
for target in ['valence', 'arousal']:
    lines.append(f"{target.title()} prediction: {result[target]['prediction']}")
    for label, prob in result[target]['probabilities'].items():
        lines.append(f'  {label}: {prob:.1%}')
lines.extend(['Combined prediction: ' + summary['combined_prediction'], summary['limitation']])
(OUT / '06_prototype_sample_output.txt').write_text('\n'.join(lines), encoding='utf-8')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15,
                     'text.color': '#192C40', 'axes.labelcolor': '#192C40',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none'})
fig, axes = plt.subplots(1, 2, figsize=(16, 9))
fig.subplots_adjust(left=.17, right=.95, top=.64, bottom=.34, wspace=.65)
fig.text(.055, .92, 'Prototype demonstration: one speech recording', fontsize=27, weight='bold')
fig.text(.055, .86, 'Demo sample A • Saved RBF SVM models • Training-recording demonstration', fontsize=16, color='#556577')
fig.text(.055, .795, f"94 acoustic features     Trimmed duration: {result['acoustic_summary']['duration_sec']} s     Mean pitch: {result['acoustic_summary']['pitch_f0_mean_hz']} Hz", fontsize=17)
for ax, target, color in zip(axes, ['valence', 'arousal'], ['#CE7959', '#248C94']):
    probs = result[target]['probabilities']
    labels = [label.replace('Low Arousal', 'Low/Moderate') for label in probs]
    vals = list(probs.values())
    ax.barh(range(len(vals)), vals, color=color, height=.46)
    ax.set(yticks=range(len(vals)), yticklabels=labels, xlim=(0, 1.15),
           xticks=[0, .25, .5, .75, 1], xticklabels=['0%', '25%', '50%', '75%', '100%'],
           xlabel='Estimated class probability')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    ax.set_title(target.title() + ': ' + result[target]['prediction'].replace('Low Arousal', 'Low/Moderate'), fontsize=20, weight='bold', pad=20)
    for idx, prob in enumerate(vals):
        ax.text(1.12, idx, f'{prob:.1%}', ha='right', va='center', fontsize=16, weight='bold')
fig.text(.055, .23, 'Combined prediction: ' + summary['combined_prediction'], fontsize=24, weight='bold')
fig.text(.055, .16, 'Valence label and largest probability disagree in this run; do not interpret probabilities as prediction confidence.', fontsize=13, color='#964D2D')
fig.text(.055, .115, 'Model loading warned of a scikit-learn version mismatch: saved with 1.8.0, running with 1.9.1.', fontsize=13, color='#964D2D')
fig.text(.055, .07, 'Training-recording demonstration only. Not Pleasant includes neutral; Low/Moderate includes arousal ratings 1–3.', fontsize=12, color='#556577')
fig.text(.055, .03, 'The output illustrates prototype operation, not independent accuracy or certainty about a person’s feelings.', fontsize=12, color='#556577')
for extension in ['png', 'svg']:
    fig.savefig(OUT / f'06_prototype_sample_output.{extension}', dpi=200, facecolor='white')
plt.close(fig)
print('\n'.join(lines))
