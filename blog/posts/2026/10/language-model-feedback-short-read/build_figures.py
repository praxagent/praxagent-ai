"""Rebuild the short read's teaching figures from its released local receipt.

Run: python build_figures.py [--verify]
The measured chart is derived from receipt values; the loop is a schematic.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCE_COMMIT = '1314528118ac1300c2cffd34fb60cfbfc67d2c82'
RECEIPT_SHA = '6ab6ec4f46f19dcc5e9d57c5010efcb1ce500b942ddaee29b3bf80cc30e5be23'
IMAGE_SHA = 'e98f580388a13acb1400b4082b29df7e46e7c6bdd08b8777acbbe67989344eb1'
sha = lambda data: hashlib.sha256(data).hexdigest()


def generate():
    source = (ROOT / 'receipts/repro-summary.json').read_bytes()
    assert sha(source) == RECEIPT_SHA
    image = (ROOT / 'islands-of-order.png').read_bytes()
    assert sha(image) == IMAGE_SHA
    raw = json.loads(source)['trajectories']['qwen0.5b steered (REPRO1)']['batch64']
    rows = []
    for key, label in [('chaotic', 'Chaotic loops'), ('periodic', 'Repeating loops')]:
        row = raw[key]
        exact = row['cells'] * row['end_with_different_tokens']
        count = round(exact)
        assert abs(exact - count) < 1e-10
        rows.append({'regime': key, 'label': label, 'changed': count,
                     'total': row['cells'], 'percent': 100 * count / row['cells']})
    assert [(x['changed'], x['total']) for x in rows] == [(173, 192), (7, 192)]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'svg.hashsalt': 'chaos-short-read-v1', 'font.size': 15,
                         'text.color': '#2C2924', 'axes.labelcolor': '#2C2924',
                         'xtick.color': '#5A544C', 'ytick.color': '#5A544C'})
    fig, ax = plt.subplots(figsize=(7.2, 4.7), facecolor='#F7F4F0')
    fig.subplots_adjust(left=.07, right=.95, bottom=.20, top=.82)
    ax.set_facecolor('#F7F4F0')
    for y, row, color in zip([1.2, .25], rows, ['#A67C52', '#4B6787']):
        ax.barh(y, 100, height=.29, color='#EDE8E1', zorder=1)
        ax.barh(y, row['percent'], height=.29, color=color, zorder=3)
        ax.text(0, y+.23, row['label'], va='bottom', weight='bold', fontsize=17)
        ax.text(100, y+.23, f"{row['changed']} of {row['total']} ({row['percent']:.1f}%)",
                va='bottom', ha='right', fontsize=14)
    ax.set_xlim(0, 100); ax.set_ylim(-.15, 1.85); ax.set_yticks([])
    ax.set_xticks([0, 50, 100], ['0%', '50%', '100%'])
    ax.tick_params(axis='x', length=0, labelsize=15, pad=10)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_xlabel('Settings with a changed token', fontsize=16, labelpad=12)
    fig.text(.07, .94, 'When rounding changes the words', fontsize=21, weight='bold')
    fig.text(.07, .875, 'Only the batch layout changed · high-precision calculation', fontsize=12)
    outputs = {'islands-of-order.png': image}
    for ext in ('svg', 'png', 'pdf'):
        buf = io.BytesIO()
        meta = {'Date': None} if ext == 'svg' else ({'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {})
        fig.savefig(buf, format=ext, dpi=300, metadata=meta, facecolor=fig.get_facecolor())
        data = buf.getvalue()
        if ext == 'svg':
            data = data.replace(b'<svg ', b'<svg role="img" aria-labelledby="title desc" ', 1)
            pos = data.index(b'>', data.index(b'<svg ')) + 1
            data = data[:pos] + b'<title id="title">Batch layout changed the displayed tokens</title><desc id="desc">In high-precision Qwen2.5-0.5B feedback runs, tokens changed at 173 of 192 chaotic settings and 7 of 192 periodic settings. Both bars share a zero-to-100-percent axis.</desc>' + data[pos:]
        outputs['batching-comparison.'+ext] = data
    plt.close(fig)
    outputs['feedback-loop.svg'] = '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="700" viewBox="0 0 360 350" role="img" aria-labelledby="title desc">
<title id="title">The model feeds a blend back into one input slot</title>
<desc id="desc">The fixed prompt and replaceable input enter an unchanged language model. Its next-token probabilities weight a blend of token representations. This blend returns to the same input slot. The leading token is displayed but is not the full feedback signal. This is a conceptual diagram, not data.</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="#A67C52"/></marker></defs>
<rect width="360" height="350" fill="#F7F4F0"/>
<style>.t{font-family:Inter,Arial,Helvetica,sans-serif;fill:#2C2924}.title{font-size:17px;font-weight:700}.h{font-size:14px;font-weight:700}.s{font-size:13px}.m{font-size:12px;fill:#5A544C}.a{fill:none;stroke:#A67C52;stroke-width:2;marker-end:url(#arrow)}</style>
<text class="t title" x="22" y="29">How the feedback loop works</text>
<text class="t m" x="22" y="50">The model's trained weights stay fixed.</text>
<rect x="22" y="70" width="281" height="55" rx="10" fill="#FBF9F6" stroke="#C4B8A8"/>
<text class="t h" x="37" y="92">Fixed prompt + one input slot</text>
<text class="t s" x="37" y="112">The previous blend is replaced.</text>
<path class="a" d="M162 125V148"/>
<rect x="22" y="150" width="281" height="55" rx="10" fill="#E8F0F7" stroke="#4B6787"/>
<text class="t h" x="37" y="173">Language model</text>
<text class="t s" x="37" y="193">Probabilities for possible next tokens</text>
<path class="a" d="M162 205V228"/>
<rect x="22" y="230" width="281" height="55" rx="10" fill="#F3E8E0" stroke="#A67C52"/>
<text class="t h" x="37" y="253">Make a probability-weighted blend</text>
<text class="t s" x="37" y="273">Feed it back into the same input slot.</text>
<path class="a" d="M303 258H330V97H305"/>
<rect x="22" y="307" width="316" height="27" rx="7" fill="#EAF1E5"/>
<text class="t s" x="33" y="325">The displayed word is only the top candidate.</text>
</svg>'''.encode()
    for name, data in outputs.items():
        if name.endswith('.svg'): ET.fromstring(data)
    fig_receipt = {'source_commit': SOURCE_COMMIT, 'source_path': 'docs/repro-summary.json',
        'source_sha256': RECEIPT_SHA, 'rows': rows,
        'scope': 'Qwen2.5-0.5B, float64; changed leading tokens in the final 500 of 1500 steps. Selected settings, not chatbot prevalence.',
        'map': {'original_bundle_file': 'chaos-bifurcations-language-model-feedback/p6-shrimp-lead.png',
                'sha256': IMAGE_SHA, 'raw_source': 'llm-soft-loop/pod64/hr10/scan.npz',
                'raw_sha256': 'b29668a41c6d906dee03c3268b62929f91e89b0f1119d15d588d8223c6da93a0',
                'operation': 'byte-identical copy'},
        'schematic': 'Conceptual only; not measured data.',
        'matplotlib_version': matplotlib.__version__,
        'provenance': {'outputs': {k: sha(v) for k,v in outputs.items()}}}
    outputs['fig-short-read.receipt.json'] = (json.dumps(fig_receipt, indent=2)+'\n').encode()
    numbers = [{'id':'chaotic-changed','appears_as':'173 of 192','source':'receipts/repro-summary.json; trajectories/qwen0.5b steered (REPRO1)/batch64/chaotic; cells times end_with_different_tokens'},
               {'id':'periodic-changed','appears_as':'7 of 192','source':'same receipt, periodic; cells times end_with_different_tokens'},
               {'id':'measurement-window','appears_as':'500','source':'Full report, batch-measurement: final 500 steps'},
               {'id':'measurement-horizon','appears_as':'1,500','source':'Full report, batch-measurement: 1500 steps'}]
    provenance = {'local_bundle': True, 'source_commit': SOURCE_COMMIT,
        'generator': {'path':'build_figures.py','sha256':sha(Path(__file__).read_bytes())},
        'receipts': {'receipts/repro-summary.json': RECEIPT_SHA,
                     'fig-short-read.receipt.json': sha(outputs['fig-short-read.receipt.json'])},
        'figures':['islands-of-order.png','feedback-loop.svg','batching-comparison.svg','batching-comparison.png','batching-comparison.pdf'],
        'numbers': numbers,
        'editorial_scope': 'Short companion to the existing report, with no new research claim. Detailed methods, prior work and evidence grading are linked rather than repeated.'}
    # The visual companion reuses frozen figures rather than redrawing data.
    evidence_bytes = (ROOT / 'receipts/evidence-summary.json').read_bytes()
    evidence = json.loads(evidence_bytes)
    assert evidence['source_commit'] == SOURCE_COMMIT
    provenance['receipts']['receipts/evidence-summary.json'] = sha(evidence_bytes)
    for filename, receipt in evidence['figures'].items():
        data = (ROOT / filename).read_bytes()
        assert sha(data) == receipt['provenance']['sha256'], filename
        outputs[filename] = data
        receipt_name = filename.replace('.png', '.receipt.json')
        receipt = {**receipt, 'provenance': {**receipt['provenance'], 'outputs': {filename: sha(data)}}}
        outputs[receipt_name] = (json.dumps(receipt, indent=2)+'\n').encode()
        provenance['receipts'][receipt_name] = sha(outputs[receipt_name])
        provenance['figures'].append(filename)
    provenance['numbers'].extend([
        {'id':'hidden-text-period', 'appears_as':'seven-token', 'source':'fig-hidden-bands.receipt.json; original Figure 39 caption'},
        {'id':'hidden-text-horizon', 'appears_as':'2,000', 'source':'fig-hidden-bands.receipt.json; original Figure 39 caption'},
    ])
    provenance['figure_sha256'] = {name: sha(outputs[name]) for name in provenance['figures']}
    provenance['source_evidence'] = 'receipts/evidence-summary.json'
    provenance['editorial_scope'] = 'Visual short read selecting existing findings, controls and limitations from the pinned report. Figures are byte-identical or generated from the released batch receipt. No new experiments.'
    outputs['provenance.json'] = (json.dumps(provenance, indent=2)+'\n').encode()
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    for name, data in generate().items():
        path = ROOT / name
        if args.verify:
            assert path.read_bytes() == data, name
        else:
            path.write_bytes(data)
    print('Verified all figure bytes and derived counts.' if args.verify else 'Built all figures and provenance.')
