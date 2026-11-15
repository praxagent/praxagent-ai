"""Check the transient-to-cycle figures (FGR5 series, twin demonstration) in both blog posts against their receipt.

Dependency-free: receipt hashes, figure bytes, and every bound number re-rendered from its receipt field with its
stated rule and found in the prose.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHORT = HERE.parent/'language-model-feedback-short-read'
EXT = 'fold-transient-figures-20261114'
WORDS = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(v, rule):
    if rule == 'int':
        return str(int(v))
    if rule == 'int:comma':
        return f'{int(v):,}'
    if rule == 'word':
        return WORDS[int(v)]
    kind, n = rule.split(':'); n = int(n)
    if kind == 'dp':
        return f'{v:.{n}f}'
    m, e = f'{v:.{n}e}'.split('e')
    return f'{m}\\times10^{{{int(e)}}}'


def check(directory):
    r = json.loads((directory/'fold-transient-figures.json').read_text())
    m = json.loads((directory/'provenance.json').read_text())
    ext = next(e for e in m['extensions'] if e['id'] == EXT)
    for name, h in ext['receipts'].items():
        assert sha(directory/name) == h, name
    for name, bound in r['figures'].items():
        if name.endswith('.png') and (directory/name).exists():
            assert (directory/name).stat().st_size == bound['bytes'] and sha(directory/name) == bound['sha256'], name
    text = ' '.join((directory/'index.md').read_text().split())
    for item in ext['new_numbers']:
        v = r
        for k in item['field']:
            v = v[k]
        assert v == item['value'], item['id']
        assert render(v, item['rule']) == item['shown'], item['id']
        assert item['appears_as'] in text, item['id']
    n = r['numbers']
    assert n['fresh_captured_within_1e-3']+n['fresh_still_wandering_at_end'] == n['fresh_trajectories']
    assert n['fresh_rows_below_1e-12_at_stop'] == n['fresh_captured_within_1e-3']
    return len(ext['new_numbers'])


if __name__ == '__main__':
    assert sha(HERE/'fold-transient-figures.json') == sha(SHORT/'fold-transient-figures.json')
    n = check(HERE), check(SHORT)
    print(f'Transient figures: receipt, figure bytes and {sum(n)} bound numbers verified in both blog posts.')
