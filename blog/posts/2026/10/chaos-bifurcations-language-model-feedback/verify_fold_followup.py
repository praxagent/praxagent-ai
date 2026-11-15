"""Check the return-test follow-up (FGR4, FGR4b, FGR5) in both blog posts against its receipt.

Dependency-free so the blog CI can run it without research arrays: receipt hashes, figure bytes, and every bound
number re-rendered from its receipt field with its stated rule and found in the prose.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHORT = HERE.parent/'language-model-feedback-short-read'
EXT = 'fold-return-followup-20261114'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prose(path):
    return ' '.join(Path(path).read_text().split())


def render(v, rule):
    if rule == 'int':
        return str(int(v))
    if rule == 'int:comma':
        return f'{int(v):,}'
    kind, n = rule.split(':'); n = int(n)
    if kind == 'dp':
        return f'{v:.{n}f}'
    m, e = f'{v:.{n}e}'.split('e')
    return f'{m}\\times10^{{{int(e)}}}'


def check(directory):
    receipt = directory/'fold-return-followup.json'
    r = json.loads(receipt.read_text())
    m = json.loads((directory/'provenance.json').read_text())
    ext = next(e for e in m['extensions'] if e['id'] == EXT)
    for name, h in ext['receipts'].items():
        assert sha(directory/name) == h, name
    for name, bound in r['figures'].items():
        if name.endswith('.png') and (directory/name).exists():
            assert (directory/name).stat().st_size == bound['bytes'] and sha(directory/name) == bound['sha256'], name
    text = prose(directory/'index.md')
    for item in ext['new_numbers']:
        v = r
        for k in item['field']:
            v = v[k]
        assert v == item['value'], item['id']
        assert render(v, item['rule']) == item['shown'], item['id']
        assert item['appears_as'] in text, item['id']
    s = r['numbers']['score']
    assert s['C1_escape_rate_passed'] and s['C2_calibration_passed'] and s['C3_pointwise_long_runs']['passed']
    assert s['prediction_sha256'] == r['prediction']['sha256']
    return len(ext['new_numbers'])


if __name__ == '__main__':
    assert sha(HERE/'fold-return-followup.json') == sha(SHORT/'fold-return-followup.json')
    n = check(HERE), check(SHORT)
    assert 'id="ref-grebogi-1983b"' in (HERE/'index.md').read_text() and '{#return-followup}' in (HERE/'index.md').read_text()
    print(f'FGR3 follow-up: receipt, figure bytes and {sum(n)} bound numbers verified in both blog posts.')
