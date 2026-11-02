#!/usr/bin/env python3
"""Check the public prose-qualification bundle without a model or tokenizer.

This checks compact copied evidence, exact text/vector bytes, every recorded
outcome and numerical statements. The source experiment's saved-array audits
remain the authority for trajectories; this does not replay their model.
"""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parent
RECEIPT='receipts/prose-qualification.json'
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify(root):
    data=json.loads((root/RECEIPT).read_text())
    vectors=json.loads((root/'receipts/prose-state-vectors.json').read_text())
    audits={name:json.loads((root/f'receipts/prose-{name}-audit.json').read_text()) for name in ('state','control')}
    article=(root/'index.md').read_text();flat=' '.join(article.replace('**','').split())
    report=(root/'prose-state-report.html').read_text()
    for name,digest in data['assets'].items():require(sha(root/name)==digest,'Asset changed: '+name)
    restored=report.replace('href="receipts/prose-qualification.json"','href="provenance.json"').replace('href="receipts/prose-state-vectors.json"','href="plotted-data.json"')
    require(hashlib.sha256(restored.encode()).hexdigest()==data['source_report']['sha256'],'HTML differs beyond documented hyperlinks')
    for name,audit in audits.items():
        require(audit['status']=='audit_passed' and audit['execution_status']=='completed',name+' audit is not a completed pass')
    rows=data['state']['rows']
    require(len(rows)==11 and [r['text_id'] for r in rows]==list(range(11)),'Full ordered prose census required')
    require(len(vectors['rows'])==11,'Full vector census required')
    audit_rows={lane['text_id']:(group,idx) for group in audits['state']['groups'] for idx,lane in enumerate(group['lanes']) if lane['kind']=='prose'}
    entries=[]
    def add(ident,text,pointer,claim):
        require(' '.join(text.split()) in flat,'Displayed evidence differs: '+ident)
        entries.append(dict(id='prose-'+ident,appears_as=text,source=RECEIPT+'#'+pointer,source_sha256=sha(root/RECEIPT),claim=claim))
    for row,vector in zip(rows,vectors['rows'],strict=True):
        require(row=={k:v for k,v in vector.items() if k!='mean'},'Compact/vector row differs')
        require(hashlib.sha256(row['text'].encode()).hexdigest()==row['text_utf8_sha256'],'Text hash differs')
        require(len(vector['mean'])==896,'Mean vector dimension differs')
        # The experiment's semantic array hash binds dtype and shape before
        # little-endian C-order bytes; it is not a bare payload-byte digest.
        header=b'float64\0[896]\0'
        require(hashlib.sha256(header+struct.pack('<896d',*vector['mean'])).hexdigest()==row['mean_sha256'],'Typed mean vector hash differs')
        group,lane=audit_rows[row['text_id']]
        require((row['bit'],row['outcome'])==(group['bits'][lane],group['lane_outcomes'][lane]),'Copied audit/row mismatch')
        require(group['instrument_valid'],'Invalid instrument cannot support a qualified row')
        require(html.escape(row['text']) in report,'Exact sentence absent from HTML')
        details=dict(token_ids=row['token_ids'],mean_sha256=row['mean_sha256'],mean=vector['mean'])
        require(html.escape(json.dumps(details,indent=2,ensure_ascii=False)) in report,'Actual vector absent from HTML')
        line='| '+row['label']+' | '+row['text'].replace('|','\\|')+' | '+str(row['bit'])+' |'
        add('row-'+row['label'],line,'/state/rows/'+str(row['text_id']),'Exact unchanged text and measured reference label; checked against copied complete audit.')
    counts=Counter(r['outcome'] for r in rows)
    expected=dict(bit0=counts['bit0'],bit1=counts['bit1'],unresolved=counts['unresolved'],instrument_invalid=counts['instrument_invalid'],not_evaluated=0)
    require(expected==data['state']['counts']==audits['state']['prose_outcome_counts'],'State counts differ')
    require(all(data['state'][key] and audits['state'][key] for key in ('reference_pair_passed','padding_control_passed','qualified_both_basin_observation')),'Control qualification differs')
    for row in data['state']['controls']:
        group=audits['state']['groups'][row['group_id']];lane=row['batch_lane']
        require((row['bit'],row['outcome'])==(group['bits'][lane],group['lane_outcomes'][lane]),'Reference/padding result differs')
    require(len(data['state']['controls'])==3,'Separate control census differs')
    control=data['control'];require(len(control['rows'])==13 and len(audits['control']['groups'])==13,'All control groups required')
    for row,audit in zip(control['rows'],audits['control']['groups'],strict=True):
        require((row['group_id'],row['kind'],row['raw_bits'],row['instrument_valid'],row['qualification'])==
                (audit['group_id'],audit['kind'],audit['bits'],audit['instrument_valid'],audit['qualification']),'Control row differs')
    prose=[r for r in control['rows'] if r['kind']=='prose']
    require(len(prose)==11 and all(r['raw_bits']==[0,1] and r['qualification']=='mixed' for r in prose),'Control preservation claim differs')
    require(control['rows'][0]['raw_bits']==[0,0] and control['rows'][1]['raw_bits']==[1,1] and control['positive_controls_passed'],'Canonical controls differ')
    require(control['counts']==dict(write0=0,write1=0,mixed=11,unresolved=0,instrument_invalid=0),'Control counts differ')
    require(data['state']['schedule']['hold_steps']==240 and data['dimension']==896,'State instrument differs')
    require(data['readout']==dict(period=6,closure_tolerance=1e-8,proper_divisors=[1,2,3],divisor_separation=1e-4,match_tolerance=1e-6),'Readout settings differ')
    add('outcomes',f"{expected['bit0']} zeros, {expected['bit1']} ones, {expected['unresolved']} unresolved, {expected['instrument_invalid']} invalid",'/state/counts','All eleven distinct text means, excluding reference/padding lanes.')
    add('state-cost',f"{data['state']['model_calls']:,} model calls in {data['state']['elapsed_seconds']:.2f} seconds",'/state','Measured complete execution including initialization, encoding and artifacts.')
    add('control-cost',f"{control['model_calls']:,} model calls in {control['elapsed_seconds']:.2f} seconds",'/control','Measured complete execution including initialization and artifacts.')
    add('state-steps','after 240 updates','/state/schedule/hold_steps','Fixed endpoint for each prose lane; not a universal convergence bound.')
    add('dimensions','896 float64 entries','/dimension','Saved raw input embedding means, independently checked by vector byte hashes.')
    for name,value in data['hold'].items():add('hold-'+name,repr(value),'/hold/'+name,'Exact fixed control.')
    add('control-census','all eleven prose-derived addresses','/control/rows','All distinct archived text controls, each with two reference starts.')
    require('separate byte test' in article and 'unfinished integration test' in article,'Separate byte-result boundary absent')
    return entries,dict(status='passed',texts=11,counts=expected,controls=3,control_groups=13,model_calls=0,scope=__doc__)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--emit',type=Path)
    args=parser.parse_args();entries,result=verify(args.root)
    if args.emit:args.emit.write_text(json.dumps(entries,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2))
