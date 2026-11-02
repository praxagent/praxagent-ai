#!/usr/bin/env python3
"""Verify the versioned public calibrated-byte presentation without a model.

This checks copied complete audits, all eight measured pairs, carry bindings,
byte/checksum arithmetic, exact report assets, plotting arrays and displayed
claims. It does not replay the Transformer, tokenizer, or raw trajectory audit.
The unmodified archived generator requires the original full raw case for its
stronger --verify operation; this page contains only compact presentation data.
"""
import argparse, hashlib, html, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
BUNDLE='public-prose-byte-v1'
PUBLICATION='receipts/public-prose-byte-publication-v1.json'
def require(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def asset(root,name):
    path=root/name
    if not path.exists() and name.endswith('report.html'):path=Path(str(path)+'.txt')
    return path
def verify(root):
    base=root/BUNDLE
    pub=json.loads((root/PUBLICATION).read_text())
    for name,row in pub['assets'].items():
        p=asset(root,name);require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Bound publication asset differs: '+name)
    for name,row in pub['correction_receipts'].items():
        path=root/'receipts'/name;require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Versioned correction receipt differs: '+name)
    receipt=json.loads((base/'receipt.json').read_text());data=json.loads((base/'data.json').read_text())
    for name,row in receipt['outputs'].items():
        p=asset(base,name);require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Original report output differs: '+name)
    require(sha(base/'report_public_prose_byte.py')==receipt['generator']['sha256'],'Original report generator differs')
    audit=json.loads((base/'scientific-audit.json').read_text());ind=json.loads((base/'independent-audit.json').read_text())
    require(sha(base/'scientific-audit.json')==receipt['scientific_audit_sha256']==data['scientific_audit_sha256']==ind['scientific_audit_sha256'],'Scientific audit identity')
    require(sha(base/'independent-audit.json')==receipt['independent_review']['sha256']==data['independent_review']['sha256'],'Independent audit identity')
    require(audit['status']=='audit_passed' and audit['execution_status']=='completed' and audit['completed_groups']==8,'Completed scientific audit gate')
    require(ind['status']=='passed' and not ind['model_loaded'] and ind['return_files']==410,'Independent complete-return gate')
    require(ind['execution_commit']==data['source_commit']==receipt['source_commit'],'Source identity')
    require(ind['return_manifest_sha256']==data['return_manifest_sha256'],'Full-return identity')
    rows=data['rows'];require(len(rows)==len(audit['groups'])==len(ind['groups'])==8,'All eight groups required')
    require(data['initializations']==1 and len(data['stage_carries'])==16,'Sole initialization and all stages')
    for i,stage in enumerate(data['stage_carries']):
        require(stage['stage_id']==i and stage['literal_previous_state_equal'],'Literal carry gate')
        if i:require(stage['initial_raw_sha256']==data['stage_carries'][i-1]['terminal_raw_sha256'],'Carry boundary hash differs')
    anchors=data['anchor_texts'];require(len(set(anchors))==2,'Finite anchor census')
    report=asset(base,'report.html').read_text();article=(root/('index.md' if (root/'index.md').exists() else 'article-source-v1.txt')).read_text();flat=' '.join(article.replace('**','').split())
    entries=[]
    def add(ident,display,pointer,claim,source=BUNDLE+'/data.json'):
        require(' '.join(display.split()) in flat,'Displayed publication value differs: '+ident)
        entries.append(dict(id='calibrated-byte-'+ident,appears_as=display,source=source+'#'+pointer,source_sha256=sha(root/source),claim=claim))
    for i,(row,observed,independent) in enumerate(zip(rows,audit['groups'],ind['groups'],strict=True)):
        require(row['index']==i+1 and row['text'] in anchors,'Carrier order/text contract')
        require(hashlib.sha256(row['text'].encode()).hexdigest()==row['text_sha256'],'Exact carrier hash')
        require(row['text']==independent['text'] and row['bits']==observed['bits']==independent['actual_pair_bits'],'Both copied audits disagree with report row')
        require(observed['instrument_valid'] and row['qualification']==observed['qualification'],'Invalid group')
        require(row['pulse_updates']==120 and row['hold_updates']==240 and row['update_range']==[360*i+1,360*(i+1)],'Pulse/hold schedule')
        require(row['controls_hex']=={k:v.hex() for k,v in row['controls'].items()},'Computed binary64 control bytes')
        require(html.escape(row['text']) in report,'Exact carrier missing from report')
        label='C'+str(anchors.index(row['text']))
        add('row-'+str(i+1),f"| {i+1} | {label} | {row['bits'][0]} | {row['bits'][1]} |",'/rows/'+str(i),'Both actual measured labels; no requested-bit substitution.')
    bits=[''.join(str(row['bits'][lane]) for row in rows) for lane in (0,1)]
    require(all(len(b)==8 and set(b)<=set('01') for b in bits),'All measured bits resolved')
    payload=[bytes([int(b,2)]) for b in bits]
    decoded=dict(status='delivered',lane_bytes_hex=[p.hex() for p in payload],lanes_agree=payload[0]==payload[1],checksum_passed=all(hashlib.sha256(p).hexdigest()==data['expected_checksum'] for p in payload),lane_checksum_passed=[hashlib.sha256(p).hexdigest()==data['expected_checksum'] for p in payload])
    require(decoded==data['decoded']==audit['decoded']==ind['decoded'] and decoded['lanes_agree'] and decoded['checksum_passed'],'Measured byte/checksum differs')
    with np.load(base/'plot-data.npz',allow_pickle=False) as arrays:
        require(set(arrays.files)=={'update_index','distance'},'Plot array fields')
        require(arrays['update_index'].dtype==np.int64 and np.array_equal(arrays['update_index'],np.arange(2881)),'All update indices')
        distance=arrays['distance'];require(distance.shape==tuple(data['distance_shape'])==(2881,2,2) and distance.dtype==np.float64 and np.isfinite(distance).all() and (distance>=0).all(),'Distance array contract')
        require(hashlib.sha256(distance.tobytes()).hexdigest()==data['distance_array_sha256'],'Distance payload bytes')
    require(data['call_counts']==dict(prefix=16,comparison=32,update=2880) and sum(data['call_counts'].values())==data['model_calls'],'Call census')
    require(data['model_calls']==ind['experiment_model_calls'] and data['elapsed_seconds']==ind['experiment_elapsed_seconds'] and data['peak_rss_bytes']==ind['experiment_peak_rss_bytes'],'Costs differ from independent return')
    add('bits',bits[0],'/rows','Eight measured bits in most-significant-bit order.')
    add('hex','0x'+payload[0].hex().upper(),'/decoded','One raw byte, not an ASCII or UTF-8 message.')
    add('cost',f"{data['model_calls']:,} model calls in {data['elapsed_seconds']:,.2f} seconds",'','Whole execution, including loading, encoding and artifacts.')
    add('minutes',f"{data['elapsed_seconds']/60:.2f} minutes",'/elapsed_seconds','Seconds divided by sixty.')
    add('rss',f"{data['peak_rss_bytes']/1024**3:.2f} GiB",'/peak_rss_bytes','Recorded peak RSS, not a memory lower bound.')
    add('carry','all 15 stage boundaries matched byte for byte','/stage_carries','Sixteen stages after a sole initialization.')
    for i,text in enumerate(anchors):
        add('anchor-'+str(i),text,'/anchor_texts/'+str(i),'Exact finite calibration carrier.')
        matches=[r for r in rows if r['text']==text]
        require(all(r['controls_hex']==matches[0]['controls_hex'] for r in matches),'Repeated carrier computed controls differ')
        control=matches[0]['controls']
        add('control-'+str(i),f"({control['alpha']}, {control['log10_temperature']})",'/rows/'+str(matches[0]['index']-1)+'/controls','Actual computed binary64 controls, not requested endpoint substitution.')
    require('finite public two-sentence codebook' in article and 'A substantive ordinary-prose\nrequest and reply remain pending' in article,'Finite byte/reply scope missing')
    require('nearest-point diagnostic is not the final cycle classifier' in article,'Diagnostic/classifier distinction missing')
    correction_path='receipts/pythia-review-corrections-v1.json';correction=json.loads((root/correction_path).read_text())
    require(correction['C1']['distinct_label_controls']==['erasure','bit_flip'] and correction['C1']['messages']==4 and correction['C1']['all_rejected'],'Corrected label control census')
    require(len(correction['C2']['modules'])==8 and all(r['byte_identical'] for r in correction['C2']['modules']),'All eight shared source modules')
    add('two-label-controls','Two distinct offline controls','/C1','Erasure and bit flip; historical opposite-trace check was the same label vector.',correction_path)
    add('same-host','two one-thread processes on the same host','/C3','Same-host repeatability; no cross-host qualification.',correction_path)
    add('zero-outcomes','24 of 32 proposed zeros were correct, five erased, and three resolved incorrectly','/C4/confusion_per_receiver/0','All proposed-zero outcomes per receiver.',correction_path)
    add('one-outcomes','all 32 proposed ones were correct','/C4/confusion_per_receiver/1','All proposed-one outcomes per receiver.',correction_path)
    require('reply template and Python two-decimal formatting were shared' in flat,'Independent SQL scope overstated')
    observer_path='receipts/pythia-observer-report-v1.json'
    observer=json.loads((root/observer_path).read_text())
    observer_receipt=json.loads((root/'receipts/pythia-observer-v1.json').read_text())
    binding=observer_receipt['bindings']['report.json']
    require(sha(root/observer_path)==binding['sha256'] and (root/observer_path).stat().st_size==binding['bytes'],'Observer report identity')
    require(observer['status']=='passed' and observer['model_calls']==0 and observer['models_loaded'] is False and observer['producer_imports']==[],'Model-free observer gate')
    messages=observer['messages']
    prior=json.loads((root/'receipts/evidence-summary.json').read_text())['transfer']['messages']
    require(len(messages)==len(prior)==observer_receipt['messages']==4,'Complete observer message census')
    for row,old in zip(messages,prior,strict=True):
        require(all(row[k]==old[k] for k in ('variant','direction','text','payload_bytes')),'Observer payload differs from numerical receipt')
        payload=row['text'].encode('utf-8')
        require(len(payload)==row['payload_bytes'] and hashlib.sha256(payload).hexdigest()==row['payload_sha256']==row['numerical_reconstruction']['sha256'],'Observer recovered bytes/hash')
        require(row['unique_coordinates']==8 and row['all_assignment_count']==2**row['unique_coordinates']==256 and row['matching_assignment_count']==1,'Exhaustive unique-match gate')
        require(row['actual_map_labels']==row['actual_no_map_labels'] and row['all_grid_coordinate_gaps_zero'] and row['both_methods_equal_numerical_output'],'Observer label/grid agreement')
    require(sum(r['payload_bytes'] for r in messages)==observer['total_application_bytes']==observer_receipt['application_bytes']==470,'Observer complete byte count')
    require(observer_receipt['map_seconds']==[r['map_ns']/1e9 for r in messages] and observer_receipt['no_map_seconds']==[r['no_map_ns']/1e9 for r in messages],'Observer timing receipt')
    add('observer-bytes',f"all {observer['total_application_bytes']} bytes without numerical inference",'/total_application_bytes','Post-run public observer baseline; both methods recover recorded payloads.',observer_path)
    add('observer-coordinates','eight unique coordinates','/messages','Each of four messages has eight distinct exact-grid addresses.',observer_path)
    add('observer-assignments',f"all {messages[0]['all_assignment_count']} binary label assignments",'/messages','Exhaustive no-map enumeration; exactly one checksum match per message.',observer_path)
    for key,title in [('map_ns','observer-map-time'),('no_map_ns','observer-no-map-time')]:
        values=[r[key] for r in messages]
        add(title,f"{min(values)/1e6:.6f}–{max(values)/1e6:.6f} ms per message",'/messages','Descriptive single local warm-filesystem timing; excludes Python startup and evidence serialization.',observer_path)
    add('observer-load',f"{observer['map_loading_and_hash_ns']/1e6:.6f} ms",'/map_loading_and_hash_ns','Separate map load and hash time.',observer_path)
    require('model inference was unnecessary for these recorded Pythia wires' in flat,'Observer necessity boundary missing')
    return entries,dict(status='passed',model_calls=0,groups=8,measured_bits=bits,decoded=decoded,literal_carry_boundaries=15,report_assets=len(receipt['outputs']),observer_messages=len(messages),observer_bytes=observer['total_application_bytes'],scope=__doc__)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--emit',type=Path)
    a=p.parse_args();entries,report=verify(a.root)
    if a.emit:a.emit.write_text(json.dumps(entries,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2))
