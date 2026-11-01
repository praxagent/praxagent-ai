#!/usr/bin/env python3
"""Check published technical values against page-local receipts; no model access.

Normal mode is read-only. --emit [PATH] writes numerical-provenance additions
(default /tmp/technical-number-additions.json). --root selects a page bundle;
otherwise use this script's directory when it contains index.md, or cwd.
This verifies compact evidence and publication arithmetic, not archived arrays.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

Q='receipts/qwen-technical-evidence.json'
L='receipts/hysteresis-loops.json'
K='qwen-kite-source.json'
T='receipts/language-interface.json'
M='receipts/shrimp-blog-methods.json'

def require(test, message):
    if not test:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def words(n):
    names=['zero','one','two','three','four','five','six','seven','eight','nine',
           'ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen']
    return names[n] if 0 <= n < len(names) else str(n)

def sci(x):
    mantissa, exponent=f'{x:.2e}'.split('e')
    return mantissa+r'\times10^{'+str(int(exponent))+'}'

def verify(root):
    documents={name:json.loads((root/name).read_text()) for name in (Q,L,K,T,M)}
    hashes={name:sha(root/name) for name in documents}
    q,l,k,t=(documents[name] for name in (Q,L,K,T))
    article=(root/'index.md').read_text()
    flat=' '.join(article.split())
    entries=[]
    def add(id,text,file,pointer,claim):
        require(' '.join(text.split()) in flat, f'Article lacks derived {id}: {text!r}')
        entries.append({'id':'technical-'+id,'appears_as':text,
                        'source':file+'#'+pointer,'source_sha256':hashes[file],
                        'claim':claim})

    # Every compact loop row is independently cross-checked against the larger
    # saved-evidence extraction. No unresolved value is coerced to zero.
    require(len(l['groups'])==len(q['loops'])==4,'Loop configuration census changed')
    rows=[]
    for gi,(group,full) in enumerate(zip(l['groups'],q['loops'])):
        require((group['fraction'],group['dwell'])==(full['fraction'],full['dwell']),f'Loop group {gi}')
        require(group['audit']==full['audit'],f'Loop audit binding {gi}')
        require(len(group['rows'])==4,f'Loop group {gi} must contain four lanes')
        for ri,row in enumerate(group['rows']):
            b,h=divmod(ri,2)
            require((row['direction'],row['initial_history'])==(b,h),f'Loop ordering {gi}/{ri}')
            original=full['comparisons'][b]['middle_control_history_comparisons'][h]
            for key,other in [('outward_period','outward_period'),('return_period','return_period'),('gap','aligned_gap')]:
                require(row[key]==original[other],f'Loop {gi}/{ri}/{key}')
            stage=full['stages'][4*b+1]
            finish=full['stages'][4*b+3]
            require(row['endpoint_period']==stage['qualified_periods'][h],f'Endpoint {gi}/{ri}')
            labels=finish['labels'] if 'labels' in finish else finish['readouts']
            require(row['final_hold_bit']==labels[h]==b,f'Final readout {gi}/{ri}')
            target=l['controls']['E'+str(b)]
            require(row['endpoint_controls']==target and row['final_controls']==l['controls']['H'],f'Controls {gi}/{ri}')
            for key in target:
                expected=l['controls']['H'][key]+group['fraction']*(target[key]-l['controls']['H'][key])
                require(row['interior_controls'][key]==expected,f'Interior control {gi}/{ri}/{key}')
            qualified=row['outward_period']==row['return_period']==6
            require((row['gap'] is not None)==qualified,f'Qualification {gi}/{ri}')
            require(row['gap'] is None or math.isfinite(row['gap']) and row['gap']>=0,f'Invalid gap {gi}/{ri}')
            rows.append(row)
    require(len(rows)==16,'Expected all sixteen saved comparisons')
    counts={'comparison_rows':len(rows),'qualified_comparisons':sum(r['gap'] is not None for r in rows),
            'unresolved_comparisons':sum(r['gap'] is None for r in rows),'endpoint_rows':len(rows),
            'endpoint_qualified_period6':sum(r['endpoint_period']==6 for r in rows),
            'endpoint_unresolved':sum(r['endpoint_period'] is None for r in rows),
            'final_hold_correct':sum(r['final_hold_bit']==r['direction'] for r in rows)}
    require(counts==l['outcomes'],'Loop outcomes disagree with individual rows')
    require(all(r['endpoint_period'] is None for r in rows if r['direction']==1),'E1 endpoint claim changed')
    add('loop-count',f"All {words(len(rows))} rows are shown.",L,'/groups','Count all four original lanes in each saved configuration.')
    add('loop-qualified',f"{counts['qualified_comparisons']} qualified comparisons",L,'/groups','Recomputed count of two qualified period-six tails.')
    add('loop-unresolved',f"{counts['unresolved_comparisons']} unresolved comparisons",L,'/groups','Recomputed count with at least one unresolved tail.')
    for gi,ri,label in [(0,1,'zero240'),(1,1,'zero480'),(1,2,'one480')]:
        value=l['groups'][gi]['rows'][ri]['gap']
        add('loop-gap-'+label,f'{value:.5f}',L,f'/groups/{gi}/rows/{ri}/gap','Round saved same-lane outward/return phase-aligned L2 gap to five decimals.')
    for dwell in sorted({g['dwell'] for g in l['groups']}):
        add('loop-dwell-'+str(dwell),str(dwell),L,'/groups','Recorded updates per visited stage; not a convergence bound.')
    add('loop-period-search','period '+str(l['recurrence']['period_scan_maximum']),L,'/recurrence/period_scan_maximum','Upper lag searched; unresolved does not mean chaos.')
    for control,values in l['controls'].items():
        require(values==q['controls'][control],f'Control receipts disagree: {control}')
        for name,value in values.items():
            add(control+'-'+name,repr(value),Q,'/controls/'+control+'/'+name,'Exact saved control, checked against independent loop receipt.')

    # Prior continuation display: coexistence overlap, not full fold span.
    summary=k['original_figure_summary']
    add('kite-temperature',f"{summary['branch_log10_temperature']:.7f}".replace('-','−'),K,'/original_figure_summary/branch_log10_temperature','Seven decimal display of prior fixed branch cut.')
    overlap='–'.join(f'{v:.6f}' for v in summary['coexistence_alpha'])
    add('kite-overlap',overlap,K,'/original_figure_summary/coexistence_alpha','Sampled attracting overlap rounded to six decimals; not interval proof.')
    add('qwen-dimension',str(k['transformations']['state_array_shape'][1]),K,'/transformations/state_array_shape/1','Dimension of saved orbit representatives.')
    instruments=documents[M]['instruments']
    require(instruments['qwen']['dimensions']==k['transformations']['state_array_shape'][1],'Qwen dimension receipts disagree')
    for model in ('qwen','pythia'):
        display='Qwen' if model=='qwen' else 'Pythia'
        add(model+'-instrument-dimension',f"{instruments[model]['dimensions']} for {display}",M,'/instruments/'+model+'/dimensions','Full carried embedding dimension from pinned instrument configuration.')

    # Exact-line critical candidate and explicit failed continuation.
    fold=q['fold']; refined=fold['fold']['refined']
    require(fold['outcome']=='p1_branch_link_unresolved' and not fold['pulse_eligible'],'Fold scope changed')
    require(refined['passed'] and refined['cycle']['passed'],'Fold residual/cycle gates failed')
    add('fold-location',f"{refined['log10_temperature']:.15f}",Q,'/fold/fold/refined/log10_temperature','Fifteen decimal candidate coordinate; no interval error bound.')
    for key in ('orbit_l2','null_l2'):
        add('fold-'+key,sci(refined[key]),Q,'/fold/fold/refined/'+key,'Residual rounded to three significant digits, not a state error bound.')
    spectrum=fold['fold']['spectrum']; pair=spectrum['pairs'][spectrum['selected_index']]
    require(pair['qualified'] and pair['value'][1]==0,'Selected fold mode unqualified or nonreal')
    add('fold-mode-distance',sci(abs(pair['value'][0]-1)),Q,'/fold/fold/spectrum/pairs/'+str(spectrum['selected_index'])+'/value/0','Absolute selected approximate eigenvalue minus one, rounded to three significant digits; incomplete spectrum.')
    require(fold['branch']['status']=='unresolved','Branch continuation status changed')
    add('fold-steps',words(fold['branch']['accepted_steps'])+' steps',Q,'/fold/branch/accepted_steps','Accepted continuation step count; initial tangent failed before an attempted step.')

    state=q['fixed_H_state_intervention']; sr=state['rows']
    require([r['decoded_bit'] for r in sr]==[1,0,0,0,0,0],'Fixed-H intervention outcomes changed')
    add('state-radius',f"{state['radius']:.6f}",Q,'/fixed_H_state_intervention/radius','Known-success displacement L2 radius, rounded to six decimals.')
    state_labels=[r'Original \(h_1\)',r'Saved successful pulse state \(x_{60}=h_1+d\)',r'Opposite displacement \(h_1-d\)',r'Orthogonal displacements \(h_1+r u\) and \(h_1-r u\)',r'Original \(h_0\)']
    for i,(label,idx) in enumerate(zip(state_labels,[0,1,2,3,5])):
        value=str(sr[idx]['decoded_bit'])+(' for each' if idx==3 else '')
        add('state-row-'+str(i),'| '+label+' | '+value+' |',Q,'/fixed_H_state_intervention/rows/'+str(idx),'Direct saved readout; orthogonal entry checks both signs.')

    # All prose rows: bytes/hashes, Euclidean nearest-corner decision, statuses,
    # and the exact displayed sixteen-row table are checked from receipt values.
    pilot=t['generated_prose_pilot']; pr=pilot['rows']; pc=pilot['counts']
    require(len(pr)==16 and [r['record'] for r in pr]==list(range(16)),'Missing/reordered language records')
    corners=pilot['symbol_corners']
    require(corners==[[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]],'Corner labels changed')
    for i,row in enumerate(pr):
        raw=row['text'].encode('utf-8')
        require(len(raw)==row['text_utf8_bytes'] and hashlib.sha256(raw).hexdigest()==row['text_utf8_sha256'],f'Language text binding {i}')
        z=row['standardized_coordinates']
        nearest=min(range(len(corners)),key=lambda j:sum((a-b)**2 for a,b in zip(z,corners[j])))
        require(nearest==row['recovered_symbol'],f'Nearest-corner classification {i}')
        require(row['correct']==(row['requested_symbol']==nearest) and not row['erasure'],f'Language outcome {i}')
        require(all(row[x] for x in ['exact_sender_receiver_token_ids','exact_sender_receiver_saved_mean_bytes','exact_sender_receiver_coordinates']),f'Sender/receiver agreement {i}')
        line=f"| {i} | {row['requested_symbol']} → {row['recovered_symbol']} | {'yes' if row['truncated'] else 'no'} |"
        add('language-row-'+str(i),line,T,f'/generated_prose_pilot/rows/{i}','All displayed symbol/truncation cells generated from this saved row; text byte binding and nearest-corner decision checked.')
    recomputed={'attempts':len(pr),'correct_symbols':sum(r['correct'] for r in pr),
                'wrong_symbols':sum(not r['correct'] for r in pr),'erasures':sum(r['erasure'] for r in pr),
                'sender_receiver_exact_pairs':sum(r['exact_sender_receiver_coordinates'] for r in pr),
                'distinct_selected_texts':len({r['text'] for r in pr}),
                'selected_truncated_attempts':sum(r['truncated'] for r in pr),
                'attempts_without_correct_candidate':sum(not r['correct_symbol_available_in_topic_pool'] for r in pr),
                'selector_misses_despite_correct_candidate':sum(not r['correct'] and r['correct_symbol_available_in_topic_pool'] for r in pr)}
    for key,value in recomputed.items(): require(pc[key]==value,'Language aggregate '+key)
    require(pilot['numerical_dynamics_calls']==pilot['numerical_memory_writes']==pilot['numerical_memory_readouts']==0,'Pilot reached dynamics unexpectedly')
    add('language-success',f"{recomputed['correct_symbols']} of {len(pr)} requested symbols",T,'/generated_prose_pilot/rows','Count individual correct requested/recovered label pairs.')
    add('language-success-fraction',f"{recomputed['correct_symbols']}/{len(pr)} corner symbols",T,'/generated_prose_pilot/rows','Same census as prose count, explicitly geometric corner symbols.')
    add('language-coordinate-agreement',f"all {recomputed['sender_receiver_exact_pairs']} cases",T,'/generated_prose_pilot/rows','All saved sender/receiver equality flags checked, not recomputed Transformer embeddings.')
    add('language-wrong',words(recomputed['wrong_symbols'])+' coordinates landed',T,'/generated_prose_pilot/rows','Count wrong resolved symbols, with no erasures.')
    add('language-no-candidate',words(recomputed['attempts_without_correct_candidate']).capitalize()+' misses',T,'/generated_prose_pilot/rows','Saved candidate-availability flags for failed requests; full candidate-pool selection audited upstream.')
    add('language-selector-miss',words(recomputed['selector_misses_despite_correct_candidate']).capitalize()+' further miss',T,'/generated_prose_pilot/rows','Failed requests despite a correctly classified candidate existing in their topic pool.')
    topics=len({r['topic_id'] for r in pr}); require(pc['candidate_slots']%topics==0,'Unequal candidate allocation')
    add('language-candidate-design',f"{words(pc['candidate_slots']//topics)} candidates for each of {words(topics)}",T,'/generated_prose_pilot/counts/candidate_slots','Recorded candidate total divided by four distinct topics; balanced design is an upstream recorded fact.')
    add('language-first-baseline',f"{pc['first_candidate_correct_symbols']} of {len(pr)}",T,'/generated_prose_pilot/counts/first_candidate_correct_symbols','Recorded first-candidate baseline; full candidate pool is not bundled.')
    for success,label in [(True,'success'),(False,'failure')]:
        row=next(r for r in pr if r['correct']==success); idx=row['record']
        require(idx==pilot['example_selection']['first_'+('success' if success else 'failure')+'_record'],'Example selection rule failed')
        add('language-'+label+'-text',row['text'],T,f'/generated_prose_pilot/rows/{idx}/text','Exact unedited first '+label+' in packet order; UTF-8 hash checked.')
        coords='('+', '.join(f'{v:.6f}'.replace('-','−') for v in row['standardized_coordinates'])+')'
        add('language-'+label+'-coordinates',coords,T,f'/generated_prose_pilot/rows/{idx}/standardized_coordinates','Saved standardized text coordinates rounded to six decimals.')
        if success:
            coords='('+', '.join(repr(v).replace('-','−') for v in row['parameter_coordinates'])+')'
            add('language-success-address',coords,T,f'/generated_prose_pilot/rows/{idx}/parameter_coordinates','Computed affine address, never executed as a numerical write.')

    integrated=t['separate_integrated_controls']; commands=integrated['explicit_public_command_memory']
    accepted=[r for r in commands['rows'] if r['parser_outcome']=='accepted']
    rejected=[r for r in commands['rows'] if r['parser_outcome']=='rejected']
    require(all(r['decoded_bit']==r['requested_bit'] for r in accepted),'Accepted command failure')
    require(all(r['model_calls']==0 and r['decoded_bit'] is None for r in rejected),'Rejected command executed')
    recovered=''.join(str(r['decoded_bit']) for r in accepted)
    require(recovered==commands['recovered_writes'],'Command string mismatch')
    add('commands-accepted',words(len(accepted)).capitalize()+' accepted commands',T,'/separate_integrated_controls/explicit_public_command_memory/rows','Count accepted command records.')
    add('commands-rejected',words(len(rejected))+' invalid commands',T,'/separate_integrated_controls/explicit_public_command_memory/rows','Count rejected command records with zero model calls.')
    add('commands-recovered',recovered,T,'/separate_integrated_controls/explicit_public_command_memory/rows','Concatenate four accepted decoded bits in saved order.')
    add('command-example',accepted[0]['message'],T,'/separate_integrated_controls/explicit_public_command_memory/rows/0/message','Exact first accepted command, not generated natural prose.')
    poems=integrated['generated_poems']
    expected=sum(len(x.encode('utf-8')) for x in poems['generated_texts'].values())
    received=sum(r['verified_bytes'] for r in poems['received_prefixes'].values())
    require(expected==poems['expected_bytes'] and received==poems['verified_bytes'] and not poems['complete_poems'],'Poem byte accounting')
    add('poems-partial',f'{received} of {expected} intended bytes',T,'/separate_integrated_controls/generated_poems','Derive intended UTF-8 total from both poem texts and delivered total from saved prefixes.')
    return entries, {'scope':'Compact receipt arithmetic and article checks only; no raw-array, tokenizer or model replay.',
                     'article_sha256':sha(root/'index.md'),'source_sha256':hashes,
                     'loop_rows_checked':len(rows),'language_rows_checked':len(pr),'provenance_entries':len(entries)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    default=Path(__file__).resolve().parent
    parser.add_argument('--root',type=Path,default=default if (default/'index.md').exists() else Path.cwd())
    parser.add_argument('--emit',nargs='?',const='/tmp/technical-number-additions.json',type=Path)
    args=parser.parse_args()
    entries,summary=verify(args.root)
    if args.emit:
        args.emit.write_text(json.dumps(entries,indent=2,ensure_ascii=False)+'\n')
        summary['emitted']=str(args.emit)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
