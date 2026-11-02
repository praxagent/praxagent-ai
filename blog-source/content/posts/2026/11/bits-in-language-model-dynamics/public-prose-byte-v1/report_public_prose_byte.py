#!/usr/bin/env python3
"""Render audited public calibrated prose-byte evidence, without model imports.

All outputs remain outside the immutable case. --verify recomputes data and
regenerates the three rendered formats to compare their exact bytes.
"""
import argparse
import hashlib
import html
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

NAMES=('report.html','report.pdf','trajectory.pdf','trajectory.png','plot-data.npz','data.json')
COLORS=('#315d9c','#bd622d')
DIAGNOSTIC=('Nearest-point diagnostic: minimum Euclidean distance in all 896 coordinates '
            'to each reference cycle\'s six saved points. This is not the final cycle classifier. '
            'Only the plotted log-axis values are floored at 1e-15; saved distances are unchanged.')
SCOPE=('Public finite two-sentence calibration, one raw byte. This is a prerequisite, not a '
       'substantive message or reply, uncalibrated arbitrary-prose reliability, concealment, '
       'or a causal demonstration of hysteresis.')


def check(ok,message):
    if not ok:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def binding(path):
    path=Path(path);raw=path.read_bytes()
    return dict(path=str(path.resolve()),bytes=len(raw),sha256=sha(raw))


def serialized(value):return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()


class Inputs:
    def __init__(self,case):self.case=Path(case).resolve();self.records={};self.inventory=None
    def raw(self,path):
        path=Path(path).resolve();raw=path.read_bytes();record=dict(path=str(path),bytes=len(raw),sha256=sha(raw))
        if str(path) in self.records:check(record==self.records[str(path)],'input changed during report')
        self.records[str(path)]=record
        returned=self.case/'returned'
        if self.inventory is not None and path.is_relative_to(returned):
            rel=str(path.relative_to(returned));check(rel in self.inventory,'unsigned return input: '+rel)
            check(all(record[k]==self.inventory[rel][k] for k in ('bytes','sha256')),'return input hash: '+rel)
        return raw
    def json(self,path):return json.loads(self.raw(path))
    def array(self,path):return np.load(io.BytesIO(self.raw(path)),allow_pickle=False)
    def bound(self,base,record):
        rel=Path(record['path']);check(not rel.is_absolute() and '..' not in rel.parts,'unsafe raw path')
        path=base/rel;raw=self.raw(path)
        check(len(raw)==record['bytes'] and sha(raw)==record['sha256'],'raw artifact hash: '+str(rel))
        return raw


def gate(inputs,independent):
    case=inputs.case
    done=inputs.json(case/'completion.json')
    check(done['status']=='verified_return' and done['remaining_owned_processes']==[],'return not closed')
    raw=inputs.raw(case/'return-manifest.json');manifest=json.loads(raw)
    check(sha(raw)==done['manifest_sha256'],'return manifest differs')
    check(manifest['finished'] is True and manifest['remaining_owned_processes']==[],'owned execution not terminal')
    names=[r['path']for r in manifest['files']]
    check(len(names)==len(set(names)),'duplicate returned path')
    inputs.inventory={r['path']:r for r in manifest['files']}
    completed=inputs.json(case/'completion-audit/completion.json')
    check(completed['status']=='audit_passed' and completed['execution_status']=='completed'
          and completed['return_verified'] is True and completed['source_identity_verified'] is True,
          'completed source and saved-array audit required')
    check(completed['return_manifest_sha256']==done['manifest_sha256'],'audit return identity')
    audit_raw=inputs.raw(case/'completion-audit/scientific-audit.json');audit=json.loads(audit_raw)
    check(sha(audit_raw)==completed['scientific_audit_sha256'],'scientific audit hash')
    check(audit['status']=='audit_passed' and audit['execution_status']=='completed'
          and audit['completed_groups']==8 and audit['model_replayed'] is False,'scientific audit incomplete')
    review=inputs.json(independent)
    check(review['status']=='passed','independent raw review has not passed')
    check(review['case_id']==case.name and Path(review['case_directory']).resolve()==case
          and review['return_manifest_sha256']==done['manifest_sha256']
          and review['scientific_audit_sha256']==completed['scientific_audit_sha256']
          and review['execution_commit']==completed['source_commit'],'independent review belongs to another source or case')
    return completed,audit,review


def stage_states(inputs,base,stage,previous):
    check(stage['status']=='completed' and stage['steps']==stage['completed_steps'],'incomplete stage')
    initial=np.load(io.BytesIO(inputs.bound(base,stage['initial'])),allow_pickle=False)
    check(initial.dtype==np.float64 and initial.shape==(2,896) and np.isfinite(initial).all(),'initial state contract')
    check(initial.tobytes()==previous.tobytes(),'literal state carry mismatch')
    arrays=[initial[None]];cursor=0
    for record in stage['blocks']:
        with np.load(io.BytesIO(inputs.bound(base,record)),allow_pickle=False) as z:
            y=z['states'];n=len(y)
            check(y.dtype==np.float64 and y.shape==(n,2,896) and np.isfinite(y).all(),'block state contract')
            check(1<=n<=12 and int(z['first_state_index'])==cursor+1 and int(z['last_state_index'])==cursor+n,'block gap/order')
            check(all(float(z[k])==v for k,v in stage['controls'].items()),'block controls differ')
            arrays.append(y.copy());cursor+=n
    check(cursor==stage['steps'],'stage update census')
    return np.concatenate(arrays)


def distances(states,refs):
    check(states.shape[1:]==(2,896) and refs.shape==(2,6,896),'distance dimensions')
    result=np.empty((len(states),2,2),dtype=np.float64)
    for begin in range(0,len(states),128):
        end=min(begin+128,len(states))
        for lane in (0,1):
            for reference in (0,1):
                delta=states[begin:end,lane,None,:]-refs[None,reference,:,:]
                result[begin:end,lane,reference]=np.linalg.norm(delta,axis=2).min(axis=1)
    return result


def decoded(bits,checksum):
    check(len(bits)==8 and all(len(row)==2 and all(b in (0,1,None)for b in row)for row in bits),'bit census')
    if any(None in row for row in bits):return dict(status='unresolved',lane_bytes_hex=None,checksum_passed=None)
    payload=[bytes([sum(row[lane]<<(7-i) for i,row in enumerate(bits))])for lane in (0,1)]
    checks=[sha(b)==checksum for b in payload];agree=payload[0]==payload[1]
    return dict(status='delivered' if agree and all(checks) else 'checksum_or_lane_disagreement',
        lane_bytes_hex=[b.hex() for b in payload],lanes_agree=agree,checksum_passed=all(checks),lane_checksum_passed=checks)


def extract(case,independent):
    inputs=Inputs(case);completed,audit,review=gate(inputs,independent);base=inputs.case/'returned/run'
    result=inputs.json(base/'result.json');plan=inputs.json(base/'prepared-plan.json');source=inputs.json(base/'source.json')
    check(result['status']=='completed' and result['model_calls']==2928 and result['completed_batch_updates']==2880,'complete fixed schedule required')
    check(result['initializations']==1 and result['call_counts']==dict(prefix=16,comparison=32,update=2880),'execution counts')
    check(source['source_commit']==completed['source_commit'] and source['files']==plan['source_files'],'source identity')
    check(result['plan_sha256']==sha(inputs.raw(base/'prepared-plan.json'))==audit['plan_sha256'],'plan binding')
    check(audit['manifest_sha256']==sha(inputs.raw(base/'manifest.json')),'scientific manifest binding')
    check(review['raw_manifest_sha256']==audit['manifest_sha256'],'independent raw manifest binding')
    packet=inputs.json(base/'received-packet.json');compiled=inputs.json(base/'compiled-records.json');calibration=inputs.json(base/'calibration.json')
    check(len(calibration['texts'])==len(set(calibration['texts']))==2
          and all(row['text'] in calibration['texts'] for row in packet['records']),'finite two-sentence scope')
    check(len(packet['records'])==len(compiled)==len(result['groups'])==8 and len(result['stages'])==16,'eight record census')
    check(sha(inputs.raw(base/'received-packet.json'))==plan['packet_sha256'],'packet binding')
    refs=np.load(io.BytesIO(inputs.bound(base,result['references'])),allow_pickle=False)
    check(refs.dtype==np.float64 and refs.shape==(2,6,896) and np.isfinite(refs).all(),'reference contract')
    current=refs[:,0].copy();states=[current[None]];rows=[];carry=[]
    for i,(group,row,control) in enumerate(zip(result['groups'],packet['records'],compiled,strict=True)):
        marker=inputs.json(base/f'group-{i:02d}.json');check(marker==group,'group sentinel mismatch')
        check(group['group_id']==row['record']==i and group['stage_ids']==[2*i,2*i+1]
              and group['status']=='completed' and group['instrument_valid'] is True,'group contract')
        check(control['encoding']['text']==row['text'] and control['actual_endpoint_bytes_equal'] is True,'text/compiled control identity')
        theta=control['projection']['controls']
        for key,value in theta.items():check(value.hex()==control['projection']['controls_hex'][key],'actual coordinate bytes')
        start=sha(current.tobytes())
        for local,sid in enumerate(group['stage_ids']):
            stage=result['stages'][sid]
            check(stage==inputs.json(base/f'stage-{sid:02d}-completed.json'),'stage marker mismatch')
            check(stage['steps']==(120 if local==0 else 240),'pulse/hold duration')
            check(stage['controls']==(theta if local==0 else plan['map']['hold']),'actual pulse/hold controls')
            a=stage_states(inputs,base,stage,current);states.append(a[1:]);current=a[-1].copy()
            carry.append(dict(stage_id=sid,initial_raw_sha256=sha(a[0].tobytes()),terminal_raw_sha256=sha(a[-1].tobytes()),literal_previous_state_equal=True))
        audited=audit['groups'][i];ind=review['groups'][i]
        check(audited['group_id']==i and audited['bits']==group['raw_bits'],'saved-array audited bits differ')
        check(ind['text']==row['text'] and ind['actual_pair_bits']==group['raw_bits'],'independent readout differs')
        rows.append(dict(index=i+1,text=row['text'],text_sha256=sha(row['text'].encode()),
            controls=theta,controls_hex=control['projection']['controls_hex'],projection_scalar_hex=control['projection']['scalar_hex'],
            bits=group['raw_bits'],qualification=group['qualification'],update_range=[360*i+1,360*(i+1)],
            pulse_updates=120,hold_updates=240,initial_raw_sha256=start,terminal_raw_sha256=sha(current.tobytes())))
    states=np.concatenate(states);check(states.shape==(2881,2,896),'continuous trajectory census')
    bits=[r['bits']for r in rows];read=decoded(bits,packet['decoded_byte_sha256'])
    check(read==result['decoded']==audit['decoded'],'byte computed from measured bits differs')
    for k,v in read.items():check(review['decoded'].get(k)==v,'independent byte outcome differs')
    dist=distances(states,refs)
    for record in inputs.records.values():check(binding(record['path'])==record,'input changed during report')
    payload=dict(schema='public-prose-byte-report-v1',case=inputs.case.name,status='completed_saved_array_audit_passed',
        completed_utc=completed['finished_utc'],source_commit=source['source_commit'],source_files=source['files'],
        elapsed_seconds=result['elapsed_seconds'],cold_initialization_seconds=result['cold_initialization_seconds'],
        model_calls=result['model_calls'],call_counts=result['call_counts'],batch_updates=2880,row_updates=5760,threads=plan['limits']['threads'],
        peak_rss_bytes=result['peak_rss_bytes'],initializations=1,stage_carries=carry,hold=plan['map']['hold'],map=plan['map'],
        rows=rows,decoded=read,expected_checksum=packet['decoded_byte_sha256'],anchor_texts=calibration['texts'],
        packet_sha256=plan['packet_sha256'],calibration_semantic_sha256=calibration['semantic_sha256'],
        scientific_audit_sha256=completed['scientific_audit_sha256'],independent_review=binding(independent),
        return_manifest_sha256=completed['return_manifest_sha256'],diagnostic=DIAGNOSTIC,scope=SCOPE,
        distance_array_sha256=sha(dist.tobytes()),distance_shape=list(dist.shape),plot_floor=1e-15,
        plotted_values_below_floor=int(np.count_nonzero(dist<1e-15)),raw_inputs=list(inputs.records.values()))
    return payload,dist


def bit(value):return 'U' if value is None else str(value)


def headline(data):
    dec=data['decoded']
    if dec['status']=='delivered':return 'Both carried trajectories reconstructed 0x'+dec['lane_bytes_hex'][0].upper()
    if dec['status']=='unresolved':return 'Unresolved readout: no complete byte accepted'
    return 'Completed sequence: checksum or lane disagreement'


def render_html(data):
    e=html.escape
    rows=''.join('<tr><td>'+str(r['index'])+'</td><td class="sentence">'+e(r['text'])+'</td><td class="coordinate">'+repr(r['controls']['alpha'])+'<br>'+repr(r['controls']['log10_temperature'])+'</td><td class="bits">'+bit(r['bits'][0])+' / '+bit(r['bits'][1])+'</td></tr>' for r in data['rows'])
    hashes=''.join('<dt>'+e(k.replace('_',' '))+'</dt><dd>'+e(str(data[k]))+'</dd>'for k in ('source_commit','packet_sha256','scientific_audit_sha256','return_manifest_sha256'))
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Public calibrated prose byte</title>'
      '<style>body{margin:0;background:#f6f7f9;color:#202b39;font:16px/1.55 system-ui,sans-serif}main{max-width:1100px;margin:35px auto;padding:32px;background:white;border:1px solid #dce2e9;border-radius:12px}h1{font-size:32px;line-height:1.18;margin:9px 0 18px}h2{font-size:20px;margin-top:30px}.kicker{letter-spacing:.1em;text-transform:uppercase;font-size:12px;color:#58697f}.metrics{display:flex;gap:24px;flex-wrap:wrap;background:#edf3f8;padding:15px 20px;border-radius:6px}.metrics strong{font-size:22px;display:block}.scope{color:#566372;font-size:14px}table{border-collapse:collapse;width:100%;font-size:14px}th{text-align:left;background:#eef2f6}td,th{padding:12px 10px;border-bottom:1px solid #dce2e9;vertical-align:top}.sentence{width:49%}.coordinate{font:12px/1.7 ui-monospace,monospace;white-space:nowrap}.bits{font-weight:700;white-space:nowrap}img{max-width:100%;height:auto}dl{display:grid;grid-template-columns:180px 1fr;gap:5px;font-size:12px}dd{margin:0;overflow-wrap:anywhere;font-family:ui-monospace,monospace}dt{color:#607083}a{color:#315d9c}code{overflow-wrap:anywhere}.note{border-left:3px solid #c58b56;padding-left:14px}footer{font-size:12px;color:#667688;margin-top:24px}@media(max-width:700px){main{margin:0;padding:18px}h1{font-size:26px}table{font-size:12px}.coordinate{font-size:10px}td,th{padding:8px 5px}dl{display:block}dd{margin-bottom:10px}}</style><main>'
      '<div class="kicker">Qwen2.5-0.5B-Instruct · public finite carrier · completed raw audit</div><h1>'+e(headline(data))+'</h1>'
      '<div class="metrics"><div><strong>'+f'{data["elapsed_seconds"]:.1f}'+'</strong>seconds, including initialization</div><div><strong>2,928</strong>model forwards</div><div><strong>2,880</strong>updates in each carried lane</div><div><strong>'+('Passed' if data['decoded']['checksum_passed'] else 'Not accepted')+'</strong>byte checksum</div></div>'
      '<p class="scope">'+e(SCOPE)+'</p><h2>Exact text → computed endpoint → measured bits</h2>'
      '<p>Each received sentence was freshly tokenized and averaged from pinned input-embedding rows in float64. The public calibration is <code>d = μ₁ − μ₀; s = ((x − μ₀) · d)/(d · d); θ = (1 − s)E₀ + sE₁</code>, fitted to the two exact sentences only. Actual computed endpoint-byte checks passed. Each pulse uses 120 updates, followed by 240 at H = (<code>'+repr(data['hold']['alpha'])+', '+repr(data['hold']['log10_temperature'])+'</code>) in (α, log₁₀ T). No expected label supplies the measured bits.</p>'
      '<table><thead><tr><th>Step</th><th>Exact received sentence</th><th>Computed α<br>Computed log₁₀ T</th><th>Measured bits<br>start 0 / start 1</th></tr></thead><tbody>'+rows+'</tbody></table>'
      '<p class="note">One initialization only: the two lanes begin at reference 0 and reference 1, phase zero. Every endpoint state enters its hold literally; every hold terminal state enters the next pulse literally. All 15 stage boundaries matched byte for byte. The lanes are parallel observations, not independent reliability trials.</p>'
      '<h2>All eight pulse-and-hold trajectories</h2><img src="trajectory.png" alt="Two panels, one per initial reference state, showing nearest-point distance to each reference cycle across all 2880 carried updates, with pulse and hold bands and measured endpoint bits."><p class="scope">'+e(DIAGNOSTIC)+'</p>'
      '<h2>Byte assembly and provenance</h2><p>Readout order is most-significant bit first. Measured lanes: <code>'+e(''.join(bit(r['bits'][0])for r in data['rows']))+'</code> and <code>'+e(''.join(bit(r['bits'][1])for r in data['rows']))+'</code>. Reconstructed hex bytes: <code>'+e(str(data['decoded']['lane_bytes_hex']))+'</code>. Status: <strong>'+e(data['decoded']['status'])+'</strong>. This is a raw byte, not a claimed ASCII or UTF-8 message.</p>'
      '<p>Expected checksum is used only after bit assembly: <code>'+e(data['expected_checksum'])+'</code>. The final labels come from audited recurrence and unique reference-orbit checks, not the plotted distance diagnostic.</p><dl>'+hashes+'</dl>'
      '<p><a href="report.pdf">Printable report</a> · <a href="trajectory.pdf">Vector figure</a> · <a href="receipt.json">Source and output receipt</a> · <a href="data.json">All report values and raw input bindings</a></p><footer>Saved-array audit only; no Transformer replay by this report. Successful-update probabilities are hash-bound in the experiment, while states and diagnostic arrays are retained. Completed '+e(data['completed_utc'])+'.</footer></main></html>')


def render_figure(data,dist,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(11.5,5.6),sharex=True,layout='constrained')
    x=np.arange(len(dist))
    for lane,ax in enumerate(axes):
        for i in range(8):
            ax.axvspan(i*360,i*360+120,color='#efd3ae',alpha=.55,lw=0)
            ax.axvspan(i*360+120,(i+1)*360,color='#dfeaf4',alpha=.40,lw=0)
            ax.axvline((i+1)*360,color='#748395',ls=':',lw=.6)
            ax.text((i+.5)*360,1.015,'#'+str(i+1)+' read '+bit(data['rows'][i]['bits'][lane]),
                    transform=ax.get_xaxis_transform(),ha='center',fontsize=9,color='#26374b')
        for ref in (0,1):ax.plot(x,np.maximum(dist[:,lane,ref],1e-15),color=COLORS[ref],lw=1.1,label='Distance to reference '+str(ref))
        ax.set_yscale('log');ax.set_xlim(0,2880);ax.set_ylabel('Start '+str(lane)+'\nEuclidean distance')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[-1].set_xticks(np.arange(0,2881,360));axes[-1].set_xlabel('Carried update index (0 = sole reference initialization; no reset between sentences)')
    handles,labels=axes[0].get_legend_handles_labels();handles += [Patch(facecolor='#efd3ae',alpha=.55),Patch(facecolor='#dfeaf4',alpha=.65)];labels += ['Computed-endpoint pulse: 120 updates','Common hold: 240 updates']
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,fontsize=9)
    fig.suptitle('Eight exact public carriers, two continuously carried trajectories\n'
                 'Nearest of six reference points: diagnostic only, not cycle readout (plot floor 1e-15)',fontsize=12)
    fig.savefig(out/'trajectory.png',dpi=180,metadata={'Software':'public-prose-byte-report'})
    fig.savefig(out/'trajectory.pdf',metadata={'Title':'Public calibrated prose byte: carried trajectories','CreationDate':None,'ModDate':None})
    plt.close(fig)


def render_pdf(data,out,pdf_dependencies=None):
    if pdf_dependencies:sys.path.append(str(pdf_dependencies))
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4,landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from matplotlib.font_manager import findfont
    pdfmetrics.registerFont(TTFont('ReportSans',findfont('DejaVu Sans')))
    class StableCanvas(Canvas):
        def __init__(self,*a,**kw):kw['invariant']=1;super().__init__(*a,**kw)
    page=landscape(A4);width=page[0]-72
    doc=SimpleDocTemplate(str(out/'report.pdf'),pagesize=page,leftMargin=36,rightMargin=36,topMargin=30,bottomMargin=30,
                          title='Public calibrated prose byte',author='Shrimp Memory evidence report')
    styles={name:ParagraphStyle(name,fontName='ReportSans',fontSize=size,leading=leading,textColor=colors.HexColor('#26374b'),spaceAfter=space) for name,size,leading,space in [('title',19,23,8),('body',9,12,7),('small',8,10,5),('cell',9,11,0)]}
    p=lambda text,style='body':Paragraph(text,styles[style])
    content=[p(html.escape(headline(data)),'title'),p(f'{data["elapsed_seconds"]:.1f} seconds | 2,928 forwards | 2,880 updates per lane | effective float64 | one CPU thread'),
             p(html.escape(SCOPE),'small')]
    cells=[[p('Step','cell'),p('Exact received sentence','cell'),p('Computed alpha<br/>Computed log10 T','cell'),p('Measured bits<br/>start 0 / start 1','cell')]]
    for r in data['rows']:
        cells.append([p(str(r['index']),'cell'),p(html.escape(r['text']),'cell'),p(repr(r['controls']['alpha'])+'<br/>'+repr(r['controls']['log10_temperature']),'cell'),p(bit(r['bits'][0])+' / '+bit(r['bits'][1]),'cell')])
    table=Table(cells,colWidths=[44,width-44-165-95,165,95],repeatRows=1)
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef5')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7fa')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#9bacbf'))]))
    content += [table,Spacer(1,9),p('Each exact sentence was freshly encoded as a float64 mean, then projected by the public two-anchor compiler. Every computed endpoint passed an exact-byte check. Each row applies its endpoint for 120 updates, then H = ('+repr(data['hold']['alpha'])+', '+repr(data['hold']['log10_temperature'])+') for 240 updates. All 15 subsequent stage boundaries preserve the literal previous terminal state.','small'),
        p('Measured bits (most-significant first): '+''.join(bit(r['bits'][0])for r in data['rows'])+' / '+''.join(bit(r['bits'][1])for r in data['rows'])+'. Raw byte hex: '+('unresolved' if data['decoded']['lane_bytes_hex'] is None else ' / '.join('0x'+x.upper()for x in data['decoded']['lane_bytes_hex']))+'. Checksum: '+str(data['decoded']['checksum_passed'])+'. This is not an ASCII or UTF-8 message.','small'),PageBreak(),
        p('Complete carried trajectories','title'),Image(str(out/'trajectory.png'),width=width,height=width*5.6/11.5),p(html.escape(DIAGNOSTIC),'small'),
        p('Final bits come from the independent saved-array recurrence and unique reference-orbit classifier. The plotted distance diagnostic does not supply them. Both starting lanes and every wrong or unresolved final readout are retained.','small'),
        p('Execution source: '+data['source_commit']+'<br/>Packet SHA-256: '+data['packet_sha256']+'<br/>Scientific-audit SHA-256: '+data['scientific_audit_sha256'],'small'),
        p('Full raw input bindings, independent-review hash, all executed source-file hashes and reproducible output hashes: data.json and receipt.json. Saved-array audit only; no model replay by this report.','small')]
    doc.build(content,canvasmaker=StableCanvas)


def outputs(data,dist,out,pdf_dependencies):
    (out/'data.json').write_bytes(serialized(data));(out/'report.html').write_text(render_html(data))
    # NumPy's ZIP timestamps are fixed; exact archive bytes reproduce.
    np.savez(out/'plot-data.npz',update_index=np.arange(len(dist),dtype=np.int64),distance=dist)
    render_figure(data,dist,out);render_pdf(data,out,pdf_dependencies)


def run(args):
    out=Path(args.output).resolve();case=Path(args.case).resolve()
    check(not out.is_relative_to(case),'report output must be outside immutable case')
    data,dist=extract(case,args.independent_review)
    if args.verify:
        receipt=json.loads((out/'receipt.json').read_text())
        check(binding(__file__)==receipt['generator'],'generator differs')
        check((out/'data.json').read_bytes()==serialized(data),'reconstructed report data differs')
        with tempfile.TemporaryDirectory(prefix='.verify-',dir=out.parent) as temp:
            fresh=Path(temp);outputs(data,dist,fresh,args.pdf_dependencies)
            for name in NAMES:
                check((fresh/name).read_bytes()==(out/name).read_bytes(),'regenerated output differs: '+name)
                expected=receipt['outputs'][name];actual=binding(out/name)
                check(all(expected[k]==actual[k]for k in ('bytes','sha256')),'output receipt differs: '+name)
        print('Verified: all gates, raw inputs, distances, measured bits, byte assembly and exact rendered output bytes.')
    else:
        out.mkdir(parents=True,exist_ok=False);outputs(data,dist,out,args.pdf_dependencies)
        import matplotlib,reportlab
        receipt=dict(schema='public-prose-byte-report-receipt-v1',generator=binding(__file__),case=str(case),
            independent_review=binding(args.independent_review),source_commit=data['source_commit'],
            scientific_audit_sha256=data['scientific_audit_sha256'],source_files=data['source_files'],
            runtime=dict(python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__,reportlab=reportlab.Version),
            outputs={name:binding(out/name)for name in NAMES},diagnostic=DIAGNOSTIC,scope=SCOPE)
        (out/'receipt.json').write_bytes(serialized(receipt));print(json.dumps({'output':str(out),'status':data['decoded']['status']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True)
    p.add_argument('--independent-review',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pdf-dependencies',type=Path);p.add_argument('--verify',action='store_true');run(p.parse_args())
