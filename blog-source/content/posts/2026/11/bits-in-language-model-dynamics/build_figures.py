"""Rebuild the blog's figures from released receipts; no model inference.

Run: python build_figures.py [--verify] [--index path/to/index.md]
Requires Matplotlib. PNG, SVG and PDF output is deterministic under the pinned
Matplotlib/font environment recorded in the figure receipts. Reused rasters are
checked byte-for-byte against their original source hashes, never traced.
Editorial fields and additional prose-number entries in provenance.json survive
regeneration. expected_numbers(evidence) exposes the generated measured claims.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent
SUMMARY_SHA='e48920cc2f042a16c67fadc56e52b85393ace0c5fe6aed5dae1b34f21c960848'
AUDIT_SHA='2cf38a7e6bcbefab8758c898f0ab9342a5a71a4568a9be6442f853de39aa258e'
BG='#F7F4F0'; INK='#292721'; MUTED='#615B52'; GRID='#DAD2C6'
BLUE='#355F7C'; TAN='#956541'; PALE='#ECE5D9'
sha=lambda b:hashlib.sha256(b).hexdigest()
jsonbytes=lambda x:(json.dumps(x,indent=2,ensure_ascii=False)+'\n').encode()
def norm(text):
 return re.sub(r'\s+',' ',text.replace('**','').replace('*','').translate(str.maketrans({'–':'-','—':'-','−':'-'}))).strip()

def expected_numbers(evidence):
 """Measured claims for the main article; caller may add paper-specific fields."""
 t=evidence['transfer'];m=t['metrics'];s=evidence['spatial_bank'];q=evidence['qwen_other']['generated_exchange']
 n=lambda ident,text,source,claim:{'id':ident,'appears_as':text,'source':source,'claim':claim}
 p='receipts/evidence-summary.json'
 return [
 n('application-total',f"{m['application_bytes']:,} bytes, or {m['payload_bits']:,} bits",p+'#/transfer/metrics','All four delivered payloads across two complete paired exchanges.'),
 n('forward-calls',f"{m['model_calls']:,} model forward calls",p+'#/transfer/metrics/model_calls','All four receiver processes, including readiness calls.'),
 n('family-seconds',f"{m['elapsed_seconds']:,.1f} seconds",p+'#/transfer/metrics/elapsed_seconds','Enclosing family execution, not sum of the bars.'),
 n('family-minutes',f"{m['elapsed_seconds']/60:.1f} minutes",p+'#/transfer/metrics/elapsed_seconds','Seconds divided by sixty.'),
 n('payload-throughput',f"{m['application_bytes_per_second']:.3f} delivered application bytes per second",p+'#/transfer/metrics/application_bytes_per_second','Application bytes divided by enclosing family execution seconds.'),
 n('known-addresses','eight known',p+'#/transfer/parameters/known_addresses','Eight previously tested addresses reused; not independent bit trials.'),
 n('batch-symbols','Sixteen separately evolved',p+'#/transfer/parameters/payload_symbols_per_batch','Sixteen separately maintained symbol states share each batch; no statistical independence claim.'),
 n('warmup',f"{t['parameters']['warmup_updates']} warm-up updates",p+'#/transfer/parameters/warmup_updates','Fixed per-symbol warm-up.'),
 n('observation',f"{t['parameters']['observation_updates']} updates",p+'#/transfer/parameters/observation_updates','Fixed observation endpoint.'),
 n('payload-counts','148-byte request and 88-byte reply, followed by a changed 148-byte request and 86-byte reply',p+'#/transfer/messages','All four message sizes, from recovered UTF-8 bytes.'),
 n('spatial-draw',f"{s['candidate_pairs']} eligible patches",p+'#/spatial_bank/candidate_pairs','One paired coordinate draw from each patch.'),
 n('spatial-candidates',f"all {s['candidate_coordinates']} coordinates",p+'#/spatial_bank/candidate_coordinates','Complete candidate denominator for each receiver.'),
 n('spatial-final-a','| A | 56 | 5 | 3 |',p+'#/spatial_bank/receivers/0/outcomes/2','Correct, erased and wrong final labels in receiver A.'),
 n('spatial-final-b','| B | 56 | 5 | 3 |',p+'#/spatial_bank/receivers/1/outcomes/2','Correct, erased and wrong final labels in receiver B.'),
 n('spatial-bank',f"{s['accepted_pairs']} pairs, or {s['qualified_coordinates']} addresses",p+'#/spatial_bank','Only complete pairs that passed in both receivers.'),
 n('english-wire-bytes',f"{m['wire_bytes']:,} English instruction bytes",p+'#/transfer/metrics/wire_bytes','Complete public English instructions across all four messages.'),
 n('english-wire-expansion',f"about {m['wire_bytes']/m['application_bytes']:.0f} times the application payload",p+'#/transfer/metrics','Wire bytes divided by application bytes; rounded ratio, not additional payload.'),
 n('qwen-application-bytes',f"{q['application_bytes']} application bytes",p+'#/qwen_other/generated_exchange/application_bytes','The recovered Bob and A messages total four UTF-8 bytes.'),
 n('qwen-model-calls',f"{q['model_calls']:,} model calls",p+'#/qwen_other/generated_exchange/model_calls','Whole Qwen exchange, including four authoring calls.'),
 n('qwen-execution-minutes',f"{q['elapsed_seconds']/60:.1f} minutes",p+'#/qwen_other/generated_exchange/elapsed_seconds','Whole Qwen exchange elapsed seconds divided by sixty.'),
 n('qwen-write-updates',f"{q['schedule']['write_updates']} write updates",'receipts/qwen-schedule.json#/write_updates','Every bit write used this endpoint duration.'),
 n('qwen-hold-updates',f"{q['schedule']['common_hold_updates']} common-hold updates",'receipts/qwen-schedule.json#/common_hold_updates','Every write was followed by this common-hold duration.'),
 n('qwen-reference-starts','two reference starts','receipts/qwen-schedule.json#/reference_starts','The same payload was tested from phase zero of both reference cycles; this does not double capacity.'),
 n('qwen-payload',f"all {q['payload_bits']} payload bits",p+'#/qwen_other/generated_exchange/payload_bits','Each reference-start lane recovers the same payload; lanes do not double it.'),
 n('and-panel','All eight input triples',p+'#/qwen_logic/0/truth_table','Complete finite AND panel, intermediate and final outputs.'),
 n('or-panel','all eight did so for OR',p+'#/qwen_logic/1/truth_table','Complete finite OR panel, intermediate and final outputs.')]

def check_evidence(d,full):
 assert d['transfer']['metrics']==full['substantive_pythia']['metrics']
 msgs=d['transfer']['messages'];assert [m['payload_bytes'] for m in msgs]==[148,88,148,86]
 assert sum(m['payload_bytes'] for m in msgs)==d['transfer']['metrics']['application_bytes']
 assert sum(m['model_calls'] for m in msgs)==d['transfer']['metrics']['model_calls']
 for m,f in zip(msgs,full['substantive_pythia']['messages'],strict=True):
  assert all(m[k]==f[k] for k in m)
  assert len(m['text'].encode())==m['payload_bytes']
 for g in d['qwen_logic']:
  assert len(g['truth_table'])==8
  for row in g['truth_table']:
   a,b,c=row['a'],row['b'],row['c'];first=a&b if g['operation']=='AND' else a|b;final=first&c if g['operation']=='AND' else first|c
   assert row['first_actual']==first and row['final_actual']==final
 assert d['spatial_bank']['qualified_coordinates']==48
 for r in d['spatial_bank']['receivers']:
  assert r['outcomes'][-1]=={'observation_updates':128,'correct':56,'erased':5,'wrong':3}
 assert d['transfer']['parameters']=={'known_addresses':8,'payload_symbols_per_batch':16,'warmup_updates':32,'observation_updates':128}

def check_prose(index,d,full):
 text=index.read_text();normalized=norm(text)
 for entry in expected_numbers(d):
  assert norm(entry['appears_as']) in normalized,'Missing measured prose: '+entry['id']
 examples=[]
 for variant,direction in [('base','alice-to-bob'),('base','bob-to-alice'),('counterfactual','bob-to-alice')]:
  row=next(m for m in d['transfer']['messages'] if m['variant']==variant and m['direction']==direction)
  assert row['text'].rstrip('\n') in text,'Actual message excerpt differs'
  examples.append({'variant':variant,'direction':direction,'text_sha256':sha(row['text'].encode()),'check':'Complete received message occurs verbatim, apart from final newline.'})
 wire=full['substantive_pythia']['messages'][0]['wire_excerpt'].splitlines()[1:3]
 for line in wire:assert line in text,'Public English wire excerpt differs'
 first=next(m for m in d['transfer']['messages'] if m['variant']=='base' and m['direction']=='alice-to-bob')['text'].encode()[0]
 assert format(first,'08b') in text and '`'+chr(first)+'`' in text,'Opening byte example differs'
 return {'examples':examples,'wire_excerpt_sha256':sha(('\n'.join(wire)+'\n').encode()),'opening_byte_bits':format(first,'08b')}

def setup():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'text.color':INK,'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':INK,'axes.edgecolor':GRID,'svg.fonttype':'none','svg.hashsalt':'shrimp-main-v1','axes.titleweight':'bold','savefig.facecolor':BG,'figure.facecolor':BG,'axes.facecolor':BG,'pdf.fonttype':42})

def formats(fig,stem,title,description):
 out={}
 for ext in ('png','svg','pdf'):
  b=io.BytesIO();meta={'Software':'shrimp-main build_figures.py'} if ext=='png' else {'Date':None,'Creator':'shrimp-main build_figures.py'} if ext=='svg' else {'CreationDate':None,'ModDate':None,'Creator':'shrimp-main build_figures.py'}
  fig.savefig(b,format=ext,dpi=300,metadata=meta,facecolor=BG)
  raw=b.getvalue()
  if ext=='svg':
   from xml.sax.saxutils import escape
   raw=raw.replace(b'<svg ',b'<svg role="img" aria-labelledby="fig-title fig-desc" ',1)
   pos=raw.index(b'>',raw.index(b'<svg '))+1
   raw=raw[:pos]+f'<title id="fig-title">{escape(title)}</title><desc id="fig-desc">{escape(description)}</desc>'.encode()+raw[pos:]
   raw=b'\n'.join(line.rstrip() for line in raw.splitlines())+b'\n'
   ET.fromstring(raw)
  out[stem+'.'+ext]=raw
 plt.close(fig);return out

def transfer(d):
 rows=d['transfer']['messages'];labels=['Base request','Base reply','Changed request','Changed reply'];colors=[BLUE,TAN,BLUE,TAN]
 fig,axs=plt.subplots(2,1,figsize=(6.8,8.2));fig.subplots_adjust(left=.34,right=.93,top=.83,bottom=.09,hspace=.63)
 fig.text(.055,.958,'Four messages, fully recovered',fontsize=22,weight='bold')
 fig.text(.055,.918,'Base and changed-input exchanges',fontsize=15,color=MUTED)
 for ax,values,title,ticks,limit,fmt in [(axs[0],[m['payload_bytes'] for m in rows],'Received application bytes',[0,75,150],175,'{:.0f}'),(axs[1],[m['seconds']/60 for m in rows],'Receiver execution (minutes)',[0,25,50],59,'{:.2f}')]:
  ax.barh(range(4),values,color=colors,height=.56,zorder=3)
  for y,value in enumerate(values):ax.text(value+limit*.025,y,fmt.format(value),va='center',ha='left',fontsize=17,weight='bold')
  ax.set(yticks=range(4),yticklabels=labels,xlim=(0,limit),xticks=ticks);ax.invert_yaxis();ax.grid(axis='x',color=GRID,lw=.8,zorder=0)
  ax.set_title(title,loc='left',fontsize=17,pad=16);ax.tick_params(axis='both',length=0,labelsize=16,pad=10)
  for spine in ax.spines.values():spine.set_visible(False)
 description='All four received messages: base request 148 bytes, base reply 88, changed request 148, changed reply 86. Separate bars show receiver execution minutes including loading, readiness and evidence recording. These are four runs, without uncertainty intervals; whole-family wall time is a separate metric.'
 return formats(fig,'transfer-results','Four complete Pythia message transfers',description),description

def logic(d):
 ordered=[sorted(g['truth_table'],key=lambda x:(x['a'],x['b'],x['c'])) for g in d['qwen_logic']]
 assert [g['operation'] for g in d['qwen_logic']]==['AND','OR']
 matrix=[[ordered[0][i]['first_actual'],ordered[0][i]['final_actual'],ordered[1][i]['first_actual'],ordered[1][i]['final_actual']] for i in range(8)]
 fig,ax=plt.subplots(figsize=(6.8,7.8));fig.subplots_adjust(left=.25,right=.95,top=.76,bottom=.10)
 fig.text(.055,.957,'Two gates, one carried state',fontsize=22,weight='bold');fig.text(.055,.914,'Measured outputs for every input triple',fontsize=15,color=MUTED)
 ax.imshow(matrix,cmap=ListedColormap([PALE,BLUE]),vmin=0,vmax=1,aspect='auto',interpolation='nearest')
 for y,row in enumerate(matrix):
  for x,v in enumerate(row):ax.text(x,y,str(v),ha='center',va='center',fontsize=23,weight='bold',color=INK if v==0 else '#FFFFFF')
 ax.set(xticks=range(4),xticklabels=['After B','After C','After B','After C'],yticks=range(8),yticklabels=[f"{r['a']}  {r['b']}  {r['c']}" for r in ordered[0]])
 ax.xaxis.tick_top();ax.tick_params(length=0,pad=10,labelsize=17)
 ax.text(-.11,1.018,'A B C',transform=ax.transAxes,ha='center',fontsize=17,weight='bold')
 ax.text(.25,1.125,'AND',transform=ax.transAxes,ha='center',fontsize=20,weight='bold');ax.text(.75,1.125,'OR',transform=ax.transAxes,ha='center',fontsize=20,weight='bold')
 for y in range(1,8):ax.axhline(y-.5,color=BG,lw=2)
 for x in range(1,4):ax.axvline(x-.5,color=BG,lw=5 if x==2 else 2)
 for spine in ax.spines.values():spine.set_visible(False)
 fig.text(.055,.037,'First output → carry the same state → second output',fontsize=13.5,color=MUTED)
 description='Eight rows contain all triples A, B, C from 000 to 111. Four columns show measured A AND B, then its carried AND C, and measured A OR B, then its carried OR C. All outputs match the Boolean targets at the tested phase-zero starts. Digits distinguish values without depending on color. This is a finite panel, not a general reliability estimate.'
 return formats(fig,'stateful-logic','Complete measured AND and OR chain panels',description),description

def routes():
 raw='''<svg xmlns="http://www.w3.org/2000/svg" width="800" height="1560" viewBox="0 0 400 780" role="img" aria-labelledby="title desc">
<title id="title">Two routes through model dynamics</title>
<desc id="desc">Conceptual diagram, not measured trajectories. Pythia begins anew for each address, observes the response and classifies a bit. Qwen carries its current state through a write excursion, returns to common hold controls and reads the value; the same state continues into the next operation. Both use public control instructions.</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#615B52"/></marker></defs>
<style>.t{font-family:Arial,Helvetica,sans-serif;fill:#292721}.h{font-size:23px;font-weight:700}.b{font-size:18px;font-weight:700}.s{font-size:17px}.m{font-size:15px;fill:#615B52}.a{fill:none;stroke:#615B52;stroke-width:2;marker-end:url(#arrow)}</style>
<rect width="400" height="780" fill="#F7F4F0"/>
<text class="t h" x="24" y="35">Two routes through</text><text class="t h" x="24" y="63">model dynamics</text>
<text class="t m" x="24" y="88">Conceptual diagram · public controls</text>
<rect x="18" y="110" width="364" height="270" rx="15" fill="#EDF1F4" stroke="#355F7C"/>
<text class="t b" x="34" y="141">Pythia: address a response</text>
<rect x="34" y="158" width="332" height="48" rx="9" fill="#FFFFFF"/><text class="t s" x="50" y="187">Reset at a chosen address</text>
<path class="a" d="M200 207V226"/>
<rect x="34" y="230" width="332" height="48" rx="9" fill="#FFFFFF"/><text class="t s" x="50" y="259">Run and observe the response</text>
<path class="a" d="M200 279V298"/>
<rect x="34" y="302" width="332" height="48" rx="9" fill="#FFFFFF"/><text class="t s" x="50" y="331">Classify a bit; assemble bytes</text>
<text class="t m" x="34" y="369">Each next address starts afresh.</text>
<rect x="18" y="405" width="364" height="340" rx="15" fill="#F1E9E0" stroke="#956541"/>
<text class="t b" x="34" y="436">Qwen: write a carried state</text>
<rect x="34" y="454" width="303" height="48" rx="9" fill="#FFFFFF"/><text class="t s" x="50" y="483">Keep the current model state</text>
<path class="a" d="M186 503V522"/>
<rect x="34" y="526" width="303" height="48" rx="9" fill="#FFFFFF"/><text class="t s" x="50" y="555">Apply a write excursion</text>
<path class="a" d="M186 575V594"/>
<rect x="34" y="598" width="303" height="66" rx="9" fill="#FFFFFF"/><text class="t s" x="50" y="624">Return to common hold</text><text class="t s" x="50" y="647">and read the resulting state</text>
<path class="a" d="M338 631H359V478H339"/>
<text class="t m" x="34" y="698">The same state continues</text><text class="t m" x="34" y="719">into the next operation.</text>
<text class="t m" x="24" y="768">Different primitives, different tests.</text>
</svg>'''.encode();ET.fromstring(raw);return {'two-routes.svg':raw}

def og():
 fig=plt.figure(figsize=(12,6.3),dpi=100);ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1200),ylim=(0,630));ax.axis('off')
 ax.text(62,568,'PRAXAGENT  /  RESEARCH NOTE',fontsize=16,weight='bold',color=MUTED)
 ax.text(62,464,'Bits in a language',fontsize=43,weight='bold');ax.text(62,386,'model’s dynamics',fontsize=43,weight='bold')
 ax.text(64,316,'Public transport. Writable state. Finite evidence.',fontsize=22,color=MUTED)
 for x,w,label,route,color in [(62,504,'PYTHIA','Reset → Classify',BLUE),(600,538,'QWEN','Carry → Hold → Read',TAN)]:
  ax.add_patch(FancyBboxPatch((x,110),w,136,boxstyle='round,pad=0,rounding_size=14',facecolor='#FCFAF7',edgecolor=color,lw=2))
  ax.text(x+26,208,label,fontsize=15,weight='bold',color=color);ax.text(x+26,153,route,fontsize=26,weight='bold')
 ax.text(64,49,'Conceptual illustration · no concealment claim',fontsize=14,color=MUTED)
 b=io.BytesIO();fig.savefig(b,format='png',dpi=100,metadata={'Software':'shrimp-main build_figures.py'},facecolor=BG);plt.close(fig)
 return {'og-card.png':b.getvalue()}

def figure_receipt(group,outputs,description,d,scope,fields=None,reproduction=None):
 r={'schema':'shrimp-main-figure-receipt-v1','figure_group':group,'description':description,'alt_text':description,'scope':scope,'source_evidence':{'path':'receipts/evidence-summary.json','sha256':SUMMARY_SHA},'data_fields':fields or [],'uncertainty':'No uncertainty intervals inferred; each execution bar is one recorded run and each logical cell is one measured panel output.','accessibility':'Direct text labels and values; meanings do not rely only on color. PNG plots also have scalable SVG and PDF counterparts.','generator':{'path':'build_figures.py','sha256':sha(Path(__file__).read_bytes())},'matplotlib_version':matplotlib.__version__,'font':{'name':'DejaVu Sans','sha256':sha(Path(font_manager.findfont('DejaVu Sans')).read_bytes())},'provenance':{'outputs':{k:sha(v) for k,v in outputs.items()}}}
 if reproduction:r['reproduction']=reproduction;r['uncertainty']='Reproduced raster; no new uncertainty or measurement inferred.'
 return r

def generate(index=None):
 sb=(ROOT/'receipts/evidence-summary.json').read_bytes();ab=(ROOT/'receipts/evidence-audit.json').read_bytes();assert sha(sb)==SUMMARY_SHA and sha(ab)==AUDIT_SHA
 d=json.loads(sb);full=json.loads(ab);check_evidence(d,full)
 schedule_binding=d['qwen_other']['generated_exchange']['schedule_receipt'];schedule_bytes=(ROOT/schedule_binding['path']).read_bytes();assert sha(schedule_bytes)==schedule_binding['sha256']
 schedule=json.loads(schedule_bytes);assert all(schedule[k]==v for k,v in d['qwen_other']['generated_exchange']['schedule'].items())
 assert schedule['write_updates']==120 and schedule['common_hold_updates']==240 and schedule['reference_starts']==2
 setup()
 old=json.loads((ROOT/'provenance.json').read_text()) if (ROOT/'provenance.json').exists() else {}
 out={};receipts={};figures=[]
 for group,fun,scope,fields in [('transfer-results',transfer,'Measured received payload sizes and receiver execution times for all four messages. No new inference.',['transfer.messages[].payload_bytes','transfer.messages[].seconds / 60']),('stateful-logic',logic,'Measured complete phase-zero input panels. Second gates use actual carried state. No inferred margins, universality or broader reliability.',['qwen_logic[].truth_table[].first_actual','qwen_logic[].truth_table[].final_actual'])]:
  assets,description=fun(d);out.update(assets);figures.extend(assets);receipts['fig-'+group+'.receipt.json']=figure_receipt(group,assets,description,d,scope,fields)
 for name,assets,desc in [('two-routes',routes(),d['conceptual_figures']['two-routes.svg']),('og-card',og(),d['conceptual_figures']['og-card.png'])]:
  out.update(assets);figures.extend(assets);receipts['fig-'+name+'.receipt.json']=figure_receipt(name,assets,desc,d,'Conceptual only; no empirical geometry or measured statistics.');receipts['fig-'+name+'.receipt.json']['uncertainty']='Not applicable: conceptual schematic.'
 for name,source in d['reproduced_images'].items():
  raw=(ROOT/name).read_bytes();assert sha(raw)==source['sha256'],'Reproduced raster differs: '+name
  out[name]=raw;figures.append(name);receipts['fig-'+name.removesuffix('.png')+'.receipt.json']=figure_receipt(name,{name:raw},source['alt_text'],d,source['caption'],reproduction=source)
 for name,r in receipts.items():out[name]=jsonbytes(r)
 manifest=dict(old);manreceipts=dict(old.get('receipts',{}));manreceipts.update({'receipts/evidence-summary.json':SUMMARY_SHA,'receipts/evidence-audit.json':AUDIT_SHA});manreceipts.update({k:sha(out[k]) for k in receipts});manreceipts[schedule_binding['path']]=schedule_binding['sha256']
 generated=expected_numbers(d);generated_ids={x['id'] for x in generated};extra=[x for x in old.get('numbers',[]) if x.get('id') not in generated_ids]
 # Retain root-added paper bindings on existing generated entries.
 oldnums={x.get('id'):x for x in old.get('numbers',[])};generated=[{**oldnums.get(x['id'],{}),**x} for x in generated]
 manifest.update(local_bundle=True,source_commit=d['source_commit'],generator={'path':'build_figures.py','sha256':sha(Path(__file__).read_bytes())},receipts=manreceipts,figures=figures,numbers=generated+extra,figure_sha256={k:sha(out[k]) for k in figures},source_evidence='receipts/evidence-summary.json',editorial_scope=old.get('editorial_scope','Figures synthesized from existing audited results and unchanged rasters; no new experiments.'))
 if index is not None:manifest['quoted_examples']=check_prose(index,d,full)
 out['provenance.json']=jsonbytes(manifest)
 return out

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true');parser.add_argument('--index',type=Path);a=parser.parse_args();idx=a.index or (ROOT/'index.md' if (ROOT/'index.md').exists() else None)
 outputs=generate(idx)
 if a.verify:
  with tempfile.TemporaryDirectory(prefix='shrimp-figure-verification-') as tmp:
   temp=Path(tmp)
   for name,raw in outputs.items():
    (temp/name).write_bytes(raw)
    assert (ROOT/name).read_bytes()==(temp/name).read_bytes(),'Regenerated bytes differ: '+name
  print(f'Verified {len(outputs)} generated or unchanged-copy outputs; prose checked: {idx is not None}.')
 else:
  for name,raw in outputs.items():(ROOT/name).write_bytes(raw)
  print(f'Wrote {len(outputs)} generated or unchanged-copy outputs; prose checked: {idx is not None}.')
