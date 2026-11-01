#!/usr/bin/env python3
"""Regenerate one figure from the bundled measured loop receipt; no model access."""
from pathlib import Path
import argparse,hashlib,json,tempfile,os
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'qwen-hysteresis-mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE=Path(__file__).resolve().parent
DATA=Path('receipts/hysteresis-loops.json')
STEM='qwen-hysteresis-loops'
PAPER='#faf7ef';INK='#272923';MUTED='#65665d';BLUE='#246b82';RUST='#ad522e';GRID='#ded9cd'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'figure.facecolor':PAPER,
 'axes.facecolor':PAPER,'axes.edgecolor':GRID,'axes.labelcolor':INK,'text.color':INK,
 'xtick.color':MUTED,'ytick.color':INK,'svg.hashsalt':'qwen-measured-hysteresis-v1','pdf.fonttype':42})
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(d):
 assert d['schema']=='qwen-hysteresis-loops-v1'
 assert [(x['fraction'],x['dwell']) for x in d['groups']]==[(.5,240),(.5,480),(.25,480),(.75,480)]
 rows=[r for g in d['groups'] for r in g['rows']]
 assert len(rows)==16 and sum(r['gap'] is None for r in rows)==3
 for g in d['groups']:
  assert [(r['direction'],r['initial_history']) for r in g['rows']]==[(0,0),(0,1),(1,0),(1,1)]
  for r in g['rows']:
   assert r['final_hold_bit']==r['direction']
   assert (r['gap'] is not None)==(r['outward_period']==r['return_period']==6)
   assert r['endpoint_period']==(6 if r['direction']==0 else None)
   target=d['controls']['E'+str(r['direction'])]
   assert r['endpoint_controls']==target and r['final_controls']==d['controls']['H']
   assert all(r['interior_controls'][k]==d['controls']['H'][k]+g['fraction']*(target[k]-d['controls']['H'][k]) for k in target)

def generate(d,out):
 check(d);out.mkdir(parents=True,exist_ok=True)
 fig=plt.figure(figsize=(9.4,12.2),facecolor=PAPER)
 top=fig.add_axes([.04,.775,.93,.205]);top.set(xlim=(0,1),ylim=(0,1));top.axis('off')
 top.text(0,.99,'The same controls, a different history',fontsize=22,weight='bold',va='top')
 top.text(0,.82,'Recorded Qwen loops · every fraction, dwell and starting history',fontsize=12.5,color=MUTED)
 xs=[.07,.285,.5,.715,.93];labels=['H',r'$P_f$',r'$E_b$',r'$P_f$','H']
 for i,(x,label) in enumerate(zip(xs,labels)):
  top.add_patch(FancyBboxPatch((x-.045,.43),.09,.19,boxstyle='round,pad=.014',
    facecolor='#e3ece6' if i in (1,3) else '#f0ece1',edgecolor=BLUE if i in (1,3) else GRID,lw=1.4))
  top.text(x,.525,label,ha='center',va='center',fontsize=17,weight='bold')
  if i<4:top.annotate('',xy=(xs[i+1]-.062,.525),xytext=(x+.062,.525),arrowprops={'arrowstyle':'->','color':MUTED,'lw':1.4})
 top.text(xs[0],.33,'start',ha='center',fontsize=11,color=MUTED)
 top.text(xs[1],.33,'outward',ha='center',fontsize=11,color=MUTED)
 top.text(xs[2],.33,'endpoint',ha='center',fontsize=11,color=MUTED)
 top.text(xs[3],.33,'return',ha='center',fontsize=11,color=MUTED)
 top.text(xs[4],.33,'read',ha='center',fontsize=11,color=MUTED)
 top.plot([xs[1],xs[1],xs[3],xs[3]],[.27,.22,.22,.27],color=BLUE,lw=1.3)
 top.text(.505,.1,'Compare the two visits within each original history',ha='center',fontsize=12.5,color=BLUE)
 fig.text(.06,.752,r'$P_f=H+f(E_b-H)$; the stated dwell applies to all four visited stages.',fontsize=12.3)
 fig.text(.06,.727,'Each direction starts separately from H0 and H1 at reference phase zero.',fontsize=12.3)

 # Whole-width annotation panel and a shared-y logarithmic plot.
 bottom,height=.153,.535
 base=fig.add_axes([.055,bottom,.90,height]);base.set(xlim=(0,1),ylim=(19.6,-2.3));base.axis('off')
 plot=fig.add_axes([.325,bottom,.32,height],sharey=base)
 plot.set_xscale('log');plot.set_xlim(1e-15,6e-2);plot.set_ylim(19.6,-2.3)
 plot.set_xticks([1e-14,1e-10,1e-6,1e-2]);plot.tick_params(axis='x',labelsize=11.5,pad=7,length=0)
 plot.yaxis.set_visible(False);plot.grid(axis='x',color=GRID,lw=.7,zorder=0)
 for name,spine in plot.spines.items():spine.set_visible(name=='bottom')
 plot.set_xlabel('Phase-aligned cycle gap (L2)',fontsize=12.3,labelpad=10)
 base.text(.016,-1.55,'Direction / original lane',fontsize=11.5,weight='bold')
 base.text(.745,-1.55,r'$P_f$ periods'+'\nout / return',ha='center',va='center',fontsize=11.2,weight='bold')
 base.text(.875,-1.55,r'$E_b$'+'\nperiod',ha='center',va='center',fontsize=11.2,weight='bold')
 base.text(.985,-1.55,'Final H\nbit',ha='center',va='center',fontsize=11.2,weight='bold')
 for gi,g in enumerate(d['groups']):
  y0=gi*5.1
  for ax in (base,plot):ax.axhspan(y0-.72,y0-.28,color='#ebe6da',zorder=1)
  base.text(.016,y0-.5,f"f = {g['fraction']:.2f}   ·   {g['dwell']} updates",fontsize=12.5,weight='bold',va='center',zorder=5)
  for ri,r in enumerate(g['rows']):
   y=y0+ri+.33;color=BLUE if r['initial_history']==0 else RUST;marker='o' if r['initial_history']==0 else 'D'
   base.text(.026,y,f"E{r['direction']}  /  H{r['initial_history']}",fontsize=12.2,va='center',color=color)
   if r['gap'] is None:
    plot.text(.5,y,'unresolved',transform=plot.get_yaxis_transform(),ha='center',va='center',fontsize=11.8,color=RUST,
     bbox={'facecolor':PAPER,'edgecolor':'none','pad':2.0},zorder=4)
   else:plot.scatter([r['gap']],[y],s=67,c=color,marker=marker,edgecolors=PAPER,linewidths=.8,zorder=5)
   periods=lambda p:'U' if p is None else str(p)
   pair=periods(r['outward_period'])+' / '+periods(r['return_period'])
   base.text(.745,y,pair,ha='center',va='center',fontsize=12.2,color=RUST if 'U' in pair else INK)
   base.text(.875,y,periods(r['endpoint_period']),ha='center',va='center',fontsize=12.2,color=RUST if r['endpoint_period'] is None else INK)
   base.text(.985,y,str(r['final_hold_bit']),ha='center',va='center',fontsize=12.2,weight='bold')
 fig.text(.06,.087,'6: qualified primitive period six.  U: no qualified period from 1 to 64.',fontsize=11.7)
 fig.text(.06,.062,'A gap compares one lane across two visits, never H0 against H1.',fontsize=11.7)
 fig.text(.06,.037,'Finite-dwell history dependence; branch loss and long-term stability remain unproved.',fontsize=11.2,color=MUTED)
 for ext in ('png','svg','pdf'):
  p=out/(STEM+'.'+ext)
  meta={'Software':'Matplotlib; build_hysteresis_figure.py'} if ext=='png' else {'Date':None,'Creator':'build_hysteresis_figure.py'} if ext=='svg' else {'CreationDate':None,'ModDate':None,'Creator':'build_hysteresis_figure.py'}
  fig.savefig(p,dpi=200,metadata=meta,facecolor=PAPER)
  if ext=='svg':p.write_bytes(b'\n'.join(line.rstrip() for line in p.read_bytes().splitlines())+b'\n')
 plt.close(fig)

def metadata(d,out):
 numbers=[]
 def num(id,text,src,claim):numbers.append({'id':id,'appears_as':text,'source':str(DATA)+'#'+src,'claim':claim})
 for gi,g in enumerate(d['groups']):
  num(f'loop-{gi}-configuration',f"f = {g['fraction']:.2f}   ·   {g['dwell']} updates",f'/groups/{gi}','Fraction and dwell, fixed for all four stages.')
  for ri,r in enumerate(g['rows']):
   src=f'/groups/{gi}/rows/{ri}'
   for key in ('outward_period','return_period','endpoint_period','final_hold_bit'):
    num(f'loop-{gi}-{ri}-{key}','U' if r[key] is None else str(r[key]),src+'/'+key,'Direct saved outcome; U retained distinctly from a resolved period or bit.')
   numbers.append({'id':f'loop-{gi}-{ri}-gap','source':str(DATA)+'#'+src+'/gap','plotted_value':r['gap'],'claim':'Same original history, outward P_f versus return P_f; null is shown as unresolved and has no numeric plotted coordinate.'})
 assets={STEM+'.'+ext:sha(out/(STEM+'.'+ext)) for ext in ('png','svg','pdf')}
 receipt={'schema':'qwen-hysteresis-figure-v1','generator':{'file':Path(__file__).name,'sha256':sha(__file__)},
  'data':{'file':str(DATA),'sha256':sha(HERE/DATA)},'outputs':assets,'provenance':{'outputs':assets},'numbers':numbers,
  'comparison_definition':d['metric'],'scope':d['scope'],
  'verification':'--verify regenerates PNG/SVG/PDF byte-for-byte from the bundled compact receipt. This checks presentation, not model inference or full scientific raw-array audits.',
  'style':'Measured points and status cells; schematic path at top. No interpolated measured branch, fitted uncertainty, numerical value for unresolved rows or omitted endpoint failures.'}
 (out/('fig-'+STEM+'.receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
 (out/'qwen-hysteresis-numerical-provenance.json').write_text(json.dumps(numbers,indent=2)+'\n')
 prov={'local_bundle':True,'generator':Path(__file__).name,'receipts':[str(DATA),'fig-'+STEM+'.receipt.json'],
  'figures':list(assets),'figure_sha256':assets,'numbers':numbers,'source_evidence':{'receipt':str(DATA),'sha256':sha(HERE/DATA)},'editorial_scope':d['scope']}
 (out/'qwen-hysteresis-provenance.json').write_text(json.dumps(prov,indent=2)+'\n')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',action='store_true');a=p.parse_args()
 d=json.loads((HERE/DATA).read_text());check(d)
 if a.verify:
  old=json.loads((HERE/('fig-'+STEM+'.receipt.json')).read_text())
  assert old['data']['sha256']==sha(HERE/DATA) and old['generator']['sha256']==sha(__file__)
  with tempfile.TemporaryDirectory(prefix='qwen-hysteresis-verify-') as tmp:
   out=Path(tmp);generate(d,out);metadata(d,out)
   files=[STEM+'.'+e for e in ('png','svg','pdf')]+['fig-'+STEM+'.receipt.json','qwen-hysteresis-numerical-provenance.json','qwen-hysteresis-provenance.json']
   for name in files:assert (out/name).read_bytes()==(HERE/name).read_bytes(),'regeneration mismatch: '+name
  print('Verified deterministic PNG/SVG/PDF, figure receipt, numerical provenance and manifest;16 comparison rows, including3 unresolved.')
 else:generate(d,HERE);metadata(d,HERE);print('Wrote measured-loop figure and provenance.')
if __name__=='__main__':main()
