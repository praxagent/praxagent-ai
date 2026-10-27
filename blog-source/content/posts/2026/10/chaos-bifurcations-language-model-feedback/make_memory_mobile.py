#!/usr/bin/env python3
"""Stack the audited memory-continuation panels for narrow screens.

This is a display-only rendering of fig-memory-continuation.receipt.json.
Run with --verify to compare a fresh rendering byte-for-byte with the PNG.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BUNDLE = Path(__file__).resolve().parent
SOURCE = BUNDLE / 'fig-memory-continuation.receipt.json'
SOURCE_SHA256 = 'bb166748c2d71f2dd19f2d1190ba4739c7529bdb56b05b2b96e439d947757e27'


def render():
    data = SOURCE.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('audited plot data receipt changed')
    receipt = json.loads(data)
    rows = receipt['plotted_data']
    if len(rows) != 18 or len({p['beta'] for p in rows}) != 18:
        raise ValueError('expected all eighteen distinct accepted samples')
    root = next(p for p in rows if p['beta'] == receipt['highlighted_root_beta'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2, 1, figsize=(4.7, 8.1), gridspec_kw={'height_ratios':[1,1]})
    fig.subplots_adjust(left=.21, right=.88, top=.82, bottom=.14, hspace=.64)
    fig.text(.08,.96,"Following the child's",fontsize=18,fontweight='bold')
    fig.text(.08,.925,'trace saddle',fontsize=18,fontweight='bold')
    beta = [p['memory_percent'] for p in rows]
    tau = [p['tau'] for p in rows]
    axes[0].axhline(0, color='#666666',lw=1,linestyle='--')
    axes[0].plot(beta,tau,'o-',color='#175d8d',lw=1.5,ms=4)
    axes[0].plot(root['memory_percent'],root['tau'],'*',color='#e68424',markeredgecolor='#713b08',ms=12)
    axes[0].set(xlabel='Feedback memory (%)',ylabel='Pair trace at saddle',title='Measured saddle height')
    axes[0].ticklabel_format(style='sci',axis='y',scilimits=(-3,3),useOffset=False)
    axes[0].set_xticks([0,.2,.4,.6])
    x=[p['scaled_center'][0] for p in rows]; y=[p['scaled_center'][1] for p in rows]
    axes[1].plot(x,y,color='#b0b5ba',lw=1)
    sc=axes[1].scatter(x,y,c=beta,cmap='viridis',s=38,edgecolors='white',linewidths=.5,zorder=2)
    axes[1].plot(*root['scaled_center'],'*',color='#e68424',markeredgecolor='#713b08',ms=12,zorder=3)
    axes[1].set(xlabel='Scaled steering displacement',ylabel='Scaled log-temperature displacement',title='The moving center')
    axes[1].set_xticks([0,50,100]);axes[1].set_yticks([-500,-250,0])
    colorbar=fig.colorbar(sc,ax=axes[1],fraction=.055,pad=.04,ticks=[0,.35,.7])
    colorbar.set_label('Memory (%)',fontsize=10);colorbar.ax.tick_params(labelsize=10)
    for ax in axes:ax.grid(alpha=.18)
    fig.text(.08,.032,f"Star: numerical estimate, {root['memory_percent']:.3f}%",fontsize=11)
    out=io.BytesIO();fig.savefig(out,format='png',dpi=150);plt.close(fig)
    return out.getvalue()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    output=BUNDLE/'fig-memory-continuation-mobile.png';data=render()
    if args.verify:
        if output.read_bytes()!=data:raise ValueError('mobile PNG differs from fresh rendering')
        print('Mobile memory figure reproduces byte-for-byte.')
    else:
        output.write_bytes(data);print('Wrote',output.name)


if __name__=='__main__':main()
