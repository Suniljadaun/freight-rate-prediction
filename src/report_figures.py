"""Build the figures used in the written report (reports/figures/).

Run from the project root after python src/train.py:
    python src/report_figures.py
"""
import json
from pathlib import Path
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from preprocess import load_dataset
TEAL, ORANGE, GRAY, INK, MUTED, GRID = '#00889E','#D0782E','#B7C2C5','#1F2A2E','#5B6B70','#E3E9EA'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.edgecolor':'#9DAFB3','axes.labelcolor':INK,
 'xtick.color':MUTED,'ytick.color':MUTED,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.titlesize':12,'axes.titlelocation':'left'})
ROOT = Path(__file__).resolve().parents[1]
out = str(ROOT / 'reports' / 'figures') + '/'
Path(out).mkdir(parents=True, exist_ok=True)
t=load_dataset('train_test'); t['rpm']=t.posted_rate/t.distance
m=json.loads((ROOT / 'reports' / 'metrics.json').read_text())

# 1 market index vs rate per mile, indexed Jan=100
g=t.groupby(t.date.dt.to_period('M')).agg(mi=('market_index','mean'),rpm=('rpm',lambda s:s[(s>=1.3)&(s<=4)].mean()))
idx=g/g.iloc[0]*100; x=np.arange(len(idx)); lab=[p.strftime('%b') for p in idx.index]
fig,ax=plt.subplots(figsize=(8,3.8),dpi=200)
ax.plot(x,idx.mi,color=ORANGE,lw=2,marker='o',ms=5); ax.plot(x,idx.rpm,color=TEAL,lw=2,marker='o',ms=5)
ax.text(x[-1]+.15,idx.mi.iloc[-1],'market_index',color=INK,va='center',fontsize=9)
ax.text(x[-1]+.15,idx.rpm.iloc[-1],'rate per mile',color=INK,va='center',fontsize=9)
ax.set_xticks(x,lab); ax.set_xlim(-.3,len(x)+1.2); ax.axhline(100,color=GRID,lw=1,zorder=0)
ax.grid(axis='y',color=GRID,lw=.8); ax.set_ylabel('Index (Jan 2025 = 100)')
ax.set_title('market_index swings 40%; rate per mile moves ~10%')
fig.tight_layout(); fig.savefig(out+'market_vs_rate.png'); plt.close(fig)

# 2 rpm distribution with kept band
fig,ax=plt.subplots(figsize=(8,3.6),dpi=200)
bins=np.linspace(0,15,151)
ax.hist(t.rpm,bins=bins,color=TEAL,edgecolor='white',linewidth=.3)
ax.set_yscale('log'); ax.axvspan(1.3,4.0,color=TEAL,alpha=.07,zorder=0)
ax.axvline(1.3,color=MUTED,ls='--',lw=1); ax.axvline(4.0,color=MUTED,ls='--',lw=1)
ax.text(4.15,3000,'kept band\n\$1.30–\$4.00/mi',ha='left',color=INK,fontsize=9)
ax.text(0.65,250,'~0.3×\ncluster',ha='center',color=INK,fontsize=9); ax.text(8,40,'~3.5× cluster',ha='center',color=INK,fontsize=9)
ax.set_xlabel('Rate per mile (\$)'); ax.set_ylabel('Loads (log scale)'); ax.grid(axis='y',color=GRID,lw=.8)
ax.set_title('Rate per mile: 677 label errors sit in two clusters outside the normal band')
fig.tight_layout(); fig.savefig(out+'rpm_distribution.png'); plt.close(fig)

# 3 model comparison
mean=pd.DataFrame(m['mean']).set_index('model')
order=['Naive (equipment rpm)','Ridge','LightGBM + market_index','LightGBM (final)']
names=['Naive baseline','Ridge regression','LightGBM + market_index','LightGBM (final)']
vals=mean.loc[order,'MAE'].values; mape=mean.loc[order,'MAPE_%'].values
fig,ax=plt.subplots(figsize=(8,2.9),dpi=200)
y=np.arange(len(order))[::-1]
ax.barh(y,vals,color=[GRAY,GRAY,GRAY,TEAL],height=.6)
for yi,v,p in zip(y,vals,mape): ax.text(v+3,yi,f'\${v:,.2f}  ({p:.2f}% MAPE)',va='center',color=INK,fontsize=9)
ax.set_yticks(y,names); ax.set_xlim(0,max(vals)*1.45); ax.set_xlabel('Mean absolute error on total rate (\$), mean of Sep & Oct folds')
ax.grid(axis='x',color=GRID,lw=.8); ax.set_axisbelow(True)
ax.set_title('Out-of-time backtest: model comparison')
fig.tight_layout(); fig.savefig(out+'model_comparison.png'); plt.close(fig)

# 4 fold timeline
fig,ax=plt.subplots(figsize=(8,2.1),dpi=200)
months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
rows=[('Fold 1',8,1,None),('Fold 2',9,1,None),('Final',10,0,2)]
for i,(name,ntr,nte,npred) in enumerate(rows):
    yy=2-i
    ax.barh(yy,ntr,left=0,color=TEAL,height=.55,edgecolor='white')
    if nte: ax.barh(yy,nte,left=ntr,color=ORANGE,height=.55,edgecolor='white')
    if npred: ax.barh(yy,npred,left=ntr,color='white',height=.55,edgecolor=MUTED,hatch='///',linewidth=.8)
ax.set_yticks([2,1,0],[r[0] for r in rows]); ax.set_xticks(np.arange(12)+.5,months); ax.tick_params(axis='x',length=0)
ax.set_xlim(0,12); ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
ax.text(4,2,'train',color='white',ha='center',va='center',fontsize=9,fontweight='bold')
ax.text(8.5,2,'test',color='white',ha='center',va='center',fontsize=9,fontweight='bold')
ax.text(4.5,1,'train',color='white',ha='center',va='center',fontsize=9,fontweight='bold'); ax.text(5,0,'train',color='white',ha='center',va='center',fontsize=9,fontweight='bold'); ax.text(9.5,1,'test',color='white',ha='center',va='center',fontsize=9,fontweight='bold')
ax.text(11,0,'predict',color=INK,ha='center',va='center',fontsize=9,bbox=dict(fc='white',ec='none',pad=1))
ax.set_title('Expanding-window, out-of-time validation (2025)')
fig.tight_layout(); fig.savefig(out+'fold_timeline.png'); plt.close(fig)

# 5 feature importance top 8
imp=pd.Series(m['feature_importance_pct']).sort_values(ascending=False).head(8)[::-1]
fig,ax=plt.subplots(figsize=(8,3),dpi=200)
ax.barh(imp.index,imp.values,color=TEAL,height=.6)
for i,v in enumerate(imp.values): ax.text(v+.5,i,f'{v:.1f}%',va='center',color=INK,fontsize=9)
ax.set_xlabel('Share of total gain (%)'); ax.grid(axis='x',color=GRID,lw=.8); ax.set_axisbelow(True); ax.set_xlim(0,imp.max()*1.15)
ax.set_title('LightGBM feature importance (top 8)')
fig.tight_layout(); fig.savefig(out+'feature_importance.png'); plt.close(fig)

print(f'Saved report figures to {out}')
