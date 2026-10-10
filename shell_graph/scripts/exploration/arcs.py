import sys,json
import numpy as np
from scipy.optimize import minimize
src=open('relax.py').read().split('# --- relax')[0].replace("sys.argv[1]","'0'").replace("SP=28","SP=30")
exec(src)
K=len(groups)  # 15 strata, index 7 = center line
xs_grid=np.linspace(15,365,71)
def layout(kap):
    kk=np.zeros(K)
    for k in range(7): kk[k]=kap[k];kk[14-k]=-kap[k]     # open/filled flip mirrors the bends
    Q=Q0.copy()
    for s in range(64):
        k=stratum[s];Q[s,1]=Yk(k)+kk[k]*(Q[s,0]-190)**2
    return Q,kk
def objective(kap):
    Q,kk=layout(kap)
    curves=np.array([[Yk(k)+kk[k]*(x-190)**2 for x in xs_grid] for k in range(K)])
    gap=curves[1:]-curves[:-1]
    pen=np.clip(14-gap,0,None).sum()*500                     # strata stay at least 14px apart everywhere
    return elen(Q)+pen
best=None
for start in [np.zeros(7),np.full(7,.002),np.full(7,-.002)]:
    r=minimize(objective,start,method='Nelder-Mead',options={'maxiter':6000,'xatol':1e-7,'fatol':1e-3})
    if best is None or r.fun<best.fun: best=r
Q,kk=layout(best.x)
print('kappa',np.round(kk*1e4,2))
print(f'arcs: length {round(elen(Q))} crossings {crossings(Q)}  (flat {round(elen(Q0))} / {crossings(Q0)})  y range {Q[:,1].min():.0f}-{Q[:,1].max():.0f}')
json.dump({'Q':Q.round(1).tolist(),'kk':kk.tolist(),'H':H},open('arcs.json','w'))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
COL={0:'#B4B2A9',63:'#5F5E5A',1:'#F0997B',31:'#993C1D',3:'#85B7EB',15:'#185FA5',5:'#5DCAA5',23:'#0F6E56',9:'#AFA9EC',27:'#534AB7',7:'#BA7517',21:'#639922',11:'#ED93B1',13:'#993556'}
fig,ax=plt.subplots(figsize=(6,8))
for k in range(K): ax.plot(xs_grid,[Yk(k)+kk[k]*(x-190)**2 for x in xs_grid],color='#ddd',lw=.6,zorder=0)
for s,t2 in EDG: ax.plot(*zip(Q[s],Q[t2]),color='#999',lw=.4,zorder=1)
ax.scatter(Q[:,0],Q[:,1],c=[COL[canon(s)] for s in range(64)],s=60,edgecolors='k',linewidths=.4,zorder=2)
ax.invert_yaxis();ax.set_aspect('equal')
plt.savefig('arcs.png',dpi=75,bbox_inches='tight')
