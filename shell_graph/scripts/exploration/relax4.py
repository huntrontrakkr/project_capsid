import sys,json
import numpy as np
exec(open('relax.py').read().split('# --- relax')[0].replace("sys.argv[1]","'0'"))
BAND=float(sys.argv[2]); lam=float(sys.argv[1])
def project(Q):
    for k,g in enumerate(groups):
        m=sorted(g,key=lambda s:Q[s][0]);n=len(m)
        xs=sym_x(m,[Q[s][0] for s in m]);ys=[Q[s][1] for s in m]
        ys=[(ys[j]+ys[n-1-j])/2 for j in range(n)]
        for s,x,y in zip(m,xs,ys): Q[s]=(x,min(max(y,Yk(k)-BAND),Yk(k)+BAND))
    for s in range(64):
        t=comp(s)
        if s<t:
            a=(Q[s]+np.array([380,H])-Q[t])/2;Q[s]=a;Q[t]=np.array([380,H])-a
    return Q
Q=Q0.copy()
for it in range(6000):
    d=Q[EDG[:,1]]-Q[EDG[:,0]];u=d/np.linalg.norm(d,axis=1,keepdims=True)
    F=np.zeros((64,2));np.add.at(F,EDG[:,0],u);np.add.at(F,EDG[:,1],-u)   # pull along every edge: shortens total length
    F[:,0]+=lam*(Q0[:,0]-Q[:,0])
    Q=project(Q+0.15*F)
sc=170/np.abs(Q[:,0]-190).max();Q[:,0]=190+(Q[:,0]-190)*sc
print(f'lam {lam} band {BAND}: crossings {crossings(Q)} length {round(elen(Q))} (flat {round(elen(Q0))}, flat crossings {crossings(Q0)}) scale {sc:.2f}')
json.dump(Q.round(1).tolist(),open(f'r4_{lam}_{BAND}.json','w'))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
COL={0:'#B4B2A9',63:'#5F5E5A',1:'#F0997B',31:'#993C1D',3:'#85B7EB',15:'#185FA5',5:'#5DCAA5',23:'#0F6E56',9:'#AFA9EC',27:'#534AB7',7:'#BA7517',21:'#639922',11:'#ED93B1',13:'#993556'}
fig,ax=plt.subplots(figsize=(6,8))
for s,t2 in EDG: ax.plot(*zip(Q[s],Q[t2]),color='#999',lw=.4,zorder=1)
for k in range(15): ax.axhspan(Yk(k)-BAND,Yk(k)+BAND,color='#eee',zorder=0)
ax.scatter(Q[:,0],Q[:,1],c=[COL[canon(s)] for s in range(64)],s=60,edgecolors='k',linewidths=.4,zorder=2)
ax.invert_yaxis();ax.set_aspect('equal');ax.set_title(f'lam {lam} band {BAND}')
plt.savefig(f'r4_{lam}_{BAND}.png',dpi=80,bbox_inches='tight')
