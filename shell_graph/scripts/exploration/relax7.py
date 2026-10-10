import sys,json
import numpy as np
src=open('relax.py').read().split('# --- relax')[0].replace("sys.argv[1]","'0'").replace("SP=28","SP=float(sys.argv[1])")
exec(src)
BAND=float(sys.argv[2])
ends={}
for k,g in enumerate(groups):
    m=sorted(g,key=lambda s:Q0[s][0]); ends[k]=(Q0[m[0]][0],Q0[m[-1]][0])
def project(Q):
    for k,g in enumerate(groups):
        m=sorted(g,key=lambda s:Q[s][0]);n=len(m)
        xs=sym_x(m,[Q[s][0] for s in m]);ys=[Q[s][1] for s in m]
        if n>1: xs[0],xs[-1]=ends[k]          # each stratum keeps its 12-gon width
        ys=[(ys[j]+ys[n-1-j])/2 for j in range(n)]
        for s,x,y in zip(m,xs,ys): Q[s]=(x,min(max(y,Yk(k)-BAND),Yk(k)+BAND))
    for s in range(64):
        t=comp(s)
        if s<t:
            a=(Q[s]+np.array([380,H])-Q[t])/2;Q[s]=a;Q[t]=np.array([380,H])-a
    return Q
Q=project(Q0.copy())
for it in range(6000):
    d=Q[EDG[:,1]]-Q[EDG[:,0]];u=d/np.linalg.norm(d,axis=1,keepdims=True)
    F=np.zeros((64,2));np.add.at(F,EDG[:,0],u);np.add.at(F,EDG[:,1],-u)
    Q=project(Q+0.1*F)
tag=f'{SP:g}_{BAND:g}'
print(f'{tag}: crossings {crossings(Q)} length {round(elen(Q))} (flat {round(elen(Q0))} / {crossings(Q0)}) H {H}')
json.dump(Q.round(1).tolist(),open(f'r7_{tag}.json','w'))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
COL={0:'#B4B2A9',63:'#5F5E5A',1:'#F0997B',31:'#993C1D',3:'#85B7EB',15:'#185FA5',5:'#5DCAA5',23:'#0F6E56',9:'#AFA9EC',27:'#534AB7',7:'#BA7517',21:'#639922',11:'#ED93B1',13:'#993556'}
fig,ax=plt.subplots(figsize=(6,8))
for s,t2 in EDG: ax.plot(*zip(Q[s],Q[t2]),color='#999',lw=.4,zorder=1)
ax.scatter(Q[:,0],Q[:,1],c=[COL[canon(s)] for s in range(64)],s=60,edgecolors='k',linewidths=.4,zorder=2)
ax.invert_yaxis();ax.set_aspect('equal');ax.set_title(tag)
plt.savefig(f'r7_{tag}.png',dpi=75,bbox_inches='tight')
