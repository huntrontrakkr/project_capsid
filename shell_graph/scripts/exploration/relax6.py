import sys,json
import numpy as np
src=open('relax.py').read().split('# --- relax')[0].replace("sys.argv[1]","'0'").replace("SP=28","SP=float(sys.argv[2])")
exec(src)
BAND=float(sys.argv[3]); lam=float(sys.argv[1])
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
# restore the 12-gon's within-necklace curvature, each stratum's arc scaled to fill its band
A=Q0.copy()
for k,g in enumerate(groups):
    ys=np.array([P[s][1] for s in g]);mu=ys.mean();sp=np.abs(ys-mu).max()
    for s in g: A[s,1]=Yk(k)+(BAND*(P[s][1]-mu)/sp if sp>1e-6 else 0)
A=project(A)
print('curved: crossings',crossings(A),'length',round(elen(A)))
Q=A.copy()
for it in range(4000):
    d=Q[EDG[:,1]]-Q[EDG[:,0]];u=d/np.linalg.norm(d,axis=1,keepdims=True)
    F=np.zeros((64,2));np.add.at(F,EDG[:,0],u);np.add.at(F,EDG[:,1],-u)
    F[:,0]=0;F[:,1]+=lam*(A[:,1]-Q[:,1])
    Q=project(Q+0.1*F)
tag=f'{lam}_{SP:g}_{BAND:g}'
print(f'{tag}: crossings {crossings(Q)} length {round(elen(Q))} (flat {round(elen(Q0))} / {crossings(Q0)})  H {H}')
json.dump(Q.round(1).tolist(),open(f'r6_{tag}.json','w'))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
COL={0:'#B4B2A9',63:'#5F5E5A',1:'#F0997B',31:'#993C1D',3:'#85B7EB',15:'#185FA5',5:'#5DCAA5',23:'#0F6E56',9:'#AFA9EC',27:'#534AB7',7:'#BA7517',21:'#639922',11:'#ED93B1',13:'#993556'}
fig,axs=plt.subplots(1,2,figsize=(11,8))
for ax,QQ,t in ((axs[0],A,'curved start'),(axs[1],Q,'relaxed')):
    for s,t2 in EDG: ax.plot(*zip(QQ[s],QQ[t2]),color='#999',lw=.4,zorder=1)
    ax.scatter(QQ[:,0],QQ[:,1],c=[COL[canon(s)] for s in range(64)],s=60,edgecolors='k',linewidths=.4,zorder=2)
    ax.invert_yaxis();ax.set_aspect('equal');ax.set_title(t)
plt.savefig(f'r6_{tag}.png',dpi=75,bbox_inches='tight')
