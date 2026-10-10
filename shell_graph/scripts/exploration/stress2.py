import sys,json
import numpy as np
src=open('relax.py').read().split('# --- relax')[0].replace("sys.argv[1]","'0'").replace("SP=28","SP=float(sys.argv[1])")
exec(src)
BAND=float(sys.argv[2])
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
Ham=np.array([[bin(a^b).count('1') for b in range(64)] for a in range(64)],float)
D0=np.linalg.norm(Q0[:,None]-Q0[None],axis=-1)
mask=Ham>0
W=np.where(mask,1/np.maximum(Ham,1)**2,0)
scale=(W*D0*Ham).sum()/(W*Ham*Ham).sum()      # best-fit unit length for one edge
T=Ham*scale
def stress(Q):
    D=np.linalg.norm(Q[:,None]-Q[None],axis=-1);return float((W*(D-T)**2).sum()/2)
A=Q0.copy()
for k,g in enumerate(groups):
    ys=np.array([P[s][1] for s in g]);mu=ys.mean();sp=np.abs(ys-mu).max()
    for s in g: A[s,1]=Yk(k)+(BAND*(P[s][1]-mu)/sp if sp>1e-6 else 0)
def ratio(Q):
    D=np.linalg.norm(Q[:,None]-Q[None],axis=-1);e=D[EDG[:,0],EDG[:,1]].mean();return e/D[np.triu_indices(64,1)].mean()
print('flat ratio',round(ratio(Q0),3))
Q=project(A.copy());print('curved ratio',round(ratio(Q),3),'crossings',crossings(Q));print('start stress',round(stress(Q)),'unit',round(scale,1))
for it in range(3000):
    Dv=Q[:,None]-Q[None];D=np.linalg.norm(Dv,axis=-1)+np.eye(64)
    G=(W*(D-T)/D)[...,None]*Dv
    F=-G.sum(1)
    Q=project(Q+0.08*F)
Q[:,0]=190+(Q[:,0]-190)*170/np.abs(Q[:,0]-190).max()
tag=f'c{SP:g}_{BAND:g}'
print('final ratio',round(ratio(Q),3))
print(f'{tag}: stress {round(stress(Q))} crossings {crossings(Q)} length {round(elen(Q))} (flat {round(elen(Q0))} / {crossings(Q0)}) H {H}')
json.dump(Q.round(1).tolist(),open(f'st_{tag}.json','w'))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
COL={0:'#B4B2A9',63:'#5F5E5A',1:'#F0997B',31:'#993C1D',3:'#85B7EB',15:'#185FA5',5:'#5DCAA5',23:'#0F6E56',9:'#AFA9EC',27:'#534AB7',7:'#BA7517',21:'#639922',11:'#ED93B1',13:'#993556'}
fig,ax=plt.subplots(figsize=(6,8))
for s,t2 in EDG: ax.plot(*zip(Q[s],Q[t2]),color='#999',lw=.4,zorder=1)
ax.scatter(Q[:,0],Q[:,1],c=[COL[canon(s)] for s in range(64)],s=60,edgecolors='k',linewidths=.4,zorder=2)
ax.invert_yaxis();ax.set_aspect('equal');ax.set_title(tag)
plt.savefig(f'st_{tag}.png',dpi=75,bbox_inches='tight')
