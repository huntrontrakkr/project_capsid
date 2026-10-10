import sys,json
import numpy as np
exec(open('relax.py').read().split('# --- relax')[0].replace("sys.argv[1]","'0'"))
# curvature: each necklace keeps its own 12-gon height profile, scaled down to fit inside its band
BAND=10.0
cen={k:np.mean([P[s][1] for s in g]) for k,g in enumerate(groups)}
spread=max(abs(P[s][1]-cen[stratum[s]]) for s in range(64))
c=BAND/spread
A=Q0.copy()
for s in range(64): A[s,1]=Yk(stratum[s])+c*(P[s][1]-cen[stratum[s]])
A=project(A) if 'project' in dir() else A
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
A=project(A)
print('curved start: crossings',crossings(A),'length',round(elen(A)))
lam=float(sys.argv[1])
Q=A.copy()
for it in range(3000):
    F=np.zeros((64,2));d=Q[EDG[:,1]]-Q[EDG[:,0]]
    np.add.at(F,EDG[:,0],d);np.add.at(F,EDG[:,1],-d)
    F+=lam*(A-Q)
    Q=project(Q+0.02*F)
print('lam',lam,'relaxed: crossings',crossings(Q),'length',round(elen(Q)),'flat length',round(elen(Q0)),'x',round(Q[:,0].min(),1),round(Q[:,0].max(),1))
json.dump(Q.round(1).tolist(),open(f'curv_{lam}.json','w'))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6,8))
for s,t2 in EDG: ax.plot(*zip(Q[s],Q[t2]),color='#999',lw=.4,zorder=1)
ax.scatter(Q[:,0],Q[:,1],c=[COL[canon(s)] for s in range(64)],s=60,edgecolors='k',linewidths=.4,zorder=2)
ax.invert_yaxis();ax.set_aspect('equal');ax.set_title(f'lam {lam}')
plt.savefig(f'curv_{lam}.png',dpi=80,bbox_inches='tight')
