import math, json, itertools, sys
import numpy as np
rot=lambda s,k:((s<<k)|(s>>(6-k)))&63 if k%6 else s
canon=lambda s:min(rot(s,k) for k in range(6))
pc=lambda s:bin(s).count('1'); comp=lambda s:63^s
pat=lambda s:''.join('●' if (s>>i)&1 else '○' for i in range(6))
P={}
for s in range(64):
    x=y=0
    for i in range(6):
        c=((s>>i)&1)-.5;a=math.radians(i*30+15);x+=c*math.cos(a);y+=c*math.sin(a)
    P[s]=np.array([190+88*x,250+88*y])
cls={}
for s in range(64): cls.setdefault(canon(s),[]).append(s)
EDG=np.array([(s,s^(1<<i)) for s in range(64) for i in range(6) if s<s^(1<<i)])
def crossings(Q):
    A=Q[EDG[:,0]];B=Q[EDG[:,1]]
    def o(p,q,r): return np.sign((q[...,0]-p[...,0])*(r[...,1]-p[...,1])-(q[...,1]-p[...,1])*(r[...,0]-p[...,0]))
    a=A[:,None];b=B[:,None];c=A[None];d=B[None]
    sh=np.zeros((len(EDG),)*2,bool)
    for i in (0,1):
        for j in (0,1): sh|=EDG[:,i][:,None]==EDG[:,j][None,:]
    return int((((o(a,b,c)*o(a,b,d))<0)&((o(c,d,a)*o(c,d,b))<0)&~sh).sum())//2
def elen(Q): return float(np.linalg.norm(Q[EDG[:,0]]-Q[EDG[:,1]],axis=1).sum())
# --- strata: amber split by 12-gon height, center pair decided by flag
amber=sorted(cls[7],key=lambda s:(P[s][1],P[s][0]))
flag=int(sys.argv[1]) if len(sys.argv)>1 else 0
mid=amber[2:4] if flag==0 else amber[3:1:-1]
amU=amber[:2]+[mid[0]]; amD=[comp(s) for s in amU]
assert sorted(amU+amD)==sorted(cls[7])
groups=[[0],cls[1],cls[3],cls[9],cls[5],cls[11],amU,cls[21],amD,cls[13],cls[23],cls[27],cls[15],cls[31],[63]]
SP=28;Y0=40;H=2*(Y0+7*SP)  # center line = green
stratum={}
for k,g in enumerate(groups):
    for s in g: stratum[s]=k
Yk=lambda k:Y0+k*SP
# --- stratified start: x from 12-gon, symmetrized, min gap
GAP=17
def sym_x(order,xs):
    n=len(xs);xs=[190+(xs[k]-xs[n-1-k])/2 for k in range(n)]
    for _ in range(100):
        for k in range(n-1):
            d=xs[k+1]-xs[k]
            if d<GAP: xs[k]-=(GAP-d)/2;xs[k+1]+=(GAP-d)/2
        xs=[190+(xs[k]-xs[n-1-k])/2 for k in range(n)]
    return xs
Q=np.zeros((64,2))
for k,g in enumerate(groups):
    m=sorted(g,key=lambda s:(round(P[s][0],6),P[s][1]))
    for s,x in zip(m,sym_x(m,[P[s][0] for s in m])): Q[s]=(x,Yk(k))
Q0=Q.copy()
print('flat: crossings',crossings(Q),'length',round(elen(Q)))
# --- relax: springs on edges pull neighbors together; stratum mates repel; y may drift inside its band
BAND=8.5
def project(Q):
    for k,g in enumerate(groups):
        m=sorted(g,key=lambda s:Q[s][0]);n=len(m)
        xs=[Q[s][0] for s in m];ys=[Q[s][1] for s in m]
        xs=sym_x(m,xs)
        ys=[(ys[j]+ys[n-1-j])/2 for j in range(n)]
        for s,x,y in zip(m,xs,ys): Q[s]=(x,min(max(y,Yk(k)-BAND),Yk(k)+BAND))
    for s in range(64):  # open/filled flip = point reflection through the center
        t=comp(s)
        if s<t:
            a=(Q[s]+np.array([380,H])-Q[t])/2
            Q[s]=a;Q[t]=np.array([380,H])-a
    return Q
same=np.array([[stratum[a]==stratum[b] and a!=b for b in range(64)] for a in range(64)])
step=0.02
for it in range(4000):
    F=np.zeros((64,2))
    d=Q[EDG[:,1]]-Q[EDG[:,0]]
    np.add.at(F,EDG[:,0],d);np.add.at(F,EDG[:,1],-d)       # springs
    D=Q[:,None,:]-Q[None,:,:];r2=(D**2).sum(-1)+1e-6
    rep=(D/r2[...,None]**1.5)*same[...,None]*float(sys.argv[2] if len(sys.argv)>2 else 4e5)
    F+=rep.sum(1)
    F[:,1]*=0.5
    Q=project(Q+step*F)
print('relaxed: crossings',crossings(Q),'length',round(elen(Q)),'x range',round(Q[:,0].min(),1),round(Q[:,0].max(),1))
json.dump(Q.round(1).tolist(),open(f'relax_{flag}.json','w'))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
COL={0:'#B4B2A9',63:'#5F5E5A',1:'#F0997B',31:'#993C1D',3:'#85B7EB',15:'#185FA5',5:'#5DCAA5',23:'#0F6E56',9:'#AFA9EC',27:'#534AB7',7:'#BA7517',21:'#639922',11:'#ED93B1',13:'#993556'}
fig,ax=plt.subplots(1,2,figsize=(12,8))
for a,QQ,t in ((ax[0],Q0,'flat'),(ax[1],Q,'relaxed')):
    for s,t2 in EDG: a.plot(*zip(QQ[s],QQ[t2]),color='#999',lw=.4,zorder=1)
    a.scatter(QQ[:,0],QQ[:,1],c=[COL[canon(s)] for s in range(64)],s=60,edgecolors='k',linewidths=.4,zorder=2)
    a.invert_yaxis();a.set_aspect('equal');a.set_title(t)
plt.savefig(f'relax_{flag}.png',dpi=80,bbox_inches='tight')
