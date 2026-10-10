import math, json
rot=lambda s,k:((s<<k)|(s>>(6-k)))&63 if k%6 else s
canon=lambda s:min(rot(s,k) for k in range(6))
pat=lambda s:''.join('●' if (s>>i)&1 else '○' for i in range(6))
# 12-gon positions (same projection as the widget), center (190,250), scale 88
P={}
for s in range(64):
    x=y=0
    for i in range(6):
        c=((s>>i)&1)-.5;a=math.radians(i*30+15);x+=c*math.cos(a);y+=c*math.sin(a)
    P[s]=[190+88*x,250+88*y]
cls={}
for s in range(64): cls.setdefault(canon(s),[]).append(s)
# step 1: each class's y-centroid
cen={c:sum(P[s][1] for s in m)/len(m) for c,m in cls.items()}
for c in sorted(cls,key=lambda c:cen[c]): print(pat(c),len(cls[c]),round(cen[c],2), 'ys', sorted(round(P[s][1],1) for s in cls[c]))

import itertools
import numpy as np
EDG=np.array([(s,s^(1<<i)) for s in range(64) for i in range(6) if s<s^(1<<i)])
def crossings(Q):
    A=np.array([Q[s] for s in EDG[:,0]]);B=np.array([Q[s] for s in EDG[:,1]])
    def o(p,q,r): return np.sign((q[...,0]-p[...,0])*(r[...,1]-p[...,1])-(q[...,1]-p[...,1])*(r[...,0]-p[...,0]))
    a=A[:,None];b=B[:,None];c=A[None];d=B[None]
    sh=np.zeros((len(EDG),)*2,bool)
    for i in (0,1):
        for j in (0,1): sh|=EDG[:,i][:,None]==EDG[:,j][None,:]
    return int((((o(a,b,c)*o(a,b,d))<0)&((o(c,d,a)*o(c,d,b))<0)&~sh).sum())//2
# step 2: flatten each class onto one horizontal line; x = 12-gon x, symmetrized about the center, min gap enforced
def flat_x(c,gap=19):
    m=sorted(cls[c],key=lambda s:(round(P[s][0],6),P[s][1]))
    xs=[P[s][0] for s in m];n=len(xs)
    sym=[190+(xs[k]-xs[n-1-k])/2 for k in range(n)]
    # push apart symmetrically from the center until every neighbor gap >= gap
    for _ in range(200):
        for k in range(n-1):
            d=sym[k+1]-sym[k]
            if d<gap:
                sym[k]-=(gap-d)/2;sym[k+1]+=(gap-d)/2
        sym=[190+(sym[k]-sym[n-1-k])/2 for k in range(n)]
    return dict(zip(m,sym))
X={}
for c in cls: X.update(flat_x(c))
comp=lambda s:63^s
T2=[3,5,9]; s=28
best=None
for perm in itertools.permutations(T2):
    for amber_up in (True,False):
        order_top=[0,1]+list(perm)
        mid=[11,7,21,None,13] if amber_up else [11,None,21,7,13]
        bot=[canon(comp(c)) for c in reversed(order_top)]
        strata=order_top+mid+bot
        Y={}
        for k,c in enumerate(strata):
            if c is None: continue
            for m in cls[c]: Y[m]=40+k*s
        Q={m:(X[m],Y[m]) for m in range(64)}
        v=crossings(Q)
        if best is None or v<best[0]: best=(v,strata,Q)
        print([pat(c) if c is not None else '-' for c in strata][2:5],amber_up,v)
v,strata,Q=best
print('best',v,[pat(c) if c is not None else '-' for c in strata])
print('12-gon crossings',crossings({m:tuple(P[m]) for m in range(64)}))
print('x range',min(q[0] for q in Q.values()),max(q[0] for q in Q.values()))
json.dump({'Q':[[round(Q[m][0],1),Q[m][1]] for m in range(64)],'strata':strata},open('strat.json','w'))
print(json.dumps([[round(Q[m][0],1),Q[m][1]] for m in range(64)],separators=(',',':')))
