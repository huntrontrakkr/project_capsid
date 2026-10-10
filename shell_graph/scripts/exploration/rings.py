import math, json, random, itertools
import numpy as np
from scipy.optimize import minimize
rot=lambda s,k:((s<<k)|(s>>(6-k)))&63 if k%6 else s
canon=lambda s:min(rot(s,k) for k in range(6))
comp=lambda s:63^s
P={}
for s in range(64):
    x=y=0
    for i in range(6):
        c=((s>>i)&1)-.5;a=math.radians(i*30+15);x+=c*math.cos(a);y+=c*math.sin(a)
    P[s]=np.array([190+88*x,250+88*y])
cls={}
for s in range(64): cls.setdefault(canon(s),[]).append(s)
def cyc(c):
    m=[c];t=rot(c,1)
    while t!=c: m.append(t);t=rot(t,1)
    return m
EDG=np.array([(s,s^(1<<i)) for s in range(64) for i in range(6) if s<s^(1<<i)])
SH=np.zeros((len(EDG),)*2,bool)
for i in (0,1):
    for j in (0,1): SH|=EDG[:,i][:,None]==EDG[:,j][None,:]
def crossings(Q):
    A=Q[EDG[:,0]];B=Q[EDG[:,1]]
    def o(p,q,r): return np.sign((q[...,0]-p[...,0])*(r[...,1]-p[...,1])-(q[...,1]-p[...,1])*(r[...,0]-p[...,0]))
    a=A[:,None];b=B[:,None];c=A[None];d=B[None]
    return int((((o(a,b,c)*o(a,b,d))<0)&((o(c,d,a)*o(c,d,b))<0)&~SH).sum())//2
def elen(Q): return float(np.linalg.norm(Q[EDG[:,0]]-Q[EDG[:,1]],axis=1).sum())
GAP=22
def slots(xs):
    xs=sorted(xs);n=len(xs);xs=[190+(xs[k]-xs[n-1-k])/2 for k in range(n)]
    for _ in range(200):
        for k in range(n-1):
            d=xs[k+1]-xs[k]
            if d<GAP: xs[k]-=(GAP-d)/2;xs[k+1]+=(GAP-d)/2
        xs=[190+(xs[k]-xs[n-1-k])/2 for k in range(n)]
    return xs
# x slots per top-half class come from that class's 12-gon x spread (symmetrized)
amb=cyc(7)
SL={c:slots([P[s][0] for s in cls[c]]) for c in [1,3,9,5,11]}
SL['amU']=slots([P[s][0] for s in sorted(amb,key=lambda s:P[s][1])[:3]])
SL[21]=slots([P[s][0] for s in cls[21]])
def ring_orders(c):
    m=cyc(c);n=len(m);out=[]
    for st in range(n):
        for d in (1,-1):
            t=[m[(st+d*k)%n] for k in range(n)]
            if t not in out: out.append(t)
    return out
RO={c:ring_orders(c) for c in [1,3,9,5,11,21]}
AMB=[]                       # amber: three rotation-consecutive states on top, their flips below
for st in range(6):
    for d in (1,-1):
        AMB.append([amb[(st+d*k)%6] for k in range(3)])
SP=36;Y0=34;H=2*(Y0+7*SP)
def build(st):
    tiers=list(st['t2'])
    top=[('g',[0]),(1,RO[1][st['o'][1]])]+[(c,RO[c][st['o'][c]]) for c in tiers]+[(11,RO[11][st['o'][11]]),('amU',AMB[st['a']])]
    Q=np.zeros((64,2));strata=[]
    for k,(c,order) in enumerate(top):
        xs=[190] if c=='g' else SL[c]
        for s,x in zip(order,xs): Q[s]=(x,Y0+k*SP)
        strata.append(order)
    g=RO[21][st['o'][21]]
    for s,x in zip(g,SL[21]): Q[s]=(x,Y0+7*SP)
    strata.append(g)
    for order in reversed(strata[:-1]):
        strata.append([comp(s) for s in reversed(order)])
    for s in range(64):
        if Q[s][1]==0 and s not in (21,42): pass
    for order in strata[:7]:
        for s in order: Q[comp(s)]=(380-Q[s][0],H-Q[s][1])
    return Q,strata
rnd=random.Random(1);best=None
for restart in range(12):
    st={'t2':rnd.sample([3,9,5],3),'o':{c:rnd.randrange(len(RO[c])) for c in RO},'a':rnd.randrange(12)}
    cur=crossings(build(st)[0])
    for it in range(1500):
        new={'t2':st['t2'][:],'o':dict(st['o']),'a':st['a']}
        r=rnd.random()
        if r<.1: i,j=rnd.sample(range(3),2);new['t2'][i],new['t2'][j]=new['t2'][j],new['t2'][i]
        elif r<.25: new['a']=rnd.randrange(12)
        else: c=rnd.choice(list(RO));new['o'][c]=rnd.randrange(len(RO[c]))
        v=crossings(build(new)[0])
        if v<=cur: st,cur=new,v
    if best is None or cur<best[0]: best=(cur,st);print(restart,cur,flush=True)
st=best[1];Q,strata=build(st)
assert len({tuple(q) for q in Q.round(3)})==64
print('flat ring-ordered crossings',crossings(Q),'length',round(elen(Q)))
# arcs: one bend per stratum, mirrored top/bottom, chosen to shorten edges while strata stay >= 26px apart
K=15;xg=np.linspace(4,376,94);sk={s:k for k,o in enumerate(strata) for s in o}
def lay(kap):
    kk=np.zeros(K)
    for k in range(7): kk[k]=kap[k];kk[14-k]=-kap[k]
    R=Q.copy()
    for s in range(64): R[s,1]=Q[s,1]+kk[sk[s]]*(Q[s,0]-190)**2
    return R,kk
def obj(kap):
    R,kk=lay(kap)
    cur=np.array([[Y0+k*SP+kk[k]*(x-190)**2 for x in xg] for k in range(K)])
    pen=np.clip(26-(cur[1:]-cur[:-1]),0,None).sum()*500
    D=np.linalg.norm(R[:,None]-R[None],axis=-1)+np.eye(64)*99
    pen+=np.clip(21-D,0,None).sum()*200
    return elen(R)+pen
bestr=None
for s0 in [np.zeros(7),np.full(7,.001),np.full(7,.002)]:
    r=minimize(obj,s0,method='Nelder-Mead',options={'maxiter':8000,'xatol':1e-8,'fatol':1e-3})
    if bestr is None or r.fun<bestr.fun: bestr=r
R,kk=lay(bestr.x)
D=np.linalg.norm(R[:,None]-R[None],axis=-1)+np.eye(64)*99
print('arcs crossings',crossings(R),'length',round(elen(R)),'min node dist',round(D.min(),1),'y',round(R[:,1].min()),round(R[:,1].max()),'H',H)
print('kk e4',np.round(kk*1e4,2).tolist())
json.dump({'Q':R.round(1).tolist(),'kk':(kk*1e4).round(3).tolist(),'strata':strata,'Y0':Y0,'SP':SP,'H':H},open('rings.json','w'))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6,9))
for s,t in EDG: ax.plot(*zip(R[s],R[t]),color='#bbb',lw=.4)
for k,o in enumerate(strata):
    if len(o)>1: ax.plot(R[o][:,0],R[o][:,1],color='#1D9E75',lw=1.5)
for s in range(64):
    ax.add_patch(plt.Circle(R[s],9.5,fc='white',ec='#555',lw=.6,zorder=3))
    for i in range(6):
        a=math.radians(-90+60*i);ax.add_patch(plt.Circle(R[s]+5.5*np.array([math.cos(a),math.sin(a)]),1.7,fc='#222' if (s>>i)&1 else 'white',ec='#222',lw=.5,zorder=4))
ax.set_xlim(0,380);ax.set_ylim(H+10,0);ax.set_aspect('equal')
plt.savefig('rings.png',dpi=80,bbox_inches='tight')
