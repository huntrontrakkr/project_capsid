import math, random, json, itertools
import numpy as np
rot=lambda s,k:((s<<k)|(s>>(6-k)))&63 if k%6 else s
canon=lambda s:min(rot(s,k) for k in range(6))
pc=lambda s:bin(s).count('1'); comp=lambda s:63^s
EDG=np.array([(s,s^(1<<i)) for s in range(64) for i in range(6) if s<s^(1<<i)])
def cyc(c):
    m=[c];t=rot(c,1)
    while t!=c: m.append(t);t=rot(t,1)
    return m
def orients(c):
    m=cyc(c);n=len(m);o=[]
    for st in range(n):
        for d in (1,-1):
            t=[m[(st+d*k)%n] for k in range(n)]
            if t not in o:o.append(t)
    return o
OR={c:orients(c) for c in [1,3,5,9,7,11,13,21]}
SP=14; Y=[18,92,200,304,408,516,590]; H=608
XS={k:190+k*50 for k in range(-3,4)}
SIDE=[3,5,9,7,11,13]   # blue teal purple amber pinkL pinkD
def layout(st):
    P={0:(190,Y[0]),63:(190,Y[6])}
    def col(order,x,yc,flip=False):
        n=len(order)
        for j,s in enumerate(order):
            P[s]=(x,yc+(j-(n-1)/2)*SP)
            if flip: P[comp(s)]=(x,H-P[s][1])
    col(OR[1][st['o'][1]],190,Y[1],True)
    col(OR[21][st['o'][21]],190,Y[3])
    for c,k in st['slot'].items():
        x=XS[k]
        if c in (3,5,9): col(OR[c][st['o'][c]],x,Y[2],True)
        else: col(OR[c][st['o'][c]],x,Y[3])
    return P
def crossings(P):
    A=np.array([P[s] for s in EDG[:,0]]);B=np.array([P[s] for s in EDG[:,1]])
    def orient(p,q,r): return np.sign((q[...,0]-p[...,0])*(r[...,1]-p[...,1])-(q[...,1]-p[...,1])*(r[...,0]-p[...,0]))
    a=A[:,None,:];b=B[:,None,:];c=A[None,:,:];d=B[None,:,:]
    o1=orient(a,b,c);o2=orient(a,b,d);o3=orient(c,d,a);o4=orient(c,d,b)
    share=(EDG[:,0][:,None]==EDG[:,0][None,:])|(EDG[:,0][:,None]==EDG[:,1][None,:])|(EDG[:,1][:,None]==EDG[:,0][None,:])|(EDG[:,1][:,None]==EDG[:,1][None,:])
    X=(o1*o2<0)&(o3*o4<0)&~share
    # collinear overlaps (same column) count as bad too
    col=(o1==0)&(o2==0)&~share
    return int(X.sum())//2 + int(col.sum())//2
def rand_slots(rnd):
    pair=rnd.choice([1,2,3]); side=rnd.choice([1,-1])
    slots={11:-pair*side,13:pair*side}
    rest=[k for k in [-3,-2,-1,1,2,3] if k not in slots.values()]
    rnd.shuffle(rest)
    for c,k in zip([3,5,9,7],rest): slots[c]=k
    return slots
B=None
for seed in range(16):
    rnd=random.Random(seed)
    st={'slot':rand_slots(rnd),'o':{c:rnd.randrange(len(OR[c])) for c in OR}}
    cur=crossings(layout(st)); N=5000
    for it in range(N):
        T=40*(1-it/N)+.01
        new={'slot':dict(st['slot']),'o':dict(st['o'])}
        m=rnd.random()
        if m<.15: new['slot']=rand_slots(rnd)
        elif m<.3:
            a,b=rnd.sample([3,5,9,7],2);new['slot'][a],new['slot'][b]=new['slot'][b],new['slot'][a]
        else: c=rnd.choice(list(OR));new['o'][c]=rnd.randrange(len(OR[c]))
        v=crossings(layout(new))
        if v<=cur or rnd.random()<math.exp((cur-v)/T): st,cur=new,v
    if B is None or cur<B[0]: B=(cur,st);print(seed,cur,st['slot'],flush=True)
P=layout(B[1]); print('final',B[0])
json.dump({'P':{s:[round(x,1),round(y,1)] for s,(x,y) in P.items()},'slot':B[1]['slot']},open('sym.json','w'))
