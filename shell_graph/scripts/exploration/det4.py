import math, random, json
import numpy as np
rot=lambda s,k:((s<<k)|(s>>(6-k)))&63 if k%6 else s
canon=lambda s:min(rot(s,k) for k in range(6))
pc=lambda s:bin(s).count('1'); comp=lambda s:63^s
gapE=[np.array([(s,s^(1<<i)) for s in range(64) if pc(s)==w for i in range(6) if not (s>>i)&1]) for w in range(6)]
def gaps(rows):
    pos=np.zeros(64)
    for r in rows:
        for j,s in enumerate(r): pos[s]=j
    out=[]
    for E in gapE:
        a=pos[E[:,0]];b=pos[E[:,1]]
        out.append(int(((a[:,None]-a[None,:])*(b[:,None]-b[None,:])<0).sum())//2)
    return out
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
W2=[3,5,9];W3=[7,11,13,21]
def build(st):
    w1=OR[1][st['o'][1]]
    w2=[s for c in st['b2'] for s in OR[c][st['o'][c]]]
    w3=[s for c in st['b3'] for s in OR[c][st['o'][c]]]
    return [[0],w1,w2,w3,[comp(s) for s in w2],[comp(s) for s in w1],[63]]
B=None
for seed in range(30):
    rnd=random.Random(seed)
    st={'b2':rnd.sample(W2,3),'b3':rnd.sample(W3,4),'o':{c:rnd.randrange(len(OR[c])) for c in OR}}
    cur=sum(gaps(build(st)))
    N=6000
    for it in range(N):
        T=20*(1-it/N)+.01
        new={'b2':st['b2'][:],'b3':st['b3'][:],'o':dict(st['o'])}
        m=rnd.random()
        if m<.15: L=new['b2'];i,j=rnd.sample(range(3),2);L[i],L[j]=L[j],L[i]
        elif m<.3: L=new['b3'];i,j=rnd.sample(range(4),2);L[i],L[j]=L[j],L[i]
        else: c=rnd.choice(list(OR));new['o'][c]=rnd.randrange(len(OR[c]))
        v=sum(gaps(build(new)))
        if v<=cur or rnd.random()<math.exp((cur-v)/T): st,cur=new,v
    # greedy polish: exhaustive single-variable
    improved=True
    while improved:
        improved=False
        for c in OR:
            for k in range(len(OR[c])):
                new={'b2':st['b2'][:],'b3':st['b3'][:],'o':dict(st['o'])};new['o'][c]=k
                v=sum(gaps(build(new)))
                if v<cur: st,cur,improved=new,v,True
    if B is None or cur<B[0]: B=(cur,st); print(seed,cur,gaps(build(st)),st['b2'],st['b3'],flush=True)
rows=build(B[1]); print('final',B[0],gaps(rows))
json.dump({'rows':rows,'b2':B[1]['b2'],'b3':B[1]['b3']},open('cols.json','w'))
print(rows)
