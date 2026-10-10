import math, random, json, sys
import numpy as np
pc=lambda s:bin(s).count('1')
gapE=[np.array([(s,s^(1<<i)) for s in range(64) if pc(s)==w for i in range(6) if not (s>>i)&1]) for w in range(6)]
def crossings(rows):
    pos=np.zeros(64)
    for r in rows:
        for j,s in enumerate(r): pos[s]=j
    tot=0
    for E in gapE:
        a=pos[E[:,0]];b=pos[E[:,1]]
        tot+=int(((a[:,None]-a[None,:])*(b[:,None]-b[None,:])<0).sum())//2
    return tot
best=None
for seed in range(4):
    rnd=random.Random(seed)
    rows=[rnd.sample([s for s in range(64) if pc(s)==w],[1,6,15,20,15,6,1][w]) for w in range(7)]
    cur=crossings(rows);b=cur
    N=60000
    for it in range(N):
        T=30*(1-it/N)+.01
        new=[r[:] for r in rows];w=rnd.randrange(1,6);L=new[w];n=len(L)
        if rnd.random()<.5: i,j=rnd.sample(range(n),2);L[i],L[j]=L[j],L[i]
        else: i=rnd.randrange(n);x=L.pop(i);L.insert(rnd.randrange(n),x)
        v=crossings(new)
        if v<=cur or rnd.random()<math.exp((cur-v)/T): rows,cur=new,v; b=min(b,cur)
    print(seed,b,flush=True)
