import math, random, json, sys
import numpy as np
rot=lambda s,k:((s<<k)|(s>>(6-k)))&63 if k%6 else s
canon=lambda s:min(rot(s,k) for k in range(6))
pc=lambda s:bin(s).count('1')
comp=lambda s:63^s
def pet(s):
    x=0
    for i in range(6): x+=(((s>>i)&1)-.5)*math.cos(math.radians(i*30+15))
    return x
gapE=[]
for w in range(6):
    gapE.append(np.array([(s,s^(1<<i)) for s in range(64) if pc(s)==w for i in range(6) if not (s>>i)&1]))
def crossings(rows):
    pos=np.zeros(64)
    for r in rows:
        for j,s in enumerate(r): pos[s]=j
    tot=0
    for E in gapE:
        a=pos[E[:,0]];b=pos[E[:,1]]
        tot+=int(((a[:,None]-a[None,:])*(b[:,None]-b[None,:])<0).sum())//2
    return tot
base=[sorted([s for s in range(64) if pc(s)==w],key=pet) for w in range(7)]
print('baseline',crossings(base))
classes={}
for s in range(64): classes.setdefault(canon(s),[]).append(s)
W2=[3,5,9];W3=[7,11,13,21];W4=[15,23,27]
aligned=sys.argv[1]=='aligned'
def build(st):
    w2=[s for c in st['b2'] for s in st['o'][c]]
    w3=[s for c in st['b3'] for s in st['o'][c]]
    if aligned:
        w4=[comp(s) for s in w2]; w5=[comp(s) for s in st['o'][1]]
    else:
        w4=[s for c in st['b4'] for s in st['o'][c]]; w5=st['o'][31]
    return [[0],st['o'][1],w2,w3,w4,w5,[63]]
def anneal(seed,iters):
    rnd=random.Random(seed)
    st={'b2':rnd.sample(W2,3),'b3':rnd.sample(W3,4),'b4':rnd.sample(W4,3),'o':{c:rnd.sample(m,len(m)) for c,m in classes.items()}}
    cur=crossings(build(st)); best=(cur,build(st))
    T0=30.0
    keys=[c for c in classes if len(classes[c])>1 and (not aligned or pc(c)<=3)]
    for it in range(iters):
        T=T0*(1-it/iters)+0.01
        new={'b2':st['b2'][:],'b3':st['b3'][:],'b4':st['b4'][:],'o':{k:v[:] for k,v in st['o'].items()}}
        m=rnd.random()
        if m<.08: L=new['b2'];i,j=rnd.sample(range(3),2);L[i],L[j]=L[j],L[i]
        elif m<.16: L=new['b3'];i,j=rnd.sample(range(4),2);L[i],L[j]=L[j],L[i]
        elif m<.24 and not aligned: L=new['b4'];i,j=rnd.sample(range(3),2);L[i],L[j]=L[j],L[i]
        else:
            c=rnd.choice(keys);L=new['o'][c];n=len(L)
            if rnd.random()<.5: i,j=rnd.sample(range(n),2);L[i],L[j]=L[j],L[i]
            else: i=rnd.randrange(n);x=L.pop(i);L.insert(rnd.randrange(n),x)
        v=crossings(build(new))
        if v<=cur or rnd.random()<math.exp((cur-v)/T):
            st,cur=new,v
            if cur<best[0]: best=(cur,build(st))
    return best
B=None
for seed in range(int(sys.argv[3])):
    b=anneal(seed,int(sys.argv[2]))
    print(seed,b[0],flush=True)
    if B is None or b[0]<B[0]: B=b
print('best',B[0]); json.dump(B[1],open(f'best_{sys.argv[1]}.json','w'))
