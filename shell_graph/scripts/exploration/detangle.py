import math, random, itertools, json
rot=lambda s,k:((s<<k)|(s>>(6-k)))&63 if k%6 else s
canon=lambda s:min(rot(s,k) for k in range(6))
pc=lambda s:bin(s).count('1')
comp=lambda s:63^s
# 12-gon x for baseline order
def pet(s):
    x=y=0
    for i in range(6):
        c=((s>>i)&1)-.5; a=math.radians(i*30+15); x+=c*math.cos(a); y+=c*math.sin(a)
    return x,y
edges=[(s,s^(1<<i)) for s in range(64) for i in range(6) if s<s^(1<<i)]
def crossings(rows):
    pos={}
    for r in rows:
        for j,s in enumerate(r): pos[s]=j
    tot=0
    for w in range(6):
        E=[(pos[a],pos[b]) if pc(a)==w else (pos[b],pos[a]) for a,b in edges if min(pc(a),pc(b))==w]
        for (a,b),(c,d) in itertools.combinations(E,2):
            if (a-c)*(b-d)<0: tot+=1
    return tot
base=[sorted([s for s in range(64) if pc(s)==w],key=lambda s:pet(s)) for w in range(7)]
print('baseline layered crossings',crossings(base))
classes={}
for s in range(64): classes.setdefault(canon(s),[]).append(s)
def cyc(c):  # rotation order
    m=[c];t=rot(c,1)
    while t!=c: m.append(t);t=rot(t,1)
    return m
fam_w2=[3,5,9]; w3cls=[7,11,13,21]
def cyc_orders(c):
    m=cyc(c);n=len(m);out=[]
    for st in range(n):
        for d in (1,-1):
            out.append(tuple(m[(st+d*k)%n] for k in range(n)))
    return list(set(out))
def build(st):
    w1=list(st['w1']); w2=[s for c in st['b2'] for s in st['o'][c]]
    w3=[s for c in st['b3'] for s in st['o'][c]]
    return [[0],w1,w2,w3,[comp(s) for s in w2],[comp(s) for s in w1],[63]]
def search(free,iters=4000,restarts=40,seed=0):
    rnd=random.Random(seed); best=None
    for R in range(restarts):
        st={'w1':rnd.sample(classes[1],6),'b2':rnd.sample(fam_w2,3),'b3':rnd.sample(w3cls,4),'o':{}}
        for c in fam_w2+w3cls:
            st['o'][c]=list(rnd.choice(cyc_orders(c))) if not free else rnd.sample(classes[c],len(classes[c]))
        cur=crossings(build(st))
        for it in range(iters):
            new={'w1':st['w1'][:],'b2':st['b2'][:],'b3':st['b3'][:],'o':{k:v[:] for k,v in st['o'].items()}}
            m=rnd.random()
            if m<.1: i,j=rnd.sample(range(3),2);new['b2'][i],new['b2'][j]=new['b2'][j],new['b2'][i]
            elif m<.2: i,j=rnd.sample(range(4),2);new['b3'][i],new['b3'][j]=new['b3'][j],new['b3'][i]
            elif m<.35:
                i,j=rnd.sample(range(6),2);new['w1'][i],new['w1'][j]=new['w1'][j],new['w1'][i]
            else:
                c=rnd.choice(fam_w2+w3cls)
                if free:
                    n=len(new['o'][c])
                    if n<2: continue
                    i,j=rnd.sample(range(n),2);new['o'][c][i],new['o'][c][j]=new['o'][c][j],new['o'][c][i]
                else: new['o'][c]=list(rnd.choice(cyc_orders(c)))
            v=crossings(build(new))
            if v<=cur: st,cur=new,v
        if best is None or cur<best[0]: best=(cur,build(st)); print(R,cur,flush=True)
    return best
import sys
free=sys.argv[1]=='free'
b=search(free,iters=int(sys.argv[2]),restarts=int(sys.argv[3]))
print('best',b[0]); json.dump(b[1],open(f'best_{sys.argv[1]}.json','w'))
