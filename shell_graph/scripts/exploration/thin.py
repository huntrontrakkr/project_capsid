import math, json, random, sys
import numpy as np
exec(open('rings.py').read().split('SP=36')[0])        # reuse helpers: cyc, RO, AMB, crossings, elen, comp
D=24; SP=40; Y0=30; H=2*(Y0+7*SP)
def xs(n): return [190+(j-(n-1)/2)*D for j in range(n)]
def build(st,mirror):
    top=[[0],RO[1][st['o'][1]]]+[RO[c][st['o'][c]] for c in st['t2']]+[RO[11][st['o'][11]],AMB[st['a']]]
    Q=np.zeros((64,2));strata=[]
    for k,order in enumerate(top):
        for s,x in zip(order,xs(len(order))): Q[s]=(x,Y0+k*SP)
        strata.append(order)
    g=RO[21][st['o'][21]]
    for s,x in zip(g,xs(2)): Q[s]=(x,Y0+7*SP)
    strata.append(g)
    for order in reversed(strata[:7]):
        bot=[comp(s) for s in order] if mirror else [comp(s) for s in reversed(order)]
        strata.append(bot)
    for order in strata[:7]:
        for s in order: Q[comp(s)]=(Q[s][0] if mirror else 380-Q[s][0],H-Q[s][1])
    return Q,strata
def search(mirror,restarts=10,iters=2500,seed=0):
    rnd=random.Random(seed);best=None
    for r in range(restarts):
        st={'t2':rnd.sample([3,9,5],3),'o':{c:rnd.randrange(len(RO[c])) for c in RO},'a':rnd.randrange(12)}
        cur=crossings(build(st,mirror)[0])
        for it in range(iters):
            new={'t2':st['t2'][:],'o':dict(st['o']),'a':st['a']};x=rnd.random()
            if x<.1: i,j=rnd.sample(range(3),2);new['t2'][i],new['t2'][j]=new['t2'][j],new['t2'][i]
            elif x<.22: new['a']=rnd.randrange(12)
            else: c=rnd.choice(list(RO));new['o'][c]=rnd.randrange(len(RO[c]))
            v=crossings(build(new,mirror)[0])
            if v<=cur: st,cur=new,v
        # exhaustive polish one variable at a time
        imp=True
        while imp:
            imp=False
            for c in RO:
                for k in range(len(RO[c])):
                    new={'t2':st['t2'][:],'o':dict(st['o']),'a':st['a']};new['o'][c]=k
                    v=crossings(build(new,mirror)[0])
                    if v<cur: st,cur,imp=new,v,True
            for k in range(12):
                new={'t2':st['t2'][:],'o':dict(st['o']),'a':k};v=crossings(build(new,mirror)[0])
                if v<cur: st,cur,imp=new,v,True
        if best is None or cur<best[0]: best=(cur,st);print(('mirror' if mirror else 'point'),r,cur,flush=True)
    return best
res={m:search(m) for m in (False,True)}
for m,(v,st) in res.items(): print('mirror' if m else 'point','best',v,st)
json.dump({str(m):res[m][1] for m in res},open('thin_best.json','w'))
