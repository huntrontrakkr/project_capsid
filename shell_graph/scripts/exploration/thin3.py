import json
import numpy as np
exec(open('thin.py').read().split('def search')[0])
st=json.load(open('thin_best.json'))['False'];st['o']={int(k):v for k,v in st['o'].items()}
Q,strata=build(st,False)
w=[bin(o[0]).count('1') for o in strata]
sk={s:k for k,o in enumerate(strata) for s in o}
res=[]
for gin in [22,26,30,34,40]:
    for gout in [40,50,60,70,80]:
        ys=[0]
        for k in range(1,15): ys.append(ys[-1]+(gin if w[k]==w[k-1] else gout))
        ys=np.array(ys);ys=ys-ys[7]
        R=Q.copy()
        for s in range(64): R[s,1]=300+ys[sk[s]]
        dmin=(np.linalg.norm(R[:,None]-R[None],axis=-1)+np.eye(64)*99).min()
        if dmin<21: continue
        res.append((crossings(R),gin,gout,round(ys[-1]-ys[0])))
res.sort();print(res[:8])
