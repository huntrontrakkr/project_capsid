import json
import numpy as np
exec(open('thin.py').read().split('def search')[0])
st=json.load(open('thin_best.json'))['False'];st['o']={int(k):v for k,v in st['o'].items()}
Q,strata=build(st,False)
sk={s:k for k,o in enumerate(strata) for s in o}
print('straight',crossings(Q),round(elen(Q)),'width',Q[:,0].min(),Q[:,0].max())
prof=np.array([0,1,1,.8,.6,.4,.2])   # bend grows toward the poles
best=(crossings(Q),0,0,Q)
for shape in ('pole','flat'):
    pr=prof if shape=='pole' else np.array([0,1,1,1,1,1,1.])
    for a in np.linspace(-0.012,0.012,25):
        kk=np.zeros(15)
        for k in range(7): kk[k]=a*pr[k];kk[14-k]=-a*pr[k]
        R=Q.copy()
        for s in range(64): R[s,1]=Q[s,1]+kk[sk[s]]*(Q[s,0]-190)**2
        dmin=(np.linalg.norm(R[:,None]-R[None],axis=-1)+np.eye(64)*99).min()
        if dmin<21: continue
        v=crossings(R)
        if v<best[0]: best=(v,shape,a,R);print(shape,round(a,4),v,round(elen(R)),flush=True)
v,shape,a,R=best
print('best',v,shape,a)
json.dump({'Q':R.round(1).tolist(),'strata':strata,'shape':shape,'a':a,'SP':SP,'Y0':Y0,'H':H,'D':D},open('thin.json','w'))
