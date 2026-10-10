import json, itertools, sys
import numpy as np
exec(open('globe.py').read().split('rnd=random.Random(3)')[0])
d=json.load(open('globe.json'));st=d['st'];st['o']={int(k):v for k,v in st['o'].items()};st['b']={int(k):v for k,v in st['b'].items()}
profs={'base':PROF,'wide':([0,52,74,92,106,118,128],[0,13,14,15,16,17,27],22),'narrow':([0,42,56,68,78,88,96],[0,13,13,14,14,15,25],22)}
def sc(st,prof,bend):
    Q=build(st,prof)[0]
    if mind(Q)<20.5: return 99999
    return bez_cross(Q,bend)
out={}
for name,prof in profs.items():
    for bend in (.3,.45):
        s=cp(st);cur=sc(s,prof,bend);imp=True
        while imp:
            imp=False
            cands=[]
            for c in RO:
                for k in range(len(RO[c])): n=cp(s);n['o'][c]=k;cands.append(n)
            for c in [9,7]: n=cp(s);n['b'][c]^=1;cands.append(n)
            for a0 in range(6):
                for ad in (1,-1): n=cp(s);n['a0']=a0;n['ad']=ad;cands.append(n)
            for p in itertools.permutations([3,9,5]): n=cp(s);n['t2']=list(p);cands.append(n)
            for n in cands:
                v=sc(n,prof,bend)
                if v<cur: s,cur,imp=n,v,True
        print(name,bend,cur,flush=True);out[(name,bend)]=(cur,s)
k=min(out,key=lambda k:out[k][0]);v,s=out[k];print('best',k,v)
Q,ys,c,rings,ang,amb=build(s,profs[k[0]])
json.dump({'st':s,'Q':Q.round(2).tolist(),'ys':ys,'c':c,'rings':[[str(a),b] for a,b in rings],'prof':profs[k[0]],'bend':k[1],'amb':amb},open('globe_final.json','w'))
print('height',round(Q[:,1].max()-Q[:,1].min()),'width',round(Q[:,0].max()-Q[:,0].min()),'ymin',Q[:,1].min(),'ymax',Q[:,1].max())
