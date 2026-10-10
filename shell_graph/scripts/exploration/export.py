import json, math
import numpy as np
exec(open('globe.py').read().split('rnd=random.Random(3)')[0])
d=json.load(open('globe_final.json'));st=d['st'];st['o']={int(k):v for k,v in st['o'].items()};st['b']={int(k):v for k,v in st['b'].items()}
prof=tuple(d['prof']);Q,ys,c,rings,ang,amb=build(st,prof);rxs,rys,_=prof
R=[]
def ring(order,cy,rx,ry):
    th=[math.degrees(math.atan2((Q[s][1]-cy)/ry,(Q[s][0]-CX)/rx))%360 for s in order]
    # unwrap so angles increase along the ring order
    for j in range(1,len(th)):
        while th[j]<=th[j-1]: th[j]+=360
    R.append({'cy':round(cy,2),'rx':rx,'ry':ry,'o':order,'t':[round(t,2) for t in th]})
tops=[[0]]
for L in range(1,6):
    order=rings[L][1];ring(order,ys[L],rxs[L],rys[L]);ring([comp(s) for s in order],2*c-ys[L],rxs[L],rys[L])
ring(amb,c,rxs[6],rys[6])
g=RO[21][st['o'][21]];gL,gR=(g if Q[g[0]][0]<Q[g[1]][0] else g[::-1]);ring([gL,gR],c,st['gd'],14)
# check ring steps are true rotations
for r in R:
    o=r['o'];n=len(o)
    for j in range(n):
        a,b=o[j],o[(j+1)%n];assert b in (rot(a,1),rot(a,5)),(a,b)
print(json.dumps([[round(x,1),round(y,1)] for x,y in Q],separators=(',',':')))
print(json.dumps(R,separators=(',',':')))
print('bend',d['bend'],'c',c,'ymin',Q[:,1].min(),'ymax',Q[:,1].max())
