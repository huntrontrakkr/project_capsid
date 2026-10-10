import math, json, random
import numpy as np
exec(open('rings.py').read().split('SP=36')[0])      # helpers: rot, cyc, comp, RO, AMB, EDG, crossings, elen
CX=190
# levels top->bottom: gray, coral, tier2 x3, pink, [amber ring around the alternating pair], pink', tier2' x3, coral', gray'
def geometry(rxs,rys,gap):
    ys=[0.0];
    for L in range(1,7): ys.append(ys[-1]+rys[L-1]+rys[L]+gap)
    c=ys[6]
    ys=ys+[2*c-ys[5-j] for j in range(6)]
    return ys,c
def place(order,cx,cy,rx,ry,th0,step):
    pts={}
    for j,s in enumerate(order): a=math.radians(th0+j*step);pts[s]=(cx+rx*math.cos(a),cy+ry*math.sin(a),a)
    return pts
def build(st,prof):
    rxs,rys,gap=prof
    ys,c=geometry(rxs,rys,gap)
    ys=[y+40-ys[0] for y in ys];c=ys[6]
    Q=np.zeros((64,2));ang={}
    Q[0]=(CX,ys[0])
    rings=[('g',[0])]
    tops=[(1,RO[1][st['o'][1]])]+[(cc,RO[cc][st['o'][cc]]) for cc in st['t2']]+[(11,RO[11][st['o'][11]])]
    for L,(cc,order) in enumerate(tops,start=1):
        n=len(order);base=st['b'][cc]
        th0=(90 if base else 270) if n==3 else 0
        for s,(x,y,a) in place(order,CX,ys[L],rxs[L],rys[L],th0,360/n).items(): Q[s]=(x,y);ang[s]=a
        rings.append((cc,order))
    amb=[cyc(7)[(st['a0']+st['ad']*k)%6] for k in range(6)]
    th0=30 if st['b'][7] else 0
    for s,(x,y,a) in place(amb,CX,c,rxs[6],rys[6],th0,60).items(): Q[s]=(x,y);ang[s]=a
    g=RO[21][st['o'][21]];gd=st['gd']
    Q[g[0]]=(CX-gd,c);Q[g[1]]=(CX+gd,c)
    for cc,order in rings:
        for s in order:
            t=comp(s);Q[t]=(2*CX-Q[s][0],2*c-Q[s][1])
            if s in ang: ang[t]=ang[s]+math.pi
    rings_all=rings+[('amb',amb),('grn',g)]+[('c'+str(cc),[comp(s) for s in order]) for cc,order in rings]
    return Q,ys,c,rings_all,ang,amb
PROF=([0,46,64,80,92,104,112],[0,13,13,14,15,16,26],22)
def mind(Q): return (np.linalg.norm(Q[:,None]-Q[None],axis=-1)+np.eye(64)*99).min()
def bez_cross(Q,bend):
    # edges drawn as gentle meridian curves: control point pushed outward from the axis
    segs=[]
    for s,t in EDG:
        p,q=Q[s],Q[t];m=(p+q)/2;cpt=np.array([m[0]+bend*(m[0]-CX),m[1]])
        ts=np.linspace(0,1,7)
        pts=[(1-u)**2*p+2*(1-u)*u*cpt+u*u*q for u in ts]
        segs.append(pts)
    A=[];B=[];own=[]
    for e,pts in enumerate(segs):
        for i in range(len(pts)-1): A.append(pts[i]);B.append(pts[i+1]);own.append(e)
    A=np.array(A);B=np.array(B);own=np.array(own)
    def o(p,q,r): return np.sign((q[...,0]-p[...,0])*(r[...,1]-p[...,1])-(q[...,1]-p[...,1])*(r[...,0]-p[...,0]))
    a=A[:,None];b=B[:,None];cc=A[None];d=B[None]
    X=((o(a,b,cc)*o(a,b,d))<0)&((o(cc,d,a)*o(cc,d,b))<0)
    E=EDG;sh=SH[own][:,own]
    X&=~sh
    pairs=set()
    for i,j in zip(*np.nonzero(np.triu(X))): 
        if own[i]!=own[j]: pairs.add((min(own[i],own[j]),max(own[i],own[j])))
    return len(pairs)
def randst(rnd):
    return {'t2':rnd.sample([3,9,5],3),'o':{c:rnd.randrange(len(RO[c])) for c in RO},'b':{c:rnd.randrange(2) for c in [1,3,9,5,11,7]},'a0':rnd.randrange(6),'ad':rnd.choice([1,-1]),'gd':16}
def cp(st): return {'t2':st['t2'][:],'o':dict(st['o']),'b':dict(st['b']),'a0':st['a0'],'ad':st['ad'],'gd':st['gd']}
def score(st):
    Q=build(st,PROF)[0]
    pen=0 if mind(Q)>=20.5 else 10000
    return crossings(Q)+pen
rnd=random.Random(3);best=None
for r in range(10):
    st=randst(rnd);cur=score(st)
    for it in range(2000):
        new=cp(st);x=rnd.random()
        if x<.08: i,j=rnd.sample(range(3),2);new['t2'][i],new['t2'][j]=new['t2'][j],new['t2'][i]
        elif x<.25: c=rnd.choice([9,7]);new['b'][c]^=1
        elif x<.35: new['a0']=rnd.randrange(6);new['ad']=rnd.choice([1,-1])
        else: c=rnd.choice(list(RO));new['o'][c]=rnd.randrange(len(RO[c]))
        v=score(new)
        if v<=cur: st,cur=new,v
    if best is None or cur<best[0]: best=(cur,st);print(r,cur,flush=True)
st=best[1];Q,ys,c,rings,ang,amb=build(st,PROF)
print('straight crossings',crossings(Q),'min dist',round(mind(Q),1),'height',round(Q[:,1].max()-Q[:,1].min()),'width',round(Q[:,0].max()-Q[:,0].min()))
for bend in [0,.1,.2,.3,.45,.6]: print('bend',bend,bez_cross(Q,bend))
json.dump({'st':st,'Q':Q.round(1).tolist(),'ys':ys,'c':c,'rings':[[str(a),b] for a,b in rings],'prof':PROF},open('globe.json','w'))
