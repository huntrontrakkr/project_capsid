import json, math
import numpy as np
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
exec(open('globe.py').read().split('rnd=random.Random(3)')[0])
d=json.load(open('globe_final.json'));st=d['st'];st['o']={int(k):v for k,v in st['o'].items()};st['b']={int(k):v for k,v in st['b'].items()}
prof=tuple(d['prof']);Q,ys,c,rings,ang,amb=build(st,prof);bend=d['bend']
rxs,rys,_=prof
fig,ax=plt.subplots(figsize=(5,9))
for s,t in EDG:
    p,q=Q[s],Q[t];m=(p+q)/2;cp_=np.array([m[0]+bend*(m[0]-CX),m[1]]);u=np.linspace(0,1,20)[:,None]
    pts=(1-u)**2*p+2*(1-u)*u*cp_+u*u*q;ax.plot(pts[:,0],pts[:,1],color='#aaa',lw=.4,zorder=1)
for L in range(1,6):
    for yy in (ys[L],2*c-ys[L]): ax.add_patch(Ellipse((CX,yy),2*rxs[L],2*rys[L],fill=False,ec='#D85A30',lw=1.2,zorder=2))
ax.add_patch(Ellipse((CX,c),2*rxs[6],2*rys[6],fill=False,ec='#D85A30',lw=1.2,zorder=2))
ax.add_patch(Ellipse((CX,c),2*st['gd']+16,22,fill=False,ec='#D85A30',lw=1.2,zorder=2))
for s in range(64):
    ax.add_patch(Circle(Q[s],9.5,fc='white',ec='#333',lw=.6,zorder=3))
    for i in range(6):
        a=math.radians(-90+60*i);ax.add_patch(Circle(Q[s]+5.6*np.array([math.cos(a),math.sin(a)]),1.7,fc='#222' if (s>>i)&1 else 'white',ec='#222',lw=.4,zorder=4))
ax.set_xlim(40,340);ax.set_ylim(680,20);ax.set_aspect('equal');ax.axis('off')
plt.savefig('globe.png',dpi=85,bbox_inches='tight')
