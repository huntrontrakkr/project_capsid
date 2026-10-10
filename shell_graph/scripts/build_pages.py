# Regenerates pages 07 to 10 from layouts.json. Run from shell_graph/pages/.
import json
D=json.load(open(__import__('pathlib').Path(__file__).resolve().parents[1].joinpath('layouts.json')))['layouts']
def J(x): return json.dumps(x,separators=(',',':'))
css_strata='''.v{cursor:pointer}
.mono{font-family:var(--font-mono);color:var(--text-primary);letter-spacing:1px}
.lg{display:flex;align-items:center;gap:8px;cursor:pointer;padding:3px 6px;border-radius:var(--radius);border:.5px solid transparent}
.lg.on{background:var(--surface-1);border-color:var(--border-strong)}'''
css_ring='''.v{cursor:pointer}
.v circle.b{fill:var(--surface-1);stroke:var(--text-secondary);stroke-width:.75}
.v.n circle.b{stroke:#D85A30;stroke-width:2}
.v.s circle.b{stroke:var(--text-accent);stroke-width:2.75}
.mono{font-family:var(--font-mono);color:var(--text-primary);letter-spacing:1px}'''
def page(title,h1,note,css,body,script):
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><link rel="stylesheet" href="style.css">
<style>
{css}
</style></head><body><main>
<h1>{h1}</h1>
<p class="note">{note}</p>
{body}
<nav><a href="index.html">All graphs</a></nav>
{script}
</main></body></html>
'''
def strata_body(h,desc): return f'''<div style="font-size:13px;color:var(--text-secondary);margin:0 0 2px">Top: all open · bottom: all filled · one necklace per line</div>
<svg id="g" width="100%" viewBox="0 0 380 {h}" role="img"><title>Stratified necklaces</title><desc>{desc}</desc></svg>
<div id="out" style="font-size:13px;color:var(--text-secondary);line-height:1.7;margin:2px 0 8px;min-height:46px"></div>
<div id="leg" style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 10px;font-size:13px;color:var(--text-secondary)"></div>'''
Q=D['stratified']['pos']
open('07_stratified.html','w').write(page('Stratified 12-gon','Stratified 12-gon',
 "Each necklace shifted vertically onto its own line, lines pushed apart in 12-gon height order, and every line made symmetric about the center (k-th from the left averaged with k-th from the right).",
 css_strata, strata_body(456,'Each rotation class flattened onto its own horizontal line'),
 f'''<script src="strata_view.js"></script>
<script>
const Q={J(Q)};
let gl='';[...new Set(Q.map(q=>q[1]))].forEach(y=>gl+=`<line x1="8" x2="372" y1="${{y}}" y2="${{y}}" stroke="var(--border)" stroke-width=".5"/>`);
gl+=`<line x1="190" x2="190" y1="16" y2="448" stroke="var(--border)" stroke-width=".5" stroke-dasharray="3 4"/>`;
strataView({{Q,guides:gl}});
</script>'''))
Q=D['stratified_arcs']['pos']
open('08_stratified_arcs.html','w').write(page('Stratified arcs','Stratified arcs',
 "The three-in-a-row necklace is split around the centered alternating pair, and every stratum is bent into a nested arc. The bends minimize total edge length with strata kept at least 14 px apart.",
 css_strata, strata_body(500,'Each rotation class on its own nested arc'),
 f'''<script src="strata_view.js"></script>
<script>
const Q={J(Q)};
const KK=[27.83,31.26,26.03,20.81,15.67,10.45,5.22,0,-5.22,-10.45,-15.67,-20.81,-26.03,-31.26,-27.83];
let gl='';for(let k=1;k<14;k++){{const y0=40+30*k,kk=KK[k]*1e-4,a=172,ye=y0+kk*a*a,yc=y0-kk*a*a;gl+=`<path d="M${{190-a}} ${{ye.toFixed(1)}}Q190 ${{yc.toFixed(1)}} ${{190+a}} ${{ye.toFixed(1)}}" fill="none" stroke="var(--border)" stroke-width=".75"/>`}}
strataView({{Q,guides:gl}});
</script>'''))
def ring_body(h,desc): return f'''<div style="display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;font-size:13px;color:var(--text-secondary);margin:0 0 4px">
<span style="display:flex;align-items:center;gap:6px"><svg width="22" height="8" aria-hidden="true"><line x1="0" y1="4" x2="22" y2="4" stroke="#1D9E75" stroke-width="2"/></svg>Rotate one step</span>
<span style="display:flex;align-items:center;gap:6px"><svg width="22" height="8" aria-hidden="true"><line x1="0" y1="4" x2="22" y2="4" stroke="var(--text-muted)" stroke-width="1"/></svg>Add or remove one unit</span>
</div>
<svg id="g" width="100%" viewBox="0 0 380 {h}" role="img"><title>Necklace rings</title><desc>{desc}</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs></svg>
<div id="out" style="font-size:13px;color:var(--text-secondary);line-height:1.7;margin-top:2px;min-height:46px"></div>'''
L=D['arc_rings']
open('09_arc_rings.html','w').write(page('Necklace rings on arcs','Necklace rings on arcs',
 "Each state drawn as its shell pattern. Each necklace is ordered by rotation along its arc, so neighbors are one click apart, and the dashed arrows leaving both ends are the same wrap-around step. The three-in-a-row ring encircles the alternating pair.",
 css_ring, ring_body(572,'Shell states as ring patterns on nested arcs, rotation rings wrapping off the edges'),
 f'''<script src="ring_view.js"></script>
<script>
const KK=[14.51,15.797,14.416,11.556,8.665,5.781,2.89,0,-2.89,-5.781,-8.665,-11.556,-14.416,-15.797,-14.51];
ringView({{Q:{J(L['pos'])},ST:{J(L['strata'])},yk:(k,x)=>34+36*k+KK[k]*1e-4*(x-190)**2,stubL:8,stubR:372,bulge:44,greenBulge:34}});
</script>'''))
L=D['compact_rings']
open('10_compact_rings.html','w').write(page('Compact necklace rings','Compact necklace rings',
 "Thin and tall: every necklace on an evenly spaced row one glyph apart, in rotation order. Row order and each ring's start and direction were searched for fewest crossings (988). Swapping open and filled turns the picture 180 degrees.",
 css_ring, ring_body(620,'Shell states in rotation order on evenly spaced rows, rings wrapping off both sides'),
 f'''<script src="ring_view.js"></script>
<script>
ringView({{Q:{J(L['pos'])},ST:{J(L['strata'])},yk:(k,x)=>30+40*k,stubL:62,stubR:318,bulge:40,greenBulge:24}});
</script>'''))
print('ok')
