// Shared code for the stratified views (pages 07 and 08): one necklace per line or arc,
// nodes colored by necklace. opt.Q = 64 positions, opt.guides = SVG markup drawn behind.
function strataView(opt) {
  const $ = id => document.getElementById(id), NS = 'http://www.w3.org/2000/svg', svg = $('g');
  const rot = (s, k) => { k = ((k % 6) + 6) % 6; return k ? ((s << k) | (s >> (6 - k))) & 63 : s };
  const canon = s => { let m = s; for (let k = 1; k < 6; k++) m = Math.min(m, rot(s, k)); return m };
  const pc = s => { let c = 0; while (s) { c += s & 1; s >>= 1 } return c };
  const pat = s => Array.from({ length: 6 }, (_, i) => (s >> i) & 1 ? '●' : '○').join('');
  const COL = { 0: '#B4B2A9', 63: '#5F5E5A', 1: '#F0997B', 31: '#993C1D', 3: '#85B7EB', 15: '#185FA5', 5: '#5DCAA5', 23: '#0F6E56', 9: '#AFA9EC', 27: '#534AB7', 7: '#BA7517', 21: '#639922', 11: '#ED93B1', 13: '#993556' };
  const ORDER = [0, 1, 3, 9, 5, 11, 7, 21, 13, 23, 27, 15, 31, 63];
  const CL = {}, SZ = {}; for (let s = 0; s < 64; s++) { const c = canon(s); CL[s] = c; SZ[c] = (SZ[c] || 0) + 1 }
  const Q = opt.Q;
  svg.insertAdjacentHTML('beforeend', `<g>${opt.guides}</g>`);
  const eg = document.createElementNS(NS, 'g'); eg.setAttribute('stroke', 'var(--text-muted)'); eg.setAttribute('stroke-width', '.6'); svg.appendChild(eg);
  const E = []; for (let s = 0; s < 64; s++) for (let i = 0; i < 6; i++) {
    const t = s ^ (1 << i); if (s < t) {
      const l = document.createElementNS(NS, 'line'), [x1, y1] = Q[s], [x2, y2] = Q[t], dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy), o = 7 / L;
      l.setAttribute('x1', (x1 + dx * o).toFixed(1)); l.setAttribute('y1', (y1 + dy * o).toFixed(1)); l.setAttribute('x2', (x2 - dx * o).toFixed(1)); l.setAttribute('y2', (y2 - dy * o).toFixed(1)); l.setAttribute('stroke-opacity', '.4'); eg.appendChild(l); E.push([s, t, l])
    }
  }
  const V = []; for (let s = 0; s < 64; s++) { const c = document.createElementNS(NS, 'circle'); c.setAttribute('cx', Q[s][0]); c.setAttribute('cy', Q[s][1]); c.setAttribute('r', '6.5'); c.setAttribute('class', 'v'); c.setAttribute('fill', COL[CL[s]]); c.setAttribute('stroke', 'var(--text-secondary)'); c.setAttribute('stroke-width', '.75'); c.dataset.s = s; svg.appendChild(c); V.push(c) }
  $('leg').innerHTML = ORDER.map(c => `<div class="lg" data-c="${c}"><span style="width:12px;height:12px;flex:none;border-radius:50%;background:${COL[c]};border:.5px solid var(--border-strong)"></span><span class="mono">${pat(c)}</span><span>×${SZ[c]}</span></div>`).join('');
  let C = null, S = null;
  function paint() {
    V.forEach((c, s) => {
      const mem = C !== null && CL[s] === C; c.setAttribute('opacity', C === null || mem ? '1' : '.15');
      c.setAttribute('stroke', s === S ? 'var(--text-accent)' : mem ? 'var(--text-primary)' : 'var(--text-secondary)'); c.setAttribute('stroke-width', s === S ? '3' : mem ? '2' : '.75')
    });
    E.forEach(([a, b, l]) => { const on = S !== null && (a === S || b === S); l.setAttribute('stroke', on ? '#D85A30' : 'var(--text-muted)'); l.setAttribute('stroke-width', on ? '2' : '.6'); l.setAttribute('stroke-opacity', on ? '1' : C === null ? '.4' : '.15'); if (on) eg.appendChild(l) });
    document.querySelectorAll('.lg').forEach(d => d.classList.toggle('on', C !== null && +d.dataset.c === C));
    if (C === null) { $('out').innerHTML = 'Tap a node or a necklace to highlight its class.'; return }
    const top = S !== null ? `<div><span class="mono">${pat(S)}</span> &nbsp;${pc(S)} filled, ${6 - pc(S)} open</div>` : '';
    $('out').innerHTML = top + `<div>Necklace <span class="mono">${pat(C)}</span> · ${SZ[C]} rotation${SZ[C] === 1 ? '' : 's'}</div>`
  }
  svg.addEventListener('click', e => { const c = e.target.closest('.v'); if (!c) { C = null; S = null; paint(); return } const s = +c.dataset.s; if (S === s) { S = null; C = null } else { S = s; C = CL[s] } paint() });
  $('leg').addEventListener('click', e => { const d = e.target.closest('.lg'); if (!d) return; const c = +d.dataset.c; S = null; C = C === c ? null : c; paint() });
  paint();
}
