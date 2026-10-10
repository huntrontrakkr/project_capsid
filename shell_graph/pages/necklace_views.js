// Shared code for the necklace-colored views (pages 04 to 06).
// Views: P = 12-gon, L = layered rows, C = necklace rows (blocks in rotation order),
// S = spread (each necklace in its own vertical column).
function necklaceViews(opt) {
  const $ = id => document.getElementById(id), NS = 'http://www.w3.org/2000/svg', svg = $('g');
  const rot = (s, k) => { k = ((k % 6) + 6) % 6; return k ? ((s << k) | (s >> (6 - k))) & 63 : s };
  const canon = s => { let m = s; for (let k = 1; k < 6; k++) m = Math.min(m, rot(s, k)); return m };
  const pc = s => { let c = 0; while (s) { c += s & 1; s >>= 1 } return c };
  const pat = s => Array.from({ length: 6 }, (_, i) => (s >> i) & 1 ? '●' : '○').join('');
  const COL = { 0: '#B4B2A9', 63: '#5F5E5A', 1: '#F0997B', 31: '#993C1D', 3: '#85B7EB', 15: '#185FA5', 5: '#5DCAA5', 23: '#0F6E56', 9: '#AFA9EC', 27: '#534AB7', 7: '#BA7517', 21: '#639922', 11: '#ED93B1', 13: '#993556' };
  const ORDER = [0, 1, 3, 5, 9, 7, 11, 13, 21, 15, 23, 27, 31, 63];
  const CL = {}, SZ = {}; for (let s = 0; s < 64; s++) { const c = canon(s); CL[s] = c; SZ[c] = (SZ[c] || 0) + 1 }
  // necklace rows: blocks in rotation order, 2- and 4-filled rows aligned by complement
  const ROWS = [[0], [32, 1, 2, 4, 8, 16], [34, 5, 10, 20, 40, 17, 18, 36, 9, 48, 33, 3, 6, 12, 24], [42, 21, 50, 37, 11, 22, 44, 25, 13, 26, 52, 41, 19, 38, 7, 14, 28, 56, 49, 35], [29, 58, 53, 43, 23, 46, 45, 27, 54, 15, 30, 60, 57, 51, 39], [31, 62, 61, 59, 55, 47], [63]];
  // spread: each necklace in its own column, alternating pair at the center (search output)
  const SPR = [[190, 18], [190, 85], [190, 71], [240, 193], [190, 57], [140, 179], [240, 179], [340, 283], [190, 127], [40, 214], [140, 193], [290, 297], [240, 165], [90, 283], [340, 269], [240, 387], [190, 113], [140, 235], [40, 200], [90, 339], [140, 207], [190, 311], [290, 283], [140, 387], [240, 235], [290, 339], [90, 297], [40, 422], [340, 339], [140, 443], [240, 401], [190, 509], [190, 99], [240, 207], [140, 165], [340, 297], [40, 186], [290, 311], [90, 269], [240, 373], [140, 221], [90, 325], [190, 297], [140, 401], [290, 269], [40, 408], [140, 373], [190, 495], [240, 221], [340, 311], [290, 325], [240, 443], [90, 311], [140, 415], [40, 394], [190, 481], [340, 325], [240, 429], [140, 429], [190, 551], [240, 415], [190, 537], [190, 523], [190, 590]];
  const PET = {}, LAY = {}, ROW = {}, SP = {}, raw = {}, th = i => (i * 30 + 15) * Math.PI / 180, ry = opt.rowY;
  for (let s = 0; s < 64; s++) { let x = 0, y = 0; for (let i = 0; i < 6; i++) { const c = ((s >> i) & 1) - .5; x += c * Math.cos(th(i)); y += c * Math.sin(th(i)) } raw[s] = [x, y]; PET[s] = [190 + 88 * x, opt.petCy + 88 * y] }
  const grp = {}; for (let s = 0; s < 64; s++) { const k = Math.round(raw[s][0] * 1000) + ',' + Math.round(raw[s][1] * 1000); (grp[k] = grp[k] || []).push(s) }
  Object.values(grp).forEach(g => { if (g.length < 2) return; const r = g.length > 2 ? 11 : 8.5; g.forEach((s, j) => { const a = Math.PI / 4 + 2 * Math.PI * j / g.length; PET[s] = [PET[s][0] + r * Math.cos(a), PET[s][1] + r * Math.sin(a)] }) });
  const LROWS = []; for (let w = 0; w <= 6; w++) { const row = [...Array(64).keys()].filter(s => pc(s) === w).sort((a, b) => PET[a][0] - PET[b][0] || PET[a][1] - PET[b][1]); LROWS.push(row); const n = row.length; row.forEach((s, j) => LAY[s] = [190 + (j - (n - 1) / 2) * 17.9, ry(w)]) }
  ROWS.forEach((row, w) => { const xs = [0]; for (let j = 1; j < row.length; j++) xs.push(xs[j - 1] + 15.5 + (CL[row[j]] !== CL[row[j - 1]] ? 12 : 0)); const off = 190 - xs[xs.length - 1] / 2; row.forEach((s, j) => ROW[s] = [off + xs[j], ry(w)]) });
  SPR.forEach((p, s) => SP[s] = p);
  function cross(rows) {
    const pos = {}; rows.forEach(r => r.forEach((s, j) => pos[s] = j)); let t = 0;
    for (let w = 0; w < 6; w++) {
      const E = []; rows[w].forEach(s => { for (let i = 0; i < 6; i++) if (!((s >> i) & 1)) E.push([pos[s], pos[s | (1 << i)]]) });
      for (let a = 0; a < E.length; a++) for (let b = a + 1; b < E.length; b++) if ((E[a][0] - E[b][0]) * (E[a][1] - E[b][1]) < 0) t++
    } return t
  }
  const XC = { L: cross(LROWS), C: cross(ROWS) };
  const bg = document.createElementNS(NS, 'g'); bg.setAttribute('opacity', '0'); bg.style.transition = 'opacity .6s'; svg.appendChild(bg);
  if (opt.bands) [40, 90, 140, 190, 240, 290, 340].forEach(x => { const r = document.createElementNS(NS, 'rect'); r.setAttribute('x', x - 12); r.setAttribute('y', x === 190 ? 6 : 150); r.setAttribute('width', 24); r.setAttribute('height', x === 190 ? 596 : 308); r.setAttribute('rx', 12); r.setAttribute('fill', 'var(--surface-1)'); r.setAttribute('stroke', 'var(--border)'); r.setAttribute('stroke-width', '.5'); bg.appendChild(r) });
  const eg = document.createElementNS(NS, 'g'); eg.setAttribute('stroke', 'var(--text-muted)'); eg.setAttribute('stroke-width', '.6'); eg.setAttribute('fill', 'none'); svg.appendChild(eg);
  const E = []; for (let s = 0; s < 64; s++) for (let i = 0; i < 6; i++) { const t = s ^ (1 << i); if (s < t) { const l = document.createElementNS(NS, 'path'); l.setAttribute('stroke-opacity', '.4'); eg.appendChild(l); E.push([s, t, l, i]) } }
  const V = []; for (let s = 0; s < 64; s++) { const c = document.createElementNS(NS, 'circle'); c.setAttribute('r', opt.radius); c.setAttribute('class', 'v'); c.setAttribute('fill', COL[CL[s]]); c.setAttribute('stroke', 'var(--text-secondary)'); c.setAttribute('stroke-width', '.75'); c.dataset.s = s; svg.appendChild(c); V.push(c) }
  $('leg').innerHTML = ORDER.map(c => `<div class="lg" data-c="${c}"><span style="width:12px;height:12px;flex:none;border-radius:50%;background:${COL[c]};border:.5px solid var(--border-strong)"></span><span class="mono">${pat(c)}</span><span>×${SZ[c]}</span></div>`).join('');
  const POS = { S: SP, C: ROW, L: LAY, P: PET };
  let mode = opt.start, cur = {}, C = null, S = null; for (let s = 0; s < 64; s++) cur[s] = POS[mode][s].slice();
  function draw() {
    V.forEach((c, s) => { c.setAttribute('cx', cur[s][0].toFixed(1)); c.setAttribute('cy', cur[s][1].toFixed(1)) });
    E.forEach(([a, b, l, i]) => {
      const [x1, y1] = cur[a], [x2, y2] = cur[b], dx = x2 - x1, dy = y2 - y1; let cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
      if (Math.abs(dx) < 6) { const k = Math.max(0, 1 - Math.abs(dx) / 6); cx += (i % 2 ? 1 : -1) * k * (10 + Math.abs(dy) * .3) }
      const u = (px, py) => { const ex = cx - px, ey = cy - py, L = Math.hypot(ex, ey) || 1, o = Math.min(.45 * L, 7); return [px + ex / L * o, py + ey / L * o] };
      const [sx, sy] = u(x1, y1), [tx, ty] = u(x2, y2); l.setAttribute('d', `M${sx.toFixed(1)} ${sy.toFixed(1)}Q${cx.toFixed(1)} ${cy.toFixed(1)} ${tx.toFixed(1)} ${ty.toFixed(1)}`)
    })
  }
  function setCap() { if ($('cap')) $('cap').textContent = 'Top: all open · bottom: all filled' + (opt.crossings && XC[mode] !== undefined ? ` · ${XC[mode]} edge crossings` : '') }
  function go(m) {
    if (m === mode) return; mode = m; opt.views.forEach(k => $('b' + k).setAttribute('aria-pressed', k === m)); bg.setAttribute('opacity', m === 'S' ? '1' : '0'); setCap();
    const to = POS[m], from = {}; for (let s = 0; s < 64; s++) from[s] = cur[s].slice();
    const D = matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 900, t0 = performance.now();
    function f(now) {
      const t = D ? Math.min(1, (now - t0) / D) : 1, e = t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      for (let s = 0; s < 64; s++) cur[s] = [from[s][0] + (to[s][0] - from[s][0]) * e, from[s][1] + (to[s][1] - from[s][1]) * e]; draw(); if (t < 1) requestAnimationFrame(f)
    }
    requestAnimationFrame(f)
  }
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
  opt.views.forEach(k => $('b' + k).onclick = () => go(k));
  draw(); paint(); setCap();
  if (opt.then) setTimeout(() => go(opt.then), 500);
}
