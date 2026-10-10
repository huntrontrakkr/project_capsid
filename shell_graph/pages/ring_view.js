// Shared code for the ring-glyph views (pages 09 and 10): each state drawn as its shell
// pattern, each necklace's rotation ring drawn along its row and wrapping off both edges.
// opt: Q positions, ST strata (rotation order, left to right), yk(k,x) row curve,
// stubL/stubR wrap stub ends, bulge (amber side connectors), greenBulge.
function ringView(opt) {
  const $ = id => document.getElementById(id), svg = $('g'), f = v => v.toFixed(1);
  const rot = (s, k) => { k = ((k % 6) + 6) % 6; return k ? ((s << k) | (s >> (6 - k))) & 63 : s };
  const canon = s => { let m = s; for (let k = 1; k < 6; k++) m = Math.min(m, rot(s, k)); return m };
  const pc = s => { let c = 0; while (s) { c += s & 1; s >>= 1 } return c };
  const pat = s => Array.from({ length: 6 }, (_, i) => (s >> i) & 1 ? '●' : '○').join('');
  const { Q, ST, yk } = opt, h = .01;
  const sl = (k, x) => (yk(k, x + h) - yk(k, x - h)) / (2 * h), dl = (k, x) => 11 / Math.sqrt(1 + sl(k, x) ** 2);
  const seg = (k, a, b) => { const ya = yk(k, a), cx = (a + b) / 2, cy = ya + sl(k, a) * (cx - a); return `M${f(a)} ${f(ya)}Q${f(cx)} ${f(cy)} ${f(b)} ${f(yk(k, b))}` };
  let mesh = '';
  for (let s = 0; s < 64; s++) for (let i = 0; i < 6; i++) {
    const t = s ^ (1 << i); if (s < t) {
      const [x1, y1] = Q[s], [x2, y2] = Q[t], dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy), o = 10.5 / L;
      mesh += `<line data-e="${s}-${t}" x1="${f(x1 + dx * o)}" y1="${f(y1 + dy * o)}" x2="${f(x2 - dx * o)}" y2="${f(y2 - dy * o)}"/>`
    }
  }
  let ring = '', wrap = '';
  ST.forEach((row, k) => {
    if (row.length < 2 || k === 7) return;
    for (let j = 0; j + 1 < row.length; j++) { const a = Q[row[j]][0], b = Q[row[j + 1]][0]; ring += `<path d="${seg(k, a + dl(k, a), b - dl(k, b))}"/>` }
    if (k === 6 || k === 8) return;
    const L = Q[row[0]][0], R = Q[row[row.length - 1]][0];
    wrap += `<path d="${seg(k, L - dl(k, L), opt.stubL)}" marker-end="url(#arrow)"/><path d="${seg(k, R + dl(k, R), opt.stubR)}" marker-end="url(#arrow)"/>`
  });
  const B = opt.bulge, [ua, , ub] = ST[6].map(s => Q[s]), [la, , lb] = ST[8].map(s => Q[s]);
  ring += `<path d="M${ub[0] + 10} ${ub[1] + 4}C${ub[0] + B} ${ub[1] + 14} ${lb[0] + B} ${lb[1] - 14} ${lb[0] + 10} ${lb[1] - 4}"/>`;
  ring += `<path d="M${la[0] - 10} ${la[1] - 4}C${la[0] - B} ${la[1] - 14} ${ua[0] - B} ${ua[1] + 14} ${ua[0] - 10} ${ua[1] + 4}"/>`;
  const G = opt.greenBulge, [gL, gR] = ST[7].map(s => Q[s]);
  ring += `<path d="M${gL[0] + 7} ${gL[1] - 7}Q190 ${gL[1] - G} ${gR[0] - 7} ${gR[1] - 7}"/><path d="M${gR[0] - 7} ${gR[1] + 7}Q190 ${gR[1] + G} ${gL[0] + 7} ${gL[1] + 7}"/>`;
  let nodes = '';
  for (let s = 0; s < 64; s++) {
    const [x, y] = Q[s]; let d = '';
    for (let i = 0; i < 6; i++) {
      const a = (-90 + i * 60) * Math.PI / 180, px = f(x + 5.6 * Math.cos(a)), py = f(y + 5.6 * Math.sin(a));
      d += (s >> i) & 1 ? `<circle cx="${px}" cy="${py}" r="1.9" fill="var(--text-primary)"/>` : `<circle cx="${px}" cy="${py}" r="1.6" fill="none" stroke="var(--text-muted)" stroke-width=".8"/>`
    }
    nodes += `<g class="v" data-s="${s}"><circle class="b" cx="${x}" cy="${y}" r="9.5"/>${d}</g>`
  }
  svg.insertAdjacentHTML('beforeend', `<g id="mesh" stroke="var(--text-muted)" stroke-opacity=".35" stroke-width=".6">${mesh}</g><g fill="none" stroke="#1D9E75" stroke-width="1.6" stroke-linecap="round">${ring}</g><g fill="none" stroke="#1D9E75" stroke-width="1.2" stroke-dasharray="3 3">${wrap}</g>${nodes}`);
  let S = null;
  function paint() {
    svg.querySelectorAll('.v').forEach(g => g.classList.remove('s', 'n'));
    svg.querySelectorAll('#mesh line').forEach(l => { const [a, b] = l.dataset.e.split('-').map(Number), on = S !== null && (a === S || b === S); l.setAttribute('stroke', on ? '#D85A30' : 'var(--text-muted)'); l.setAttribute('stroke-opacity', on ? '1' : '.35'); l.setAttribute('stroke-width', on ? '2' : '.6') });
    if (S === null) { $('out').innerHTML = 'Tap a state to see its pattern and its six one-unit neighbors.'; return }
    svg.querySelector(`.v[data-s="${S}"]`).classList.add('s'); for (let i = 0; i < 6; i++) svg.querySelector(`.v[data-s="${S ^ (1 << i)}"]`).classList.add('n');
    const sz = new Set([0, 1, 2, 3, 4, 5].map(k => rot(S, k))).size;
    $('out').innerHTML = `<div><span class="mono">${pat(S)}</span> &nbsp;${pc(S)} filled, ${6 - pc(S)} open</div><div>Necklace <span class="mono">${pat(canon(S))}</span> · ${sz} rotation${sz === 1 ? '' : 's'}</div>`
  }
  svg.addEventListener('click', e => { const g = e.target.closest('.v'); S = g ? (+g.dataset.s === S ? null : +g.dataset.s) : null; paint() });
  paint();
}
