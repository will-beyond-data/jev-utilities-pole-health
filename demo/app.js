/* Pole health triage with Matilda Jev: static demo front end.
   No framework, no build step. Reads data/*.json, or sample/*.json when data/ is missing. */
(() => {
'use strict';

// ---------------------------------------------------------------- helpers
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ESC[c]);
// No em or en dashes in anything we show, whatever the data files contain.
const clean = (s) => esc(String(s ?? '').replace(/\s*[—–]\s*/g, ', '));
const num = (x) => (typeof x === 'number' && isFinite(x) ? x : null);
const MINUS = '−';
const pct = (p) => {
  if (num(p) === null) return '-';
  const v = p * 100;
  if (p > 0 && v < 0.5) return '<1%';
  return Math.round(v) + '%';
};
const pct1 = (p) => (num(p) === null ? '-' : (p * 100).toFixed(1) + '%');
const signedPct = (f) => (num(f) === null ? '-' : (f < 0 ? MINUS : '+') + Math.round(Math.abs(f) * 100) + '%');
const words = (s) => String(s ?? '').replace(/_/g, ' ');
const cap = (s) => { s = words(s); return s.charAt(0).toUpperCase() + s.slice(1); };
const ms1 = (x) => (num(x) === null ? '-' : x >= 100 ? Math.round(x) : x.toFixed(1));
const fmtDate = (iso) => {
  const d = new Date(iso);
  return isNaN(d) ? String(iso || '') : d.toISOString().slice(0, 16).replace('T', ' ') + ' UTC';
};
const usd = (x) => (num(x) === null ? '-' : '$' + x.toLocaleString('en-AU', { minimumFractionDigits: 2, maximumFractionDigits: 2 }));

const ACTIONS = ['replace', 'maintain', 'defer', 'reinspect'];
const STATE_LABEL = { replace: 'Replace', maintain: 'Maintain', defer: 'Defer', engineer: 'Engineer review', pending: 'Not inspected' };
const SHAPES = {
  replace: '<path d="M10 2.2 18.2 17.2H1.8Z"/>',
  maintain: '<path d="M3.2 3.2h13.6v13.6H3.2Z"/>',
  defer: '<circle cx="10" cy="10" r="7.4"/>',
  engineer: '<path d="M10 1.4 18.6 10 10 18.6 1.4 10Z"/>',
  pending: '<circle cx="10" cy="10" r="4"/>',
};
const shapeSvg = (state, cls = '') => `<svg class="shape ${state} ${cls}" viewBox="0 0 20 20" aria-hidden="true">${SHAPES[state]}</svg>`;
const ACTION_STATE = { replace: 'replace', maintain: 'maintain', defer: 'defer', reinspect: 'engineer' };
const ACTION_VAR = { replace: '--c-replace', maintain: '--c-maintain', defer: '--c-defer', reinspect: '--c-engineer' };

// ---------------------------------------------------------------- state
const S = {
  base: 'data/', usingSample: false,
  poles: [], byId: new Map(), meta: {}, decisions: {}, dmeta: {}, flip: null, evalS: null,
  order: [],              // ids in replay order (decisions.json order, restricted to known poles)
  markers: new Map(),     // id -> L.marker
  decided: new Set(),
  selected: null,
  view: 'map',
  threshold: 0.6,
};
const R = {               // replay engine
  idx: 0, running: false, speed: '1', credit: 0, last: 0, raf: 0,
  msSorted: [], stamps: [], activeMs: 0, feed: [],
  tally: { replace: 0, maintain: 0, defer: 0, engineer: 0 },
  dirty: true,
};

// ---------------------------------------------------------------- data loading
async function getJSON(url) {
  const r = await fetch(url, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}
async function optJSON(url) { try { return await getJSON(url); } catch { return null; } }
async function loadSet(base) {
  const [poles, decisions] = await Promise.all([getJSON(base + 'poles.json'), getJSON(base + 'decisions.json')]);
  const [flip, evalS] = await Promise.all([optJSON(base + 'flip.json'), optJSON(base + 'eval_summary.json')]);
  return { poles, decisions, flip, evalS };
}
async function loadData() {
  let set = null;
  try { set = await loadSet('data/'); S.base = 'data/'; }
  catch (e) {
    try { set = await loadSet('sample/'); S.base = 'sample/'; S.usingSample = true; }
    catch (e2) { throw new Error('Could not load data/ or sample/. Serve the demo folder over HTTP: python -m http.server -d demo 8000'); }
  }
  S.meta = set.poles;
  S.poles = (set.poles.poles || []).filter((p) => num(p.lat) !== null && num(p.lon) !== null);
  S.byId = new Map(S.poles.map((p) => [p.id, p]));
  S.dmeta = set.decisions;
  S.decisions = set.decisions.decisions || {};
  S.threshold = num(set.decisions.engineer_threshold) ?? 0.6;
  for (const [id, d] of Object.entries(S.decisions)) normaliseDecision(d);
  S.order = Object.keys(S.decisions).filter((id) => S.byId.has(id));
  S.flip = normFlip(set.flip);
  S.evalS = normEval(set.evalS);
}
const isMock = () => S.dmeta.backend === 'mock' || S.dmeta.mock === true || S.usingSample;

function answerOf(d, pass, key) { return d && d[pass] && d[pass].answers && d[pass].answers[key] || null; }
function normaliseDecision(d) {
  if (!d) return;
  const act = answerOf(d, 'v2', 'action');
  if (!d.action && act) d.action = act.choice;
  if (num(d.confidence) === null && act) {
    const p = act.probabilities && act.probabilities[d.action];
    d.confidence = num(act.confidence) ?? num(p) ?? 0;
  }
  if (!d.route) d.route = (d.confidence < S.threshold) ? 'engineer' : 'auto';
}
const passMs = (p) => (p ? (num(p.model_ms) ?? num(p.latency_ms)) : null);
// Model time for one pole: the decision's own model_ms if present, else photo read + decision passes.
function poleMs(d) {
  const own = num(d.model_ms) ?? num(d.latency_ms);
  if (own !== null) return own;
  const a = passMs(d.v1), b = passMs(d.v2);
  if (a === null && b === null) return null;
  return (a || 0) + (b || 0);
}
const stateOf = (d) => (d.route === 'engineer' || d.action === 'reinspect') ? 'engineer' : (ACTION_STATE[d.action] || 'defer');
const photoUrl = (p) => (!p ? null : /^(https?:|data:|\/)/.test(p) ? p : S.base + p);

// ---------------------------------------------------------------- shared renderers
function bar(label, p, opts = {}) {
  const w = Math.max(0, Math.min(1, p || 0)) * 100;
  const style = opts.color ? ` style="--ac:var(${opts.color})"` : '';
  return `<div class="bar${opts.top ? ' top' : ''}"${style} title="${esc(label)}: ${pct1(p)}">
    <span class="lb">${esc(opts.rawLabel ? label : cap(label))}</span>
    <span class="tr"><span class="fl" style="width:${w.toFixed(1)}%"></span></span>
    <span class="pc">${pct(p)}</span></div>`;
}
function choiceBars(ans, cls = '') {
  if (!ans || !ans.probabilities) return '<p class="reason">No answer recorded.</p>';
  const entries = Object.entries(ans.probabilities);
  const max = Math.max(...entries.map(([, p]) => p));
  return `<div class="bars ${cls}">${entries.map(([k, p]) => bar(k, p, { top: p === max })).join('')}</div>`;
}
function actionBars(ans) {
  if (!ans || !ans.probabilities) return '<p class="reason">No answer recorded.</p>';
  const entries = Object.entries(ans.probabilities);
  const max = Math.max(...entries.map(([, p]) => p));
  return `<div class="bars actions">${entries.map(([k, p]) => bar(k, p, { top: p === max, color: ACTION_VAR[k] })).join('')}</div>`;
}
function shortLevel(legend, k) {
  const t = legend && legend[String(k)];
  return t ? String(t).split(':')[0] : String(k);
}
function healthBlock(h) {
  if (!h) return '';
  const probs = h.probabilities || {};
  const keys = Object.keys(probs).sort((a, b) => a - b);
  const n = keys.length || 5;
  const legend = h.legend || {};
  const score = num(h.score);
  const level = score === null ? -1 : Math.round(score);
  const pos = score === null ? 0 : ((score + 0.5) / n) * 100;
  const segs = Array.from({ length: n }, (_, k) => `<span class="hs-seg" style="--a:${Math.round(18 + (72 * k) / Math.max(n - 1, 1))}%"></span>`).join('');
  const labels = Array.from({ length: n }, (_, k) => `<span class="${k === level ? 'on' : ''}">${esc(shortLevel(legend, k))}</span>`).join('');
  const max = Math.max(...keys.map((k) => probs[k]), 0);
  const rows = keys.map((k) => bar(shortLevel(legend, k), probs[k], { top: probs[k] === max })).join('');
  return `<div class="qgroup">
    <div class="qname">Health<span>5-level scale</span></div>
    <div class="hscore"><b>${esc(level >= 0 ? cap(shortLevel(legend, level)) : '-')}</b><span>health, 5-level scale</span></div>
    <div class="healthscale" role="img" aria-label="Health: ${esc(level >= 0 ? shortLevel(legend, level) : 'unknown')}, on a 5-level scale">
      <div class="hs-row">${segs}${score === null ? '' : `<span class="hs-marker" style="left:${pos.toFixed(1)}%"></span>`}</div>
      <div class="hs-labels">${labels}</div>
    </div>
    <div class="bars">${rows}</div></div>`;
}
function v1Block(d) {
  const groups = [['lean', 'Lean'], ['crossarm', 'Crossarm'], ['vegetation', 'Vegetation']].map(([k, name]) => {
    const a = answerOf(d, 'v1', k);
    if (!a) return '';
    return `<div class="qgroup"><div class="qname">${name}<span>${esc(cap(a.choice))}</span></div>${choiceBars(a)}</div>`;
  }).join('');
  return groups + healthBlock(answerOf(d, 'v1', 'health'));
}
function recordRows(rec) {
  if (!rec) return [];
  const since = rec.years_since_maintenance;
  const crit = rec.critical_customer ? `Yes: ${words(rec.critical_customer_type || 'critical customer')}` : 'No';
  return [
    ['age', 'Age', num(rec.age_years) === null ? '-' : `${rec.age_years} years`],
    ['install', 'Installed', rec.install_year ?? '-'],
    ['maint', 'Since maintenance', since === null || since === undefined ? 'Never recorded' : `${since} years`],
    ['bush', 'Bushfire zone', cap(rec.bushfire_zone || '-')],
    ['cust', 'Customers served', num(rec.customers_served) === null ? '-' : String(rec.customers_served)],
    ['crit', 'Critical customer', crit],
    ['trans', 'Transformer', cap(rec.transformer || 'none')],
    ['health', 'Last health index', (() => { const l = (rec.inspections || []).slice(-1)[0]; return l && l.health_index != null ? `${l.health_index} of 5` : '-'; })()],
  ];
}
function kvBlock(rec, other) {
  const a = recordRows(rec);
  const b = other ? Object.fromEntries(recordRows(other).map((r) => [r[0], r[2]])) : null;
  return `<dl class="kv${other ? ' flipkv' : ''}">${a.map(([k, l, v]) => `<div class="${b && String(b[k]) !== String(v) ? 'diff' : ''}"><dt>${l}</dt><dd>${clean(v)}</dd></div>`).join('')}</dl>`;
}
const meanOf = (xs) => { const v = (xs || []).filter((x) => num(x) !== null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
function sparkline(rec) {
  const pts = ((rec && rec.inspections) || []).map((i) => ({ year: i.year, v: meanOf(i.shell_thickness) })).filter((p) => p.v !== null);
  if (pts.length < 2) return '<p class="reason">No shell thickness history recorded.</p>';
  const W = 340, H = 104, L = 22, Rr = 22, T = 26, B = 22;
  const lo = Math.min(...pts.map((p) => p.v)), hi = Math.max(...pts.map((p) => p.v));
  const pad = (hi - lo || 0.1) * 0.35;
  const y0 = lo - pad, y1 = hi + pad;
  const x = (i) => L + (i * (W - L - Rr)) / (pts.length - 1);
  const y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const line = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(p.v).toFixed(1)}`).join('');
  const area = `${line}L${x(pts.length - 1).toFixed(1)} ${H - B}L${x(0).toFixed(1)} ${H - B}Z`;
  const change = num(rec.shell_thickness_change_10y);
  const dots = pts.map((p, i) => `<circle class="dot" cx="${x(i).toFixed(1)}" cy="${y(p.v).toFixed(1)}" r="5"><title>${p.year}: ${p.v.toFixed(2)}</title></circle>`).join('');
  const vals = pts.map((p, i) => `<text class="val" x="${x(i).toFixed(1)}" y="${(y(p.v) - 10).toFixed(1)}" text-anchor="${i === 0 ? 'start' : i === pts.length - 1 ? 'end' : 'middle'}">${p.v.toFixed(2)}</text>`).join('');
  const yrs = pts.map((p, i) => `<text x="${x(i).toFixed(1)}" y="${H - 6}" text-anchor="${i === 0 ? 'start' : i === pts.length - 1 ? 'end' : 'middle'}">${p.year}</text>`).join('');
  const label = `Shell thickness by inspection: ${pts.map((p) => `${p.year} ${p.v.toFixed(2)}`).join(', ')}`;
  return `<div class="spark"><div class="cap"><span>Shell thickness, mean of 3 readings</span><b class="num">${change === null ? '' : signedPct(change) + ' over 10 years'}</b></div>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}"><line class="ax" x1="${L - 8}" x2="${W - Rr + 8}" y1="${H - B + 2}" y2="${H - B + 2}"/>
    <path class="ar" d="${area}"/><path class="ln" d="${line}"/>${dots}${vals}${yrs}</svg></div>`;
}
function photoHTML(path, alt, credit) {
  const u = photoUrl(path);
  return `<div class="photo ${u ? '' : 'missing'}">${u ? `<img data-ph src="${esc(u)}" alt="${esc(alt)}">` : ''}
    <div class="ph"><svg viewBox="0 0 48 48" aria-hidden="true"><rect x="6" y="10" width="36" height="28" rx="4"/><circle cx="18" cy="21" r="3.5"/><path d="M8 36l11-10 7 6 6-5 8 8"/></svg><span>Photo not available in this build</span></div></div>${credit ? `<div class="credit">${credit}</div>` : ''}`;
}
function wirePhotos(root) {
  $$('img[data-ph]', root).forEach((img) => {
    const bad = () => img.closest('.photo').classList.add('missing');
    img.addEventListener('error', bad, { once: true });
    if (img.complete && img.naturalWidth === 0) bad();
  });
}
function creditLine(src) {
  if (!src) return '';
  const c = clean(src.credit || '').replace(/\.$/, '');
  const lab = src.label ? `Source label: ${esc(src.label)}` : '';
  return [c, lab].filter(Boolean).join('. ') + '.';
}
function routeReason(d) {
  const lean = answerOf(d, 'v1', 'lean');
  const parts = [];
  if (d.confidence < S.threshold) parts.push(`Top action at ${pct(d.confidence)}, under the ${pct(S.threshold)} threshold`);
  if (lean && lean.choice === 'cannot_assess') parts.push('Lean cannot be assessed from the photo');
  if (d.action === 'reinspect') parts.push('The model asks for a crew to reinspect');
  if (!parts.length) return d.route === 'engineer' ? 'Routed to an engineer.' : `Confident enough to act without review (${pct(d.confidence)}, threshold ${pct(S.threshold)}).`;
  return parts.join('. ') + '.';
}
function latencyLine(d) {
  const a = passMs(d.v1), b = passMs(d.v2), t = poleMs(d);
  const wall = (num(d.v1 && d.v1.latency_ms) || 0) + (num(d.v2 && d.v2.latency_ms) || 0);
  return `<div class="latency"><span><b class="num">${ms1(t)} ms</b> model time${a !== null && b !== null ? ` (photo read ${ms1(a)} + decision ${ms1(b)})` : ''}</span>${wall ? `<span><b class="num">${ms1(wall)} ms</b> round trip</span>` : ''}</div>`;
}
function decisionHead(d, small) {
  const st = stateOf(d);
  const act = d.action;
  return `<div class="decision">${shapeSvg(st)}<div class="what"><b>${esc(act)}</b><span>${pct(d.confidence)} confident${st === 'engineer' ? ', engineer review' : ''}</span></div></div>`;
}
function safetyBar(v2) {
  const s = answerOf(v2 ? { v2 } : null, 'v2', 'safety_risk_now');
  if (!s || num(s.noul) === null) return '';
  return `<div class="qgroup"><div class="qname">Public safety risk before the next inspection<span>probability</span></div><div class="bars">${bar('Safety risk', s.noul, { top: true, rawLabel: true })}</div></div>`;
}

// ---------------------------------------------------------------- pole panel
function emptyPanel() {
  const rows = R.feed.slice(0, 9).map((f) => `<li><button type="button" data-pole="${esc(f.id)}">${shapeSvg(f.state, 'sm')}<span><span class="act">${esc(f.action)}</span> <span class="id">${esc(f.id)}</span></span><span class="ms num">${ms1(f.ms)} ms</span></button></li>`).join('');
  return `<div class="panel-empty">
    <div><h2>Click a pole to see why</h2>
    <p>Each coloured marker is one decision. Open it to see the photo, what the model read in it, the pole's record and the action it recommended.</p></div>
    <div><div class="h-sub" style="margin-bottom:6px">Latest decisions</div>
    ${rows ? `<ul class="feed">${rows}</ul>` : '<p class="reason">Nothing yet. Press Run inspection to start.</p>'}</div></div>`;
}
function polePanel(id) {
  const p = S.byId.get(id);
  if (!p) return emptyPanel();
  const d = S.decisions[id];
  const decided = d && S.decided.has(id);
  const rec = p.record;
  let head;
  if (decided) {
    const st = stateOf(d);
    head = `<div class="p-head"><div class="top"><span class="pid">${esc(id)}</span><button class="close" type="button" data-close aria-label="Close pole panel">&times;</button></div>
      ${decisionHead(d)}
      <div class="badges"><span class="badge-pill ${st === 'engineer' ? 'engineer' : 'auto'}">${st === 'engineer' ? 'Engineer review' : 'Auto decision'}</span></div>
      <p class="reason">${esc(routeReason(d))}</p>${latencyLine(d)}</div>`;
  } else {
    head = `<div class="p-head"><div class="top"><span class="pid">${esc(id)}</span><button class="close" type="button" data-close aria-label="Close pole panel">&times;</button></div>
      <div class="decision">${shapeSvg('pending')}<div class="what"><b style="font-size:18px">${p.inspected && d ? 'Awaiting decision' : 'Not inspected'}</b><span>${p.inspected && d ? 'Run the inspection to see what the model decides.' : 'This pole has no photo and no decision in this demo.'}</span></div></div></div>`;
  }
  const photo = p.inspected ? `<div class="sec">${photoHTML(p.photo, `Photo of pole ${id}`, creditLine(p.photo_source))}</div>` : '';
  const v1 = decided ? `<div class="sec"><h3>What the model reads in the photo<em>${passMs(d.v1) !== null ? `pass 1, ${ms1(passMs(d.v1))} ms` : 'pass 1'}</em></h3>${v1Block(d)}</div>` : '';
  const recSec = rec ? `<div class="sec"><h3>Asset record</h3>${kvBlock(rec)}${sparkline(rec)}</div>` : '';
  const v2 = decided ? `<div class="sec"><h3>Decision from photo and record<em>${passMs(d.v2) !== null ? `pass 2, ${ms1(passMs(d.v2))} ms` : 'pass 2'}</em></h3>
      <div class="qgroup"><div class="qname">Recommended action<span>${esc(cap(d.action))}</span></div>${actionBars(answerOf(d, 'v2', 'action'))}</div>
      ${safetyBar(d.v2)}
      </div>` : '';
  return head.replace('<div class="p-head">', '<div class="sec p-head">') + photo + v1 + recSec + v2;
}
function renderPanel() {
  const root = $('#panel');
  root.innerHTML = S.selected ? polePanel(S.selected) : emptyPanel();
  wirePhotos(root);
}

// ---------------------------------------------------------------- map
let map;
function iconFor(state) { return L.divIcon({ className: `pm ${state}`, html: shapeSvg(state), iconSize: [22, 22], iconAnchor: [11, 11] }); }
function initMap() {
  map = L.map('map', { preferCanvas: true, zoomSnap: 0.25, zoomControl: true, attributionControl: true });
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors',
  }).addTo(map);
  map.setView([-27.56, 151.95], 13);
  const canvas = L.canvas({ padding: 0.5 });
  for (const p of S.poles) {
    if (p.inspected && S.decisions[p.id]) {
      const m = L.marker([p.lat, p.lon], { icon: iconFor('pending'), keyboard: true, title: p.id, riseOnHover: true, alt: `Pole ${p.id}` });
      m.on('click', () => selectPole(p.id, false));
      m.addTo(map);
      S.markers.set(p.id, m);
    } else {
      L.circleMarker([p.lat, p.lon], { renderer: canvas, radius: 3.2, weight: 1, color: getCss('--ring'), opacity: .55, fillColor: getCss('--c-pending'), fillOpacity: .9 })
        .bindTooltip(`${p.id}: not inspected`, { direction: 'top' }).addTo(map);
    }
  }
  const zoomClass = () => map.getContainer().classList.toggle('zoom-low', map.getZoom() < 14.5);
  map.on('zoomend', zoomClass); zoomClass();
  // grey dots use theme colours at draw time; redraw if the theme flips
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => location.reload());
}
function fitMain() {
  const pts = S.poles.map((p) => [p.lat, p.lon]);
  if (pts.length) map.fitBounds(L.latLngBounds(pts), { padding: [40, 40], maxZoom: 15 });
}
function getCss(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888'; }
function setMarkerState(id, state, pop) {
  const m = S.markers.get(id);
  if (!m) return;
  const e = m.getElement();
  if (!e) return;
  e.classList.remove('pending', 'replace', 'maintain', 'defer', 'engineer', 'decided');
  e.classList.add(state);
  if (state !== 'pending') e.classList.add('decided');
  e.innerHTML = shapeSvg(state);
  m.setZIndexOffset(state === 'pending' ? 0 : 500);
  if (pop) { e.classList.add('pop'); setTimeout(() => e.classList.remove('pop'), 500); }
}
function selectPole(id, fly) {
  const prev = S.selected && S.markers.get(S.selected);
  if (prev && prev.getElement()) prev.getElement().classList.remove('sel');
  S.selected = id;
  const m = id && S.markers.get(id);
  if (m && m.getElement()) m.getElement().classList.add('sel');
  if (fly && m) map.flyTo(m.getLatLng(), Math.max(map.getZoom(), 15.5), { duration: 0.6 });
  R.dirty = true;
  renderPanel();
  $('#panel').scrollTop = 0;
}

// ---------------------------------------------------------------- replay engine
const DEFAULT_DELAY = 40;
const delayAt = (i) => { const d = S.decisions[S.order[i]]; const t = poleMs(d); return t === null || t <= 0 ? DEFAULT_DELAY : t; };
function insertSorted(arr, v) { let lo = 0, hi = arr.length; while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < v) lo = mid + 1; else hi = mid; } arr.splice(lo, 0, v); }
function median(arr) { const n = arr.length; if (!n) return null; return n % 2 ? arr[(n - 1) / 2] : (arr[n / 2 - 1] + arr[n / 2]) / 2; }

function decideAt(i, now) {
  const id = S.order[i];
  const d = S.decisions[id];
  const st = stateOf(d);
  S.decided.add(id);
  R.tally[st]++;
  const t = poleMs(d);
  if (t !== null) insertSorted(R.msSorted, t);
  R.stamps.push(now);
  R.feed.unshift({ id, state: st, action: d.action, ms: t });
  if (R.feed.length > 30) R.feed.length = 30;
  setMarkerState(id, st, true);
  if (S.selected === id) renderPanel();
  R.dirty = true;
}
function tick(ts) {
  if (!R.running) return;
  const dt = Math.min(250, ts - R.last);
  R.last = ts;
  R.activeMs += dt;
  const n = S.order.length;
  const mult = R.speed === 'video' ? 1 : Number(R.speed);
  R.credit += dt * mult;
  let guard = 400;
  while (R.idx < n && guard-- > 0) {
    const need = R.speed === 'video' ? 1000 / 40 : delayAt(R.idx);
    // 1x and 10x scale real time against the recorded model time; video is a fixed 40 per second.
    if (R.credit < need) break;
    R.credit -= need;
    decideAt(R.idx++, ts);
  }
  updateStats(ts);
  if (R.idx >= n) { R.running = false; updateButtons(); updateStats(ts, true); return; }
  R.raf = requestAnimationFrame(tick);
}
function run() {
  if (R.idx >= S.order.length) reset();
  if (!S.order.length) return;
  R.running = true;
  R.last = performance.now();
  updateButtons();
  cancelAnimationFrame(R.raf);
  R.raf = requestAnimationFrame(tick);
}
function pause() { R.running = false; cancelAnimationFrame(R.raf); updateButtons(); updateStats(performance.now(), true); }
function reset() {
  R.running = false; cancelAnimationFrame(R.raf);
  R.idx = 0; R.credit = 0; R.msSorted = []; R.stamps = []; R.activeMs = 0; R.feed = [];
  R.tally = { replace: 0, maintain: 0, defer: 0, engineer: 0 };
  S.decided.clear();
  for (const id of S.markers.keys()) setMarkerState(id, 'pending', false);
  R.dirty = true;
  updateButtons(); updateStats(performance.now(), true);
  if (S.selected) renderPanel();
  if (S.view === 'queue') renderQueue();
}
function showAll() {
  const wasRunning = R.running;
  R.running = false; cancelAnimationFrame(R.raf);
  const now = performance.now();
  while (R.idx < S.order.length) decideAt(R.idx++, now);
  R.stamps = [];
  updateButtons(); updateStats(now, true);
  if (S.view === 'queue') renderQueue();
  void wasRunning;
}

// ---------------------------------------------------------------- counters
function updateButtons() {
  const n = S.order.length, done = R.idx >= n && n > 0;
  const run = $('#runBtn');
  run.disabled = R.running || n === 0;
  $('span', run).textContent = R.running ? 'Running' : done ? 'Run again' : R.idx > 0 ? 'Resume' : 'Run inspection';
  $('#pauseBtn').disabled = !R.running;
  $('#resetBtn').disabled = R.idx === 0 && !R.running;
  $('#allBtn').disabled = done || n === 0;
}
function renderLegend() {
  const items = ['replace', 'maintain', 'defer', 'engineer', 'pending'];
  $('#legend').innerHTML = items.map((k) => `<div class="chip" data-k="${k}">${shapeSvg(k)}<div class="t"><b>${STATE_LABEL[k]}</b><span class="num" id="lg-${k}">0</span></div></div>`).join('');
}
function updateStats(now, force) {
  if (!R.dirty && !force) return;
  R.dirty = false;
  const n = S.order.length, d = S.decided.size;
  $('#sDecided').textContent = d;
  $('#sTotal').textContent = n;
  $('#sOf').textContent = `${S.poles.length} poles on the map`;
  $('#progressBar').style.width = n ? `${(d / n) * 100}%` : '0';
  const med = median(R.msSorted);
  $('#sMedian').textContent = med === null ? '-' : ms1(med);
  let rate = null, note = 'live, last second';
  if (R.running) {
    const cut = now - 1000;
    while (R.stamps.length && R.stamps[0] < cut) R.stamps.shift();
    rate = R.stamps.length ? R.stamps.length / Math.min(1, Math.max(R.activeMs, 250) / 1000) : 0;
  } else if (R.activeMs > 0 && d > 0) {
    rate = d / (R.activeMs / 1000); note = R.idx >= n ? 'average over the run' : 'paused, run average';
  } else note = d ? 'shown instantly' : 'press Run inspection';
  $('#sRate').textContent = rate === null ? '-' : rate >= 100 ? Math.round(rate) : rate.toFixed(1);
  $('#sRateNote').textContent = note;
  const q = R.tally.engineer;
  $('#sQueue').textContent = q;
  const badge = $('#queueBadge'); badge.textContent = q; badge.hidden = q === 0;
  for (const k of ['replace', 'maintain', 'defer', 'engineer']) $('#lg-' + k).textContent = R.tally[k];
  $('#lg-pending').textContent = S.poles.length - d;
  if (!S.selected && S.view === 'map') renderPanel();
}

// ---------------------------------------------------------------- flip view
function normFlip(f) {
  if (!f || typeof f !== 'object') return null;
  const hasDec = (o) => o && typeof o === 'object' && (o.v2 || o.action || o.decision || o.result);
  const side = (o) => {
    const dec = (o.v2 || o.action) ? o : (o.decision || o.result || o);
    const rec = o.record || dec.record || null;
    const d = JSON.parse(JSON.stringify(dec)); normaliseDecision(d);
    return { label: o.label || o.name || o.title || '', record: rec, d, v1: o.v1 || dec.v1 || null };
  };
  let a = null, b = null;
  const lists = [f.sides, f.variants, f.cases, f.records, f.decisions, f.items].find((x) => x && typeof x === 'object' && Object.keys(x).length >= 2 && Object.values(x).every((v) => v && typeof v === 'object'));
  if (lists) { const v = Array.isArray(lists) ? lists : Object.values(lists); a = v[0]; b = v[1]; }
  if (!a) {
    for (const [x, y] of [['a', 'b'], ['left', 'right'], ['low_risk', 'high_risk'], ['young', 'old'], ['before', 'after'], ['record_a', 'record_b'], ['one', 'two'], ['first', 'second']]) {
      if (f[x] && f[y]) { a = f[x]; b = f[y]; break; }
    }
  }
  if (!a) {
    const objs = Object.values(f).filter((v) => v && typeof v === 'object' && !Array.isArray(v) && v.record && hasDec(v));
    if (objs.length >= 2) { a = objs[0]; b = objs[1]; }
  }
  if (!a || !b) return null;
  const A = side(a), B = side(b);
  return {
    photo: f.photo || a.photo || null, photo_source: f.photo_source || a.photo_source || null,
    v1: f.v1 || A.v1 || B.v1 || null, explanation: f.explanation || f.note || f.summary || '',
    sides: [A, B], backend: f.backend,
  };
}
function flipExplanation(F) {
  if (F.explanation) return clean(F.explanation);
  const [a, b] = F.sides.map((s) => s.record || {});
  const bits = [];
  if (num(a.age_years) !== null && a.age_years !== b.age_years) bits.push(`${a.age_years} against ${b.age_years} years old`);
  if (num(a.shell_thickness_change_10y) !== null && a.shell_thickness_change_10y !== b.shell_thickness_change_10y) bits.push(`shell thickness ${signedPct(a.shell_thickness_change_10y)} against ${signedPct(b.shell_thickness_change_10y)} over ten years`);
  if (a.bushfire_zone !== b.bushfire_zone) bits.push(`${a.bushfire_zone} against ${b.bushfire_zone} bushfire zone`);
  if (a.critical_customer !== b.critical_customer) bits.push(`${a.critical_customer ? words(a.critical_customer_type) : 'no critical customer'} against ${b.critical_customer ? words(b.critical_customer_type) : 'no critical customer'}`);
  if (a.years_since_maintenance !== b.years_since_maintenance) bits.push(`maintained ${a.years_since_maintenance ?? 'never'} against ${b.years_since_maintenance ?? 'never'} years ago`);
  const [da, db] = F.sides.map((s) => s.d.action);
  return clean(`Same photo, different record: ${bits.slice(0, 4).join('; ')}. The model says ${da} for one and ${db} for the other.`);
}
function renderFlip() {
  const root = $('#flipPage');
  const F = S.flip;
  if (!F) {
    root.innerHTML = `<div class="page-head"><h2>Same photo, two records</h2></div><div class="empty"><p>No flip data found (${esc(S.base)}flip.json). Run the pipeline step that records the flip, or use the sample data.</p></div>`;
    return;
  }
  const [A, B] = F.sides;
  const card = (s, other, tag) => `<article class="card flip-card">
    <div class="sec"><div class="tag">${tag}</div><div class="lbl">${clean(s.label || '')}</div>${decisionHead(s.d)}
      <div class="badges" style="margin-top:8px"><span class="badge-pill ${stateOf(s.d) === 'engineer' ? 'engineer' : 'auto'}">${stateOf(s.d) === 'engineer' ? 'Engineer review' : 'Auto decision'}</span></div></div>
    <div class="sec"><h3>Asset record</h3>${kvBlock(s.record, other.record)}${sparkline(s.record)}</div>
    <div class="sec"><h3>Decision probabilities</h3><div class="qgroup">${actionBars(answerOf(s.d, 'v2', 'action'))}</div>${safetyBar(s.d.v2)}
      <p class="reason" style="margin-top:8px">${esc(routeReason(s.d))}</p></div></article>`;
  const v1d = F.v1 ? { v1: F.v1 } : A.v1 ? { v1: A.v1 } : null;
  root.innerHTML = `<div class="page-head"><h2>Same photo, two records, two decisions</h2><p>The photo findings are identical. Only the pole's record changes, and so does the recommendation.</p></div>
    <div class="explain" role="status">${shapeSvg(stateOf(A.d), 'sm')}<span>${flipExplanation(F)}</span>${shapeSvg(stateOf(B.d), 'sm')}</div>
    <div class="flip-grid">
      ${card(A, B, 'Record A')}
      <div class="card photo-col"><div>${photoHTML(F.photo, 'The photo used for both records', creditLine(F.photo_source))}</div>
        <div class="findings"><div class="h-sub" style="margin-bottom:6px">What the model reads, same for both</div>${v1d ? v1Block(v1d) : '<p>Same photo for both.</p>'}</div></div>
      ${card(B, A, 'Record B')}
    </div>`;
  wirePhotos(root);
}

// ---------------------------------------------------------------- queue view
function queueItems() {
  return S.order.filter((id) => S.decided.has(id) && stateOf(S.decisions[id]) === 'engineer')
    .map((id) => ({ id, d: S.decisions[id], p: S.byId.get(id) }))
    .sort((a, b) => a.d.confidence - b.d.confidence);
}
function renderQueue() {
  const root = $('#queuePage');
  const items = queueItems();
  const head = `<div class="page-head"><h2>Engineer queue</h2><p>The model knows when it does not know. A pole goes to an engineer when its top action is under ${pct(S.threshold)} confidence, when its lean cannot be assessed, or when the model asks for a reinspection. Lowest confidence first.</p></div>`;
  if (!items.length) {
    const none = S.decided.size === 0;
    root.innerHTML = head + `<div class="empty"><p>${none ? 'No decisions yet. The queue fills as the inspection runs.' : 'Nothing has been routed to an engineer so far.'}</p><div class="row">${none ? '<button class="btn primary" type="button" data-act="run">Run inspection</button><button class="btn" type="button" data-act="all">Show all results</button>' : ''}</div></div>`;
    return;
  }
  const thr = S.threshold * 100;
  root.innerHTML = head + `<div class="meta" style="margin-bottom:12px"><b class="num">${items.length}</b> poles waiting${S.decided.size < S.order.length ? `, ${S.decided.size} of ${S.order.length} decided so far` : ''}</div><div class="qgrid">` +
    items.map((it, i) => `<button class="qcard" type="button" data-pole="${esc(it.id)}">
      ${photoHTML(it.p && it.p.photo, `Photo of pole ${it.id}`, '').replace(/<div class="credit">.*$/, '')}
      <div><div class="rank">#${i + 1} lowest confidence</div><div class="id">${esc(it.id)}</div>
        <div class="lean-act">Model leans <b>${esc(it.d.action)}</b>, ${pct(it.d.confidence)}</div>
        <div class="confbar" role="img" aria-label="Confidence ${pct(it.d.confidence)}, threshold ${pct(S.threshold)}"><i style="width:${(it.d.confidence * 100).toFixed(1)}%"></i><u style="left:${thr}%"></u></div>
        <div class="why">${esc(routeReason(it.d))}</div></div></button>`).join('') + '</div>';
  wirePhotos(root);
}

// ---------------------------------------------------------------- accuracy view
const TASK_INFO = {
  lean: ['Lean', 'Photo only: straight, leaning or cannot assess.'],
  crossarm: ['Crossarm', 'Photo only: straight, tilted or not visible.'],
  vegetation: ['Vegetation', 'Photo only: clear, encroaching or not visible.'],
  record_health_index: ['Record health index', 'Record only: predicts the 5-level health index and is compared with the recorded one.'],
};
const pctile = (o) => {
  if (!o) return null;
  if (Array.isArray(o)) return { p50: o[0], p90: o[1], p99: o[2] };
  return { p50: o.p50 ?? o['50'], p90: o.p90 ?? o['90'], p99: o.p99 ?? o['99'] };
};
function normRel(r) {
  return (r || []).map((b) => {
    if (Array.isArray(b)) return { conf: b[0], acc: b[1], n: b[2] };
    const lo = b.bin_lo ?? b.lo ?? b.low, hi = b.bin_hi ?? b.hi ?? b.high;
    return {
      conf: b.mean_confidence ?? b.mean_conf ?? b.confidence ?? b.conf ?? (num(lo) !== null && num(hi) !== null ? (lo + hi) / 2 : null),
      acc: b.accuracy ?? b.acc, n: b.count ?? b.n ?? 0, lo, hi,
    };
  }).filter((b) => num(b.conf) !== null && num(b.acc) !== null && b.n > 0).sort((x, y) => x.conf - y.conf);
}
function normEval(e) {
  if (!e || typeof e !== 'object') return null;
  let t = e.tasks || e.per_task || e.results;
  if (!t || typeof t !== 'object') {
    t = {};
    for (const [k, v] of Object.entries(e)) if (v && typeof v === 'object' && !Array.isArray(v) && ('accuracy' in v || 'macro_f1' in v)) t[k] = v;
  }
  const tasks = Object.entries(t).map(([key, v]) => ({
    key, name: (TASK_INFO[key] || [cap(key)])[0], desc: (TASK_INFO[key] || [])[1] || '',
    n: v.n, accuracy: v.accuracy, macro_f1: v.macro_f1, ece: v.ece, confusion: v.confusion || null,
    rel: normRel(v.reliability), lat: pctile(v.latency_ms), model: pctile(v.model_ms),
    backend: v.backend, hardware: v.hardware, recorded_at: v.recorded_at,
  }));
  const first = tasks[0] || {};
  return {
    tasks, cost: e.cost || null, model: e.model,
    backend: e.backend || first.backend, hardware: e.hardware || first.hardware, recorded_at: e.recorded_at || first.recorded_at,
  };
}
let accTask = null;
function relChart(task) {
  const pts = task.rel;
  if (!pts.length) return '<p class="reason">No reliability bins recorded for this task.</p>';
  const W = 620, H = 440, L = 52, Rt = 16, T = 12, PH = 300, PW = W - L - Rt;
  const X = (c) => L + c * PW, Y = (a) => T + (1 - a) * PH;
  const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1];
  const grid = ticks.map((t) => `<line class="grid" x1="${L}" x2="${W - Rt}" y1="${Y(t)}" y2="${Y(t)}"/><text x="${L - 8}" y="${Y(t) + 4}" text-anchor="end">${Math.round(t * 100)}%</text><text x="${X(t)}" y="${T + PH + 18}" text-anchor="middle">${Math.round(t * 100)}%</text>`).join('');
  const ang = -Math.atan2(PH, PW) * 180 / Math.PI;
  const diag = `<line class="diag" x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}"/><text class="dl" transform="translate(${X(0.12)} ${Y(0.12) - 8}) rotate(${ang.toFixed(1)})">Perfect calibration</text>`;
  const line = pts.map((p, i) => `${i ? 'L' : 'M'}${X(p.conf).toFixed(1)} ${Y(p.acc).toFixed(1)}`).join('');
  const dots = pts.map((p) => `<circle class="pt" cx="${X(p.conf).toFixed(1)}" cy="${Y(p.acc).toFixed(1)}" r="5"/>`).join('');
  const hits = pts.map((p, i) => `<circle class="hit" data-i="${i}" cx="${X(p.conf).toFixed(1)}" cy="${Y(p.acc).toFixed(1)}" r="15"/>`).join('');
  const maxN = Math.max(...pts.map((p) => p.n));
  const HB = T + PH + 92, HH = 52;
  const hist = pts.map((p) => { const h = Math.max(2, (p.n / maxN) * HH); return `<rect class="hist" x="${(X(p.conf) - 11).toFixed(1)}" y="${(HB - h).toFixed(1)}" width="22" height="${h.toFixed(1)}" rx="4"/>`; }).join('');
  const last = pts[pts.length - 1];
  return `<div class="relwrap"><svg class="relchart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Reliability chart: accuracy against mean confidence for ${esc(task.name)}. Table below has the values.">
    ${grid}${diag}<path class="line" d="${line}"/>${dots}
    <text class="dl" x="${(X(last.conf) - 10).toFixed(1)}" y="${(Y(last.acc) + 22).toFixed(1)}" text-anchor="end">Matilda Jev</text>
    <text class="ttl" x="${L + PW / 2}" y="${T + PH + 38}" text-anchor="middle">Mean confidence in the bin</text>
    <text class="ttl" transform="translate(13 ${T + PH / 2}) rotate(-90)" text-anchor="middle">Accuracy</text>
    <line class="grid" x1="${L}" x2="${W - Rt}" y1="${HB}" y2="${HB}"/>${hist}
    <text class="ttl" x="${L - 8}" y="${HB - HH / 2 + 4}" text-anchor="end">Poles</text>${hits}</svg>
    <div class="tip" id="relTip" hidden></div></div>`;
}
function renderAccuracy() {
  const root = $('#accuracyPage');
  const E = S.evalS;
  if (!E || !E.tasks.length) {
    root.innerHTML = `<div class="page-head"><h2>Accuracy and calibration</h2></div><div class="empty"><p>No evaluation found (${esc(S.base)}eval_summary.json).</p></div>`;
    return;
  }
  if (!accTask || !E.tasks.find((t) => t.key === accTask)) accTask = E.tasks[0].key;
  const T = E.tasks.find((t) => t.key === accTask);
  const cards = E.tasks.map((t) => `<button class="task" type="button" data-task="${esc(t.key)}" aria-pressed="${t.key === accTask}">
    <span class="nm">${esc(t.name)}</span><span class="big num">${pct(t.accuracy)}<small>accuracy</small></span>
    <span class="row"><span>Macro F1 <b class="num">${num(t.macro_f1) === null ? '-' : t.macro_f1.toFixed(2)}</b></span><span>n <b class="num">${t.n ?? '-'}</b></span><span>ECE <b class="num">${num(t.ece) === null ? '-' : t.ece.toFixed(3)}</b></span></span>
    <span class="desc">${esc(t.desc)}</span></button>`).join('');
  const lat = (label, o) => o ? `<tr><td>${label}</td><td>${ms1(o.p50)}</td><td>${ms1(o.p90)}</td><td>${ms1(o.p99)}</td></tr>` : '';
  let conf = '';
  if (T.confusion) {
    const rows = Object.keys(T.confusion);
    const cols = Array.from(new Set(rows.flatMap((r) => Object.keys(T.confusion[r]))));
    conf = `<div class="card"><h3>Confusion, ${esc(T.name)}</h3><div class="sub">Rows are the true label, columns are the model's answer.</div>
      <table class="tbl"><thead><tr><th>true \\ model</th>${cols.map((c) => `<th>${esc(words(c))}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => {
        const tot = cols.reduce((a, c) => a + (T.confusion[r][c] || 0), 0) || 1;
        return `<tr><td>${esc(words(r))}</td>${cols.map((c) => { const v = T.confusion[r][c] || 0; return `<td class="cell" style="--o:${((v / tot) * 0.5).toFixed(2)}"><span ${c === r ? 'style="font-weight:500"' : ''}>${v}</span></td>`; }).join('')}</tr>`;
      }).join('')}</tbody></table></div>`;
  }
  const cost = E.cost ? `<div class="card"><h3>Cost of the recorded run</h3><div class="sub">From the recorded GPU run, not an estimate.</div><div class="costs">
      ${num(E.cost.gpu_hourly_usd) !== null ? `<div><b class="num">${usd(E.cost.gpu_hourly_usd)}</b><span>GPU per hour</span></div>` : ''}
      ${num(E.cost.decisions_per_hour) !== null ? `<div><b class="num">${Math.round(E.cost.decisions_per_hour).toLocaleString('en-AU')}</b><span>decisions per hour</span></div>` : ''}
      ${num(E.cost.usd_per_1000_poles) !== null ? `<div><b class="num">${usd(E.cost.usd_per_1000_poles)}</b><span>per 1,000 poles</span></div>` : ''}</div></div>` : '';
  const relRows = T.rel.map((b) => `<tr><td>${b.lo != null && b.hi != null ? `${Math.round(b.lo * 100)} to ${Math.round(b.hi * 100)}%` : ''}</td><td>${pct(b.conf)}</td><td>${pct(b.acc)}</td><td>${b.n}</td></tr>`).join('');
  root.innerHTML = `<div class="page-head"><h2>Accuracy and calibration</h2>
      <p>From one recorded evaluation run. Accuracy says how often the model is right. The reliability chart says whether its confidence can be trusted, which is what lets low-confidence cases go to an engineer.</p></div>
    <div class="meta" style="margin-bottom:14px"><span>Backend <b>${esc(E.backend || '-')}</b></span><span>Hardware <b>${clean(E.hardware || '-')}</b></span><span>Recorded <b>${esc(fmtDate(E.recorded_at))}</b></span>${E.model ? `<span>Model <b>${esc(E.model)}</b></span>` : ''}</div>
    <div class="tasks">${cards}</div>
    <div class="acc-grid">
      <div class="card"><h3>Reliability, ${esc(T.name)}</h3>
        <div class="sub">Each dot is a confidence bin. On the dashed line, confidence matches accuracy. Below it, the model is overconfident. ECE ${num(T.ece) === null ? '-' : T.ece.toFixed(3)} is the average gap.</div>
        ${relChart(T)}
        <details class="tview"><summary>Show as table</summary><table class="tbl"><thead><tr><th>Bin</th><th>Mean confidence</th><th>Accuracy</th><th>Poles</th></tr></thead><tbody>${relRows}</tbody></table></details></div>
      <div class="stack">
        <div class="card"><h3>Latency, ${esc(T.name)}</h3><div class="sub">Milliseconds per request. Model time is the server-reported compute; round trip adds client, network and queueing.</div>
          <table class="tbl"><thead><tr><th></th><th>p50</th><th>p90</th><th>p99</th></tr></thead><tbody>${lat('Model time', T.model)}${lat('Round trip', T.lat)}</tbody></table></div>
        ${cost}${conf}
      </div></div>`;
  wireRel(T);
}
function wireRel(T) {
  const svg = $('.relchart'), tip = $('#relTip');
  if (!svg || !tip) return;
  $$('.hit', svg).forEach((c) => {
    const show = () => {
      const b = T.rel[Number(c.dataset.i)];
      const sr = svg.getBoundingClientRect(), cr = c.getBoundingClientRect();
      const wrap = svg.parentElement.getBoundingClientRect();
      tip.innerHTML = `<b>${pct(b.conf)} confident</b>${pct(b.acc)} correct, ${b.n} poles`;
      tip.style.left = `${cr.left + cr.width / 2 - wrap.left}px`;
      tip.style.top = `${cr.top + cr.height / 2 - wrap.top - 4}px`;
      tip.hidden = false; void sr;
    };
    c.addEventListener('mouseenter', show);
    c.addEventListener('mouseleave', () => { tip.hidden = true; });
  });
}

// ---------------------------------------------------------------- live inspection (hero view)
// One pole at a time: photo, scan, v1 answers, record, decision. Honest by design: the model returns
// answers and probabilities only, so the "reading" effect is an abstract whole-frame sweep, never a box.
const LV = {
  items: [], idx: 0, playing: false, started: false, finished: false,
  mode: 'explain', t: 0, cur: null, sched: [], phase: 0, elapsed: 0, last: 0, raf: 0,
  msSorted: [], decidedCount: 0,
  mini: null, miniMarkers: new Map(), activeId: null, thumbs: [],
  api: false, presets: [], idleTimer: 0,
};
const EXPLAIN = { in: 600, read: 1300, v1: 900, record: 1000, decision: 900 };
// At real speed only the two model passes take time, so the decision gets a short on-screen pause
// and reveals get a minimum length. Neither changes the measured model times shown.
const DECISION_HOLD_MS = 600;
const REVEAL_MS = 300;
const clamp01 = (x) => Math.max(0, Math.min(1, x));
const rnd = Math.round;
const STATE_VAR = { replace: '--c-replace', maintain: '--c-maintain', defer: '--c-defer', engineer: '--c-engineer' };
const DEFAULT_LEVELS = ['critical', 'poor', 'fair', 'good', 'as new'];

function imgOk(url) {
  return new Promise((res) => { const i = new Image(); i.onload = () => res(true); i.onerror = () => res(false); i.src = url; setTimeout(() => res(false), 4000); });
}
// Opening order: a clear REPLACE (most visibly leaning), a clear DEFER (most clearly straight),
// the lowest-confidence engineer case, then everything else in decisions.json order.
async function liveOrder() {
  const info = S.order.map((id) => ({ id, d: S.decisions[id] }));
  const seen = new Map();
  const ok = async (id) => {
    if (seen.has(id)) return seen.get(id);
    const u = photoUrl(S.byId.get(id).photo);
    const r = u ? await imgOk(u) : false; seen.set(id, r); return r;
  };
  const chosen = [];
  const pick = async (filters, score) => {
    for (const f of filters) {
      const c = info.filter((x) => !chosen.includes(x.id) && f(x.d)).sort((a, b) => score(b.d) - score(a.d));
      for (const x of c.slice(0, 12)) if (await ok(x.id)) return x.id;
    }
    return null;
  };
  const prob = (d, k) => (answerOf(d, 'v1', 'lean') || { probabilities: {} }).probabilities[k] ?? 0;
  const add = (id) => { if (id) chosen.push(id); };
  add(await pick([(d) => d.action === 'replace' && stateOf(d) !== 'engineer', (d) => d.action === 'replace'], (d) => prob(d, 'leaning')));
  add(await pick([(d) => d.action === 'defer' && stateOf(d) !== 'engineer', (d) => d.action === 'defer'], (d) => prob(d, 'straight')));
  add(await pick([(d) => stateOf(d) === 'engineer'], (d) => -d.confidence));
  return chosen.concat(S.order.filter((id) => !chosen.includes(id)));
}

const levelOf = (h) => {
  if (!h) return null;
  const n = Object.keys(h.probabilities || {}).length || 5;
  const s = num(h.score);
  const idx = s === null ? -1 : Math.max(0, Math.min(n - 1, rnd(s)));
  const name = idx < 0 ? '-' : (h.legend && h.legend[String(idx)] ? shortLevel(h.legend, idx) : DEFAULT_LEVELS[idx] || String(idx));
  return { idx, n, name };
};

// ----- timing
function realDur(name, it) {
  const a = passMs(it.d && it.d.v1), b = passMs(it.d && it.d.v2);
  if (name === 'read') return a ?? 30;
  if (name === 'record') return b ?? 30;
  if (name === 'decision') return DECISION_HOLD_MS;
  return 4;
}
const phaseDur = (name, it, t) => {
  const e = EXPLAIN[name], r = Math.max(1, realDur(name, it));
  if (t <= 0) return e; if (t >= 1) return r;
  return Math.exp(Math.log(e) * (1 - t) + Math.log(r) * t);
};
function tFor(i, custom) {
  if (custom) return LV.mode === 'real' ? 1 : 0;
  if (LV.mode === 'explain') return 0;
  if (LV.mode === 'real') return 1;
  const k = (i - 3) / 12;
  return k <= 0 ? 0 : k >= 1 ? 1 : k * k * (3 - 2 * k);
}
function buildSched(it, t) {
  const names = ['in', 'read'];
  if (!it.waiting) { names.push('v1'); if (it.d && it.d.v2 && it.d.action) names.push('record'); names.push('decision'); }
  let at = 0;
  return names.map((name) => { const dur = phaseDur(name, it, t); const o = { name, dur, start: at }; at += dur; return o; });
}

// ----- DOM helpers
const el$ = {};
function liveEls() {
  for (const id of ['stage', 'stageImg', 'scan', 'modeChip', 'stageHint', 'capLeft', 'capRight', 'lvTimer', 'cardV1', 'cardRec', 'cardDec', 'lvPlay', 'filmTrack', 'film', 'tryBtn', 'tryPanel', 'tryMsg', 'tryFile', 'tryRecord', 'drop', 'stepper']) el$[id] = document.getElementById(id);
}
function lbar(label, p, top, anim, colorVar, raw) {
  const w = (clamp01(p || 0) * 100).toFixed(1);
  const st = colorVar ? ` style="--ac:var(${colorVar})"` : '';
  return `<div class="bar${top ? ' top' : ''}"${st} title="${esc(label)}: ${pct1(p)}"><span class="lb">${esc(raw ? label : cap(label))}</span><span class="tr"><span class="fl" data-w="${w}" style="width:${anim ? 0 : w}%"></span></span><span class="pc">${pct(p)}</span></div>`;
}
function qbars(ans, anim) {
  const e = Object.entries((ans && ans.probabilities) || {});
  const max = Math.max(...e.map(([, p]) => p), 0);
  return `<div class="bars live-bars">${e.map(([k, p]) => lbar(k, p, p === max, anim)).join('')}</div>`;
}
function flush(root, markers) {
  void root.offsetWidth;
  $$('[data-w]', root).forEach((f) => { f.style.width = f.dataset.w + '%'; });
  (markers || []).forEach((m) => { m.style.left = m.dataset.pos + '%'; });
}
const animOf = (dur) => (dur <= 0 ? 0 : Math.round(Math.min(Math.max(dur, REVEAL_MS), 900)));
function setStep(name) {
  const order = { in: 0, read: 1, v1: 1, record: 2, decision: 3 };
  const cur = order[name];
  $$('#stepper li').forEach((li) => {
    const s = Number(li.dataset.step);
    li.classList.toggle('done', s < cur || (name === 'v1' && s === 1));
    li.classList.toggle('active', s === cur && name !== 'v1');
  });
}
const setTimerText = (s) => { el$.lvTimer.textContent = s || ''; };

function clearCards() {
  el$.cardV1.innerHTML = '<h3>What the model reads in the photo</h3><p class="wait">Waiting for the photo read.</p>';
  el$.cardRec.innerHTML = '<h3>Asset record</h3><p class="wait">Checked after the photo.</p>';
  el$.cardDec.innerHTML = '<h3>Decision</h3><p class="wait">Photo and record together decide the action.</p>';
}
function stateNow(it) { return it && it.d ? stateOf(it.d) : 'pending'; }

function showStage(it, dur) {
  const stage = el$.stage, img = el$.stageImg;
  stage.classList.remove('reading', 'missing');
  img.classList.remove('slide');
  const u = it.custom ? it.photo : photoUrl(it.p && it.p.photo);
  if (!u) stage.classList.add('missing');
  else { img.onerror = () => stage.classList.add('missing'); img.src = u; }
  img.alt = `Photo of pole ${it.id}`;
  const a = animOf(dur);
  stage.style.setProperty('--anim', a + 'ms');
  if (a) { void img.offsetWidth; img.classList.add('slide'); }
  const src = it.custom ? null : (it.p && it.p.photo_source);
  const credit = it.custom ? 'Your photo. Sent to the model for this one decision, not saved by this demo' : (src && src.credit ? clean(src.credit) : 'PD-Defect, CC BY 4.0');
  el$.capLeft.innerHTML = `<code>${esc(it.id)}</code> &middot; ${esc(S.meta.town || 'Toowoomba, QLD')}`;
  el$.capRight.innerHTML = `${it.custom ? '' : 'Photo: '}${credit}`;
  el$.scan.style.top = '-24%';
}
function showV1(it, dur) {
  const d = it.d, a = animOf(dur), card = el$.cardV1;
  card.style.setProperty('--anim', a + 'ms');
  const cols = [['lean', 'Lean'], ['crossarm', 'Crossarm'], ['vegetation', 'Vegetation']].map(([k, name]) => {
    const ans = answerOf(d, 'v1', k);
    return ans ? `<div><div class="qname">${name}<span>${esc(cap(ans.choice))}</span></div>${qbars(ans, a)}</div>` : '';
  }).join('');
  const h = answerOf(d, 'v1', 'health');
  let health = '';
  if (h) {
    const lv = levelOf(h), n = lv.n, legend = h.legend || {};
    const score = num(h.score);
    const pos = score === null ? 0 : ((score + 0.5) / n) * 100;
    const segs = Array.from({ length: n }, (_, k) => `<span class="hs-seg" style="--a:${Math.round(18 + (72 * k) / Math.max(n - 1, 1))}%"></span>`).join('');
    const labs = Array.from({ length: n }, (_, k) => `<span class="${k === lv.idx ? 'on' : ''}">${esc(h.legend && h.legend[String(k)] ? shortLevel(legend, k) : DEFAULT_LEVELS[k] || k)}</span>`).join('');
    health = `<div class="health-row"><div class="lvl"><b>${esc(cap(lv.name))}</b><span>Health, 5-level scale</span></div>
      <div class="healthscale" role="img" aria-label="Health: ${esc(lv.name)}"><div class="hs-row">${segs}<span class="hs-marker" data-pos="${pos.toFixed(1)}" style="left:${a ? 0 : pos.toFixed(1)}%"></span></div><div class="hs-labels">${labs}</div></div></div>`;
  }
  const ms = passMs(d.v1);
  card.innerHTML = `<h3>What the model reads in the photo<em>${ms !== null ? rnd(ms) + ' ms' : ''}</em></h3><div class="v1cols">${cols}</div>${health}`;
  if (a) flush(card, $$('.hs-marker', card));
}
function showRecord(it, dur) {
  const card = el$.cardRec, rec = it.rec, a = animOf(dur);
  card.style.setProperty('--anim', a + 'ms');
  card.classList.remove('slide');
  if (!rec) { card.innerHTML = '<h3>Asset record</h3><p class="wait">No record paired with this photo. The answer comes from the photo alone.</p>'; return; }
  const keep = ['age', 'maint', 'bush', 'cust', 'crit'];
  const rows = recordRows(rec).filter((r) => keep.includes(r[0])).map(([, l, v]) => `<div><dt>${l}</dt><dd>${clean(v)}</dd></div>`).join('');
  const ms = passMs(it.d && it.d.v2);
  card.innerHTML = `<h3>Checking the asset record<em>${ms !== null ? rnd(ms) + ' ms' : ''}</em></h3><div class="rec-body"><dl class="kv">${rows}</dl>${sparkline(rec)}</div>`;
  if (a) { void card.offsetWidth; card.classList.add('slide'); }
}
function showDecision(it) {
  const d = it.d, card = el$.cardDec, a = passMs(d.v1), b = passMs(d.v2);
  card.style.setProperty('--anim', Math.min(LV.t >= 1 ? REVEAL_MS : 500, animOf(LV.sched[LV.phase] ? LV.sched[LV.phase].dur : 0)) + 'ms');
  const times = [a !== null ? `Photo read: ${rnd(a)} ms` : '', b !== null ? `Decision: ${rnd(b)} ms` : ''].filter(Boolean).join(', ');
  if (!d.action) {
    const lv = levelOf(answerOf(d, 'v1', 'health'));
    card.innerHTML = `<h3>Photo-only result</h3><div class="dec-body"><div class="stamp neutral">${shapeSvg('pending')}<div><b>${lv ? esc(cap(lv.name)) : 'Read'}</b><span>health from the photo</span></div></div><div class="dec-side">${times ? `<div class="times">${esc(times)}</div>` : ''}<p class="reason">Pair a record to get an action.</p></div></div>`;
    return;
  }
  const st = stateOf(d);
  const label = d.route === 'engineer' ? 'Engineer review' : d.action;
  const sub = d.route === 'engineer' ? `Model leans ${d.action}, ${pct(d.confidence)}` : `${pct(d.confidence)} confident`;
  const s = answerOf(d, 'v2', 'safety_risk_now');
  const safety = s && num(s.noul) !== null ? `<div class="qgroup"><div class="bars live-bars">${lbar('Public safety risk', s.noul, true, 0, null, true)}</div></div>` : '';
  const tot = poleMs(d);
  card.innerHTML = `<h3>Decision</h3><div class="dec-body"><div class="stamp" style="--ca:var(${STATE_VAR[st]})">${shapeSvg(st)}<div><b>${esc(label)}</b><span>${esc(sub)}</span></div></div>
    <div class="dec-side">${safety}${times ? `<div class="times">${esc(times)}<small>${tot !== null ? `Model time for this pole: ${rnd(tot)} ms` : ''}</small></div>` : ''}</div></div>`;
}

// ----- mini map
function initMini() {
  LV.mini = L.map('miniMap', { zoomControl: false, dragging: false, scrollWheelZoom: false, doubleClickZoom: false, boxZoom: false, keyboard: false, touchZoom: false, attributionControl: true });
  LV.mini.attributionControl.setPrefix(false);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>' }).addTo(LV.mini);
  const pts = S.poles.map((p) => [p.lat, p.lon]);
  if (pts.length) LV.mini.fitBounds(L.latLngBounds(pts), { padding: [14, 14] }); else LV.mini.setView([-27.56, 151.95], 12);
  const canvas = L.canvas({ padding: 0.5 });
  const live = new Set(LV.items.map((i) => i.id));
  for (const p of S.poles) {
    if (live.has(p.id)) {
      const m = L.marker([p.lat, p.lon], { icon: iconFor('pending'), interactive: false, keyboard: false });
      m.addTo(LV.mini); LV.miniMarkers.set(p.id, m);
    } else {
      L.circleMarker([p.lat, p.lon], { renderer: canvas, radius: 2, weight: 0, fillColor: getCss('--c-pending'), fillOpacity: .7, interactive: false }).addTo(LV.mini);
    }
  }
}
function miniState(id, state, pop) {
  const m = LV.miniMarkers.get(id); if (!m || !m.getElement()) return;
  const e = m.getElement();
  e.classList.remove('pending', 'replace', 'maintain', 'defer', 'engineer', 'decided');
  e.classList.add(state); if (state !== 'pending') e.classList.add('decided');
  e.innerHTML = shapeSvg(state);
  m.setZIndexOffset(state === 'pending' ? 0 : 400);
  if (pop) { e.classList.add('pop'); setTimeout(() => e.classList.remove('pop'), 500); }
}
function miniActive(id) {
  const prev = LV.activeId && LV.miniMarkers.get(LV.activeId);
  if (prev && prev.getElement()) { prev.getElement().classList.remove('active'); prev.setZIndexOffset(prev.getElement().classList.contains('decided') ? 400 : 0); }
  LV.activeId = id;
  const m = id && LV.miniMarkers.get(id);
  if (m && m.getElement()) { m.getElement().classList.add('active'); m.setZIndexOffset(900); }
}

// ----- filmstrip
function buildFilm() {
  const tr = el$.filmTrack; tr.innerHTML = '';
  LV.thumbs = LV.items.map((it) => {
    const d = document.createElement('div'); d.className = 'th';
    const u = photoUrl(it.p.photo);
    d.innerHTML = `${u ? `<img loading="lazy" decoding="async" alt="" src="${esc(u)}">` : ''}<span class="bd"></span>`;
    const im = $('img', d); if (im) im.addEventListener('error', () => im.classList.add('bad'), { once: true });
    tr.appendChild(d); return d;
  });
}
function filmUpdate() {
  LV.thumbs.forEach((th, i) => {
    const cur = i === LV.idx && !(LV.cur && LV.cur.custom);
    th.classList.toggle('cur', cur);
    if (i !== LV.idx || !cur) { /* done flag set in markDone */ }
  });
  const step = 158;
  el$.filmTrack.style.setProperty('--film', LV.t < 0.5 ? '350ms' : '0ms');
  el$.filmTrack.style.transform = `translateX(${-Math.max(0, LV.idx - 1) * step}px)`;
}
function preload(from) {
  for (let i = from; i < Math.min(LV.items.length, from + 8); i++) { const u = photoUrl(LV.items[i].p.photo); if (u && !LV.items[i].pre) { LV.items[i].pre = new Image(); LV.items[i].pre.src = u; } }
}

// ----- counters
function liveCounters() {
  $('#lvDecided').textContent = LV.decidedCount;
  $('#lvTotal').textContent = LV.items.length;
  const med = median(LV.msSorted);
  $('#lvMedian').textContent = med === null ? '-' : rnd(med);
  // Model throughput, so on-screen pauses (Explain pacing, the decision hold) never lower it.
  $('#lvRate').textContent = med === null ? '-' : (1000 / med).toFixed(1);
}
function markDone(it) {
  if (it.counted) return;
  it.counted = true;
  if (it.custom) return;
  LV.decidedCount++;
  const t = poleMs(it.d);
  if (t !== null) insertSorted(LV.msSorted, t);
  const st = stateOf(it.d);
  miniState(it.id, st, true);
  const th = LV.thumbs[LV.items.indexOf(it)];
  if (th) { $('.bd', th).innerHTML = shapeSvg(st); th.classList.add('done'); }
  liveCounters();
}
function setModeChip(it) {
  const chip = el$.modeChip, t = LV.t, tot = it.d ? poleMs(it.d) : null;
  const n = tot !== null ? ` Real time per pole: ${rnd(tot)} ms` : '';
  chip.hidden = false;
  if (t <= 0) chip.textContent = 'Slowed down so you can see it.' + n;
  else if (t < 1) chip.textContent = 'Speeding up toward real speed.' + n;
  else chip.textContent = tot !== null ? `Real speed. This pole took ${rnd(tot)} ms, then a short pause to show the decision.` : 'Real speed, with a short pause to show each decision.';
  document.body.classList.toggle('live-fast', t > 0.8);
}

// ----- state machine
function enter(i) {
  const ph = LV.sched[i], it = LV.cur;
  setStep(ph.name);
  const a = it.d ? passMs(it.d.v1) : null, b = it.d ? passMs(it.d.v2) : null;
  switch (ph.name) {
    case 'in':
      showStage(it, ph.dur);
      clearCards();
      if (!it.custom) { miniActive(it.id); filmUpdate(); preload(LV.idx + 1); } else miniActive(null);
      setTimerText('');
      break;
    case 'read':
      el$.stage.classList.add('reading');
      break;
    case 'v1':
      el$.stage.classList.remove('reading');
      el$.scan.style.top = '-24%';
      showV1(it, ph.dur);
      setTimerText(a !== null ? `Photo read: ${rnd(a)} ms` : '');
      if (!it.d.action || !it.d.v2) { if (!it.d.v2) showRecord(it, 0); }
      break;
    case 'record':
      showRecord(it, ph.dur);
      break;
    case 'decision':
      if (it.d.v2 || it.d.action) { if (!LV.sched.some((s) => s.name === 'record')) showRecord(it, 0); }
      showDecision(it);
      markDone(it);
      setTimerText(b !== null ? `Decision: ${rnd(b)} ms` : (a !== null ? `Photo read: ${rnd(a)} ms` : ''));
      break;
    default: break;
  }
}
function visuals() {
  const it = LV.cur, ph = LV.sched[LV.phase];
  if (!it || !ph) return;
  const p = ph.dur > 0 ? clamp01((LV.elapsed - ph.start) / ph.dur) : 1;
  if (ph.name === 'read') {
    if (it.waiting) {
      const y = ((it.waitMs || 0) / 900) % 1;
      el$.scan.style.top = (-24 + y * 124) + '%';
      setTimerText(`Waiting for the model ${rnd(it.waitMs || 0)} ms`);
    } else {
      const passes = Math.max(1, rnd(ph.dur / 650));
      const y = p >= 1 ? 1 : (p * passes) % 1;
      el$.scan.style.top = (-24 + y * 124) + '%';
      const a = passMs(it.d.v1);
      setTimerText(a !== null ? `Reading photo ${rnd(a * p)} ms` : 'Reading photo');
    }
  } else if (ph.name === 'record') {
    const b = passMs(it.d.v2);
    setTimerText(b !== null ? `Checking the record ${rnd(b * p)} ms` : 'Checking the record');
  }
}
function setCur(it, carry) {
  LV.cur = it;
  LV.t = tFor(LV.idx, it.custom);
  LV.sched = buildSched(it, LV.t);
  LV.phase = 0; LV.elapsed = carry;
  setModeChip(it);
  enter(0);
}
function startPole(i, carry) {
  const it = LV.items[i];
  if (!it) { finishAll(); return; }
  LV.idx = i;
  setCur(it, carry || 0);
}
function finishAll() {
  LV.playing = false; LV.finished = true;
  cancelAnimationFrame(LV.raf);
  playLabel();
  el$.modeChip.hidden = false;
  el$.modeChip.textContent = `Inspection complete: ${LV.decidedCount} poles. Press R to restart.`;
  document.body.classList.remove('hide-cursor');
}
function poleDone(left) {
  const it = LV.cur;
  if (it.custom) { LV.playing = false; cancelAnimationFrame(LV.raf); playLabel(); el$.tryMsg.textContent = 'Done. Drop another photo, or press Play to carry on with the queue.'; return false; }
  if (LV.idx + 1 >= LV.items.length) { finishAll(); return false; }
  LV.idx++;
  setCur(LV.items[LV.idx], left);
  return true;
}
function advance(dt) {
  const it = LV.cur; if (!it) return;
  let ph = LV.sched[LV.phase];
  if (it.waiting && ph && ph.name === 'read') { it.waitMs = (it.waitMs || 0) + dt; visuals(); return; }
  LV.elapsed += dt;
  let guard = 400;
  while (guard-- > 0) {
    ph = LV.sched[LV.phase];
    if (LV.elapsed < ph.start + ph.dur) break;
    if (LV.phase + 1 < LV.sched.length) { LV.phase++; enter(LV.phase); continue; }
    if (it.waiting) return;
    if (!poleDone(LV.elapsed - (ph.start + ph.dur))) return;
    // a new pole started inside poleDone; keep consuming carried time
    const cur = LV.cur; if (cur !== it) { return advanceCarry(); }
  }
  visuals();
}
function advanceCarry() {
  // after a pole boundary: consume the carried elapsed time against the new schedule
  let guard = 400;
  while (guard-- > 0) {
    const it = LV.cur, ph = LV.sched[LV.phase];
    if (LV.elapsed < ph.start + ph.dur) break;
    if (LV.phase + 1 < LV.sched.length) { LV.phase++; enter(LV.phase); continue; }
    if (it.waiting) return;
    if (!poleDone(LV.elapsed - (ph.start + ph.dur))) return;
  }
  visuals();
}
function liveTick(ts) {
  if (!LV.playing) return;
  const dt = Math.min(100, ts - LV.last); LV.last = ts;
  advance(dt);
  if (LV.playing) LV.raf = requestAnimationFrame(liveTick);
}
function playLabel() {
  const b = el$.lvPlay;
  $('span', b).textContent = LV.playing ? 'Pause' : LV.finished ? 'Restart' : LV.started ? 'Resume' : 'Play';
  $('svg', b).innerHTML = LV.playing ? '<path d="M3 2h3v10H3zM8 2h3v10H8z"/>' : '<path d="M3 1.5v11l9-5.5z"/>';
}
function livePlay() {
  if (LV.finished) { liveRestart(true); return; }
  if (LV.playing) return;
  LV.playing = true; LV.last = performance.now();
  el$.stageHint.hidden = true;
  if (!LV.started) { LV.started = true; startPole(0, 0); }
  playLabel(); armCursor();
  cancelAnimationFrame(LV.raf); LV.raf = requestAnimationFrame(liveTick);
}
function livePause() { LV.playing = false; cancelAnimationFrame(LV.raf); playLabel(); document.body.classList.remove('hide-cursor'); }
const liveToggle = () => (LV.playing ? livePause() : livePlay());
function showComplete(it) {
  // Render every phase of a pole at once (used by Next while paused).
  const keep = LV.sched;
  for (let i = 0; i < keep.length; i++) { LV.phase = i; if (keep[i].name === 'read') continue; enter(i); }
  LV.phase = keep.length - 1; LV.elapsed = keep[keep.length - 1].start + keep[keep.length - 1].dur;
  el$.stage.classList.remove('reading');
  void it;
}
function liveNext() {
  if (LV.finished) return;
  if (LV.cur && LV.cur.custom) { exitCustom(); return; }
  if (!LV.started) { LV.started = true; el$.stageHint.hidden = true; }
  if (LV.cur && !LV.cur.counted && LV.started && LV.sched.length) { const t = LV.t; LV.t = Math.max(LV.t, 1); showComplete(LV.cur); LV.t = t; }
  const i = LV.idx + 1;
  if (i >= LV.items.length) { finishAll(); return; }
  startPole(i, 0);
  if (!LV.playing) { LV.t = 1; showComplete(LV.cur); }
}
function liveRestart(andPlay) {
  const was = LV.playing || andPlay === true;
  cancelAnimationFrame(LV.raf);
  LV.playing = false; LV.finished = false; LV.started = false;
  LV.msSorted = []; LV.decidedCount = 0;
  LV.items.forEach((it) => { it.counted = false; miniState(it.id, 'pending', false); });
  LV.thumbs.forEach((th) => { th.classList.remove('done', 'cur'); $('.bd', th).innerHTML = ''; });
  miniActive(null); liveCounters();
  clearCards(); setTimerText('');
  LV.idx = 0; LV.cur = null;
  idleFirst();
  if (was) livePlay(); else playLabel();
}
function idleFirst() {
  const it = LV.items[0]; if (!it) return;
  LV.idx = 0; LV.cur = it; LV.t = tFor(0);
  LV.sched = buildSched(it, LV.t); LV.phase = 0; LV.elapsed = 0;
  showStage(it, 0); setStep('in'); miniActive(it.id); filmUpdate(); preload(1); setModeChip(it);
  el$.stageHint.hidden = false;
}
function setMode(m) {
  if (!['explain', 'real', 'ramp'].includes(m)) return;
  LV.mode = m;
  $$('#lvMode button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.mode === m)));
  if (LV.cur && LV.started && !LV.finished && !(LV.cur.waiting)) {
    const carry = 0;
    if (LV.cur.counted) { LV.t = tFor(LV.idx, LV.cur.custom); setModeChip(LV.cur); return; }
    setCur(LV.cur, carry);
  } else if (LV.cur) { LV.t = tFor(LV.idx, LV.cur.custom); LV.sched = buildSched(LV.cur, LV.t); setModeChip(LV.cur); }
}
function armCursor() {
  clearTimeout(LV.idleTimer);
  document.body.classList.remove('hide-cursor');
  if (LV.playing && S.view === 'live') LV.idleTimer = setTimeout(() => { if (LV.playing && S.view === 'live') document.body.classList.add('hide-cursor'); }, 2000);
}

// ----- try a photo (only when the Python proxy answers api/health)
async function detectApi() {
  try {
    const c = new AbortController(); const to = setTimeout(() => c.abort(), 2000);
    const r = await fetch('api/health', { signal: c.signal, cache: 'no-store' }); clearTimeout(to);
    if (!r.ok) return false;
    const j = await r.json();
    return !!(j && j.status === 'ready');
  } catch { return false; }
}
function buildPresets() {
  const out = [];
  if (S.flip) S.flip.sides.forEach((s, i) => { if (s.record) out.push({ label: clean(s.label || `Flip record ${'AB'[i]}`).replace(/&amp;/g, '&'), record: s.record }); });
  const withRec = S.poles.filter((p) => p.inspected && p.record).sort((a, b) => (a.record.age_years || 0) - (b.record.age_years || 0));
  if (withRec.length) {
    const p = withRec[Math.floor(withRec.length / 2)];
    out.push({ label: `Pole ${p.id}, ${p.record.age_years} years old`, record: p.record });
    if (out.length < 3 && withRec.length > 1) { const q = withRec[withRec.length - 1]; out.push({ label: `Pole ${q.id}, ${q.record.age_years} years old`, record: q.record }); }
  }
  return out.slice(0, 3);
}
async function toDataURL(file) {
  let bmp;
  try { bmp = await createImageBitmap(file, { imageOrientation: 'from-image' }); }
  catch { bmp = await new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = URL.createObjectURL(file); }); }
  const w = bmp.width || bmp.naturalWidth, h = bmp.height || bmp.naturalHeight;
  const sc = Math.min(1, 1024 / Math.max(w, h));
  const c = document.createElement('canvas'); c.width = Math.max(1, rnd(w * sc)); c.height = Math.max(1, rnd(h * sc));
  c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height);
  return c.toDataURL('image/jpeg', 0.85);
}
function exitCustom() {
  LV.playing = false; cancelAnimationFrame(LV.raf);
  const i = LV.idx;
  LV.cur = null; startPole(i, 0); playLabel();
}
async function runCustom(file) {
  const msg = el$.tryMsg;
  if (!file || !/^image\//.test(file.type)) { msg.textContent = 'Please choose an image file.'; return; }
  el$.tryPanel.hidden = true; el$.tryBtn.setAttribute('aria-expanded', 'false');
  LV.thumbs.forEach((th) => th.classList.remove('cur'));
  msg.textContent = 'Resizing the photo...';
  let dataUrl;
  try { dataUrl = await toDataURL(file); } catch { msg.textContent = 'That image could not be read.'; return; }
  const sel = Number(el$.tryRecord.value);
  const rec = sel >= 0 ? LV.presets[sel].record : null;
  const it = { id: 'your-photo', custom: true, photo: dataUrl, rec, p: { record: rec }, d: null, waiting: true, waitMs: 0 };
  cancelAnimationFrame(LV.raf);
  LV.started = true; LV.finished = false; LV.playing = true; LV.last = performance.now();
  el$.stageHint.hidden = true;
  LV.cur = it; LV.t = tFor(LV.idx, true); LV.sched = buildSched(it, LV.t); LV.phase = 0; LV.elapsed = 0;
  setModeChip({ d: null }); el$.modeChip.hidden = true;
  enter(0); playLabel(); armCursor();
  LV.raf = requestAnimationFrame(liveTick);
  msg.textContent = 'Asking the model...';
  try {
    const r = await fetch('api/decide', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image: dataUrl, record: rec }) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    normaliseDecision(d);
    it.d = d; it.waiting = false;
    LV.t = tFor(LV.idx, true);
    LV.sched = buildSched(it, LV.t);
    if (LV.phase >= 1) { LV.phase = 1; LV.elapsed = LV.sched[1].start; }
    setModeChip(it);
    msg.textContent = 'Got the model answer.';
  } catch (e) {
    msg.textContent = `The model did not answer (${e.message}). Check the proxy and try again.`;
    el$.tryPanel.hidden = false; el$.tryBtn.setAttribute('aria-expanded', 'true');
    exitCustom();
  }
}
function bindTry() {
  el$.tryBtn.addEventListener('click', () => { const h = !el$.tryPanel.hidden; el$.tryPanel.hidden = h; el$.tryBtn.setAttribute('aria-expanded', String(!h)); if (!h && LV.playing) livePause(); });
  const drop = el$.drop;
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => { const f = e.dataTransfer && e.dataTransfer.files[0]; if (f) runCustom(f); });
  el$.tryFile.addEventListener('change', () => { const f = el$.tryFile.files[0]; if (f) runCustom(f); el$.tryFile.value = ''; });
}

// ----- init + events
async function initLive() {
  liveEls();
  const ids = await liveOrder();
  LV.items = ids.map((id) => ({ id, d: S.decisions[id], p: S.byId.get(id), rec: (S.byId.get(id) || {}).record || null }));
  if (!LV.items.length) { $('#stageHint').innerHTML = '<b>No decisions to show</b><span>decisions.json is empty.</span>'; return; }
  buildFilm(); initMini(); liveCounters(); clearCards(); idleFirst(); playLabel();
  $('#lvPlay').addEventListener('click', liveToggle);
  $('#lvNext').addEventListener('click', liveNext);
  $('#lvRestart').addEventListener('click', () => liveRestart(false));
  $('#lvMode').addEventListener('click', (e) => { const b = e.target.closest('button[data-mode]'); if (b) setMode(b.dataset.mode); });
  document.addEventListener('mousemove', armCursor);
  document.addEventListener('keydown', (e) => {
    if (S.view !== 'live' || e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'select' || tag === 'textarea') return;
    const k = e.key;
    if (k === ' ' || k === 'Spacebar') { e.preventDefault(); liveToggle(); }
    else if (k === 'ArrowRight') { e.preventDefault(); liveNext(); }
    else if (k === 'r' || k === 'R') liveRestart(false);
    else if (k === '1') setMode('explain'); else if (k === '2') setMode('real'); else if (k === '3') setMode('ramp');
  });
  LV.api = await detectApi();
  if (LV.api) {
    LV.presets = buildPresets();
    el$.tryRecord.innerHTML = '<option value="-1">No record (photo only)</option>' + LV.presets.map((p, i) => `<option value="${i}">${esc(p.label)}</option>`).join('');
    el$.tryBtn.hidden = false; bindTry();
  }
  const q = new URLSearchParams(location.search);
  if (['explain', 'real', 'ramp'].includes(q.get('mode'))) setMode(q.get('mode'));
  if (q.get('autoplay') === '1' && S.view === 'live') livePlay();
}

// ---------------------------------------------------------------- views & events
function setView(v, replace) {
  S.view = v;
  $$('.tab').forEach((t) => t.setAttribute('aria-selected', String(t.dataset.view === v)));
  $$('.view').forEach((s) => s.classList.toggle('active', s.id === 'view-' + v));
  if (v !== 'live' && LV.playing) livePause();
  if (v === 'live') setTimeout(() => { if (LV.mini) { LV.mini.invalidateSize(); const pts = S.poles.map((p) => [p.lat, p.lon]); if (pts.length) LV.mini.fitBounds(L.latLngBounds(pts), { padding: [14, 14] }); } }, 0);
  if (v === 'map') setTimeout(() => { if (!S.mapFitted) { map.invalidateSize(); fitMain(); S.mapFitted = true; } else map.invalidateSize(); }, 0);
  if (v === 'flip') renderFlip();
  if (v === 'queue') renderQueue();
  if (v === 'accuracy') renderAccuracy();
  if (v === 'map') { R.dirty = true; updateStats(performance.now(), true); if (S.selected) renderPanel(); }
  if (!replace) history.replaceState(null, '', '#' + v);
}
function bindEvents() {
  $$('.tab').forEach((t) => t.addEventListener('click', () => setView(t.dataset.view)));
  $('#runBtn').addEventListener('click', run);
  $('#pauseBtn').addEventListener('click', pause);
  $('#resetBtn').addEventListener('click', reset);
  $('#allBtn').addEventListener('click', showAll);
  $('#speedSeg').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-speed]'); if (!b) return;
    R.speed = b.dataset.speed; R.credit = 0;
    $$('#speedSeg button').forEach((x) => x.setAttribute('aria-checked', String(x === b)));
  });
  const openPole = (id) => { setView('map'); selectPole(id, true); };
  document.body.addEventListener('click', (e) => {
    const pole = e.target.closest('[data-pole]');
    if (pole) { openPole(pole.dataset.pole); return; }
    if (e.target.closest('[data-close]')) { selectPole(null, false); return; }
    const act = e.target.closest('[data-act]');
    if (act) { setView('map'); if (act.dataset.act === 'run') run(); else showAll(); return; }
    const task = e.target.closest('[data-task]');
    if (task) { accTask = task.dataset.task; renderAccuracy(); }
  });
  window.addEventListener('resize', () => map && map.invalidateSize());
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && S.selected && S.view === 'map') selectPole(null, false); });
}
function renderChrome() {
  const attr = (S.meta.attribution || []).map(clean);
  $('#credits').innerHTML = `<b>Credits.</b> ${attr.length ? attr.join(' &middot; ') + ' &middot; ' : ''}Map data &copy; OpenStreetMap contributors, tiles from tile.openstreetmap.org.`;
  document.title = 'Pole health triage with Matilda Jev';
}

async function init() {
  const q = new URLSearchParams(location.search).get('theme');
  if (q === 'dark' || q === 'light') document.documentElement.dataset.theme = q;
  try { await loadData(); }
  catch (e) {
    $('#view-map').innerHTML = `<div class="page"><div class="empty"><h2>Could not load the demo data</h2><p>${esc(e.message)}</p></div></div>`;
    return;
  }
  if (isMock()) console.warn('Pole health demo: showing MOCK data (' + (S.usingSample ? 'data/ not found, using sample/' : 'decisions.backend is mock') + '). These are not model outputs.');
  renderChrome(); renderLegend(); initMap(); bindEvents();
  updateButtons(); updateStats(performance.now(), true); renderPanel();
  const hash = location.hash.replace('#', '');
  const view = ['live', 'map', 'flip', 'queue', 'accuracy'].includes(hash) ? hash : 'live';
  S.view = view;
  await initLive();
  setView(view, true);
  window.__demo = { S, R, LV, run, pause, reset, showAll, selectPole }; // handy for capture scripts
}
init();
})();
