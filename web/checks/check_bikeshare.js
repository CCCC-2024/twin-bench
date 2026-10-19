// Check the page's bike-share twin against Python (twins/bikeshare/scripts/export_web.py, scripts/compare_rates.py).
// Usage: node web/checks/check_bikeshare.js twins/bikeshare/results web
// Needs web/bikeshare_replay.js (local only): python twins/bikeshare/scripts/export_web.py --replay-only
const fs = require('fs'), path = require('path');
const [refDir, webDir] = process.argv.slice(2);
global.window = {};
eval(fs.readFileSync(path.join(webDir, 'bikeshare_model.js'), 'utf8'));
const B = require(path.resolve(webDir, 'bikeshare_core.js')), M = window.BIKESHARE_MODEL;
const ref = JSON.parse(fs.readFileSync(path.join(refDir, 'ref_web_bikeshare.json')));
const C = M.C;
let fails = 0, diff = 0, ties = 0;
const d = (a, b) => { diff = Math.max(diff, Math.abs(a - b)); };
const bad = msg => { fails++; if (fails < 8) console.log(msg); };

// model only: arbitrary rates and starting distributions
for (const c of ref.cases) {
  const u = B.udf(c.lp, c.lr, C); u.forEach((v, i) => d(v, c.udf[i]));
  const best = B.bestStart(c.lp, c.lr, C)[0];
  if (best !== c.best) { if (Math.abs(u[best] - u[c.best]) / u[c.best] < 1e-9) ties++; else bad(`best start ${best} vs ${c.best}`); }
  const f = B.forecast(c.lp, c.lr, C, [c.start]);
  f.dist[f.dist.length - 1][0].forEach((v, i) => d(v, c.dist_end[i]));
  f.lost_rent.forEach((v, h) => d(v[0], c.lost_rent[h])); f.lost_return.forEach((v, h) => d(v[0], c.lost_return[h]));
}

// the reference run: same simulated events, the twin continuing from what it learned up to 31 Oct
const st = new B.Station({ C, day0: ref.day0, bikes: ref.start_bikes, events: k => ref.changes[k] });
const tw = new B.Twin(C, new B.HistoricalRates(M.base));
const predsAt = {}; ref.preds.forEach(p => (predsAt[p.k] = predsAt[p.k] || []).push(p));
ref.rows.forEach((r, k) => {
  if (k === ref.surge_at) st.surge = 1.5;
  if (k === ref.truck_at) st.act({ set_bikes: 5 });
  for (const p of predsAt[k] || []) {
    const q = tw.predict(12, p.set_bikes);
    for (const key of ['mean_bikes', 'p_empty', 'p_full', 'lost_rent', 'lost_return', 'rent_rate', 'return_rate']) q[key].forEach((v, i) => d(v, p[key][i]));
    d(q.expected_lost, p.expected_lost);
    if (q.p10.join() !== p.p10.join() || q.p90.join() !== p.p90.join()) bad(`10-90 % band differs at hour ${k}, set ${p.set_bikes}`);
  }
  const o = st.observe(), e = tw.update(o), a = tw.decide();
  if (o.bikes !== r.bikes || o.lost_rent !== r.lost_rent || o.lost_return !== r.lost_return || o.pickups !== r.pickups || o.returns !== r.returns || e.days_seen !== r.days_seen)
    bad(`hour ${k}: station or days differ`);
  d(e.next_rent_rate, r.rent_rate); d(e.next_return_rate, r.return_rate); d(e.next_rent_sd, r.rent_sd); d(e.next_return_sd, r.return_sd);
  if ((a === null) !== (r.set_bikes === null)) bad(`hour ${k}: decision presence differs`);
  else if (a) { d(a.expected_lost, r.expected_lost); if (a.set_bikes !== r.set_bikes) bad(`hour ${k}: decision ${a.set_bikes} vs ${r.set_bikes}`); st.act(a); }
  st.advance();
});

// the recorded days: with each rate model the page's replay takes the nightly decisions, and loses the rentals and
// returns, of the Python twin with the same rate model (results/compare_rates.json, the README's table)
global.window = {};
eval(fs.readFileSync(path.join(webDir, 'bikeshare_replay.js'), 'utf8'));
const R = window.BIKESHARE_REPLAY, cmp = JSON.parse(fs.readFileSync(path.join(refDir, 'compare_rates.json')));
const days = Object.keys(cmp.days).sort();
function recordedDays(name, rates, start, nHours) {
  const s = new B.Station({ C, day0: R.day0, bikes: start, events: k => k < nHours ? R.hours[k] : null });
  const twin = new B.Twin(C, rates), starts = [start], expected = [null], lost = [];
  for (let o; (o = s.observe()); s.advance()) {
    if (o.hour === 0) lost.push(0);
    lost[lost.length - 1] += o.lost_rent + o.lost_return;
    twin.update(o); const a = twin.decide();
    if (a) { s.act(a); starts.push(a.set_bikes); expected.push(a.expected_lost); }
  }
  days.forEach((day, i) => {
    const c = cmp.days[day][name];
    if (i < starts.length && starts[i] !== c.start) bad(`${name} ${day}: start ${starts[i]} vs ${c.start}`);
    if (i > 0 && i < expected.length) d(expected[i], c.expected);
    if (i < lost.length && lost[i] !== c.lost_rent + c.lost_return) bad(`${name} ${day}: lost ${lost[i]} vs ${c.lost_rent + c.lost_return}`);
  });
  return `${name} ${lost.length} days`;
}
const replays = [recordedDays('historical_all', new B.HistoricalRates(M.base), M.start_bikes, R.hours.length)];
if (R.rnn) replays.push(recordedDays('rnn', new B.PrecomputedRates(R.rnn.table), R.rnn.start_bikes, R.rnn.table.length));
else console.log('bikeshare_replay.js has no VP-RNN rates (export_web.py ran without the rnn extra): rnn not checked');

const nDec = ref.rows.filter(r => r.set_bikes !== null).length;
console.log(`${ref.cases.length} model cases (ties ${ties}), ${ref.rows.length} simulated hours, ${nDec} nightly decisions, ${ref.preds.length} forecasts, ` +
            `recorded: ${replays.join(', ')}: exact mismatches ${fails}; max |difference| ${diff.toExponential(2)}`);
console.log(fails === 0 && diff < 1e-9 ? 'BIKESHARE WEB CHECK OK' : 'BIKESHARE WEB CHECK FAILED');
