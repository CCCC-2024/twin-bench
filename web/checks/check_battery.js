// Check the page's battery decision math against Python (twins/battery/scripts/export_web.py).
// Usage: node web/checks/check_battery.js twins/battery/results web
const fs = require('fs'), path = require('path');
const [refDir, webDir] = process.argv.slice(2);
const BAT = require(path.resolve(webDir, 'battery_core.js'));
global.window = {};
eval(fs.readFileSync(path.join(webDir, 'battery_cells.js'), 'utf8'));
const cells = Object.fromEntries(window.BATTERY_CELLS.map(c => [c.idx, c]));
const ref = JSON.parse(fs.readFileSync(path.join(refDir, 'ref_web_battery.json')));
let xMis = 0, hMis = 0, uDiff = 0;
for (const r of ref) {
  const c = cells[r.idx], j = c.cycles.indexOf(r.cycle), sj = r.cycle - c.k_fpt;
  const d = BAT.decideOnline(c.q.slice(0, j + 1), c.q0, c.st.log10_a[sj], c.st.b[sj], c.st.eol_pred[sj], [r.w, 1 - r.w]);
  if (d.x !== r.x) { xMis++; if (xMis < 5) console.log('x mismatch', r, d.x); }
  if (d.horizon !== r.horizon) hMis++;
  if (r.umax !== null) uDiff = Math.max(uDiff, Math.abs(d.umax - r.umax));
}
// the automatic decisions stored for each cell
let autoOk = true;
for (const c of window.BATTERY_CELLS) {
  const k = c.decision.cycle_decided, j = c.cycles.indexOf(k), sj = k - c.k_fpt;
  const d = BAT.decideOnline(c.q.slice(0, j + 1), c.q0, c.st.log10_a[sj], c.st.b[sj], c.st.eol_pred[sj]);
  autoOk = autoOk && d.x === c.decision.x_twin && c.q[j] <= BAT.DECIDE_AT;
}
console.log(`${ref.length} what-if cases: retirement-cycle mismatches ${xMis}, horizon mismatches ${hMis}, max |utility diff| ${uDiff.toExponential(2)}`);
console.log(`automatic decisions reproduced: ${autoOk}`);
console.log(xMis === 0 && hMis === 0 && uDiff < 1e-9 && autoOk ? 'BATTERY WEB CHECK OK' : 'BATTERY WEB CHECK FAILED');
