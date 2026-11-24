// Check the page's tumor data against the twin's saved runs (twins/tumor/scripts/run_twin.py, export_web.py).
// The page runs no tumor model; it replays web/tumor_run.js. Usage: node web/checks/check_tumor.js twins/tumor/results web
const fs = require('fs'), path = require('path');
const [refDir, webDir] = process.argv.slice(2);
global.window = {};
eval(fs.readFileSync(path.join(webDir, 'tumor_run.js'), 'utf8'));
const R = window.TUMOR_RUN, read = f => JSON.parse(fs.readFileSync(path.join(refDir, 'HGG_demo_001', f)));
const run10 = read('twin_run_10.json'), run5 = read('twin_run_5.json');
const rel = (a, b) => Math.abs(a - b) / Math.abs(b);
let worst = 0, fails = 0;
run10.visits.forEach((v, k) => {
  const w = R.visits[k];
  if (w.day !== v.day || (v.expected === null) !== (w.expected === null)) fails++;
  worst = Math.max(worst, rel(w.measured, v.measured), v.expected === null ? 0 : rel(w.expected, v.expected),
                   ...['d', 'k', 'alpha_RT', 'chemo_sensitivity'].map(p => rel(w.params[p], v[p])));
});
const last = R.visits[9].curves.plan;
run10.forecast.day.forEach((d, j) => { const i = R.days.indexOf(d); if (i < 0) fails++; else worst = Math.max(worst, rel(last[i], run10.forecast.cell_count[j])); });
const fifth = R.visits[4].curves;
for (const [dose, count] of Object.entries(run5.whatif_chemo_total)) worst = Math.max(worst, rel(fifth[dose][fifth[dose].length - 1], count));
console.log(`${R.visits.length} visits: structural mismatches ${fails}; max relative difference from the saved runs ${worst.toExponential(2)}`);
console.log(fails === 0 && worst < 1e-6 ? 'TUMOR WEB CHECK OK' : 'TUMOR WEB CHECK FAILED');
