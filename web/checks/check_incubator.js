// Check the page's JavaScript core against the Python reference runs (twins/incubator/scripts/export_web.py).
// Usage: node web/checks/check_incubator.js twins/incubator/results web
const fs = require('fs'), path = require('path');
const [refDir, webDir] = process.argv.slice(2);
const IDT = require(path.resolve(webDir, 'incubator_core.js'));
const ref = JSON.parse(fs.readFileSync(path.join(refDir, 'ref_web_incubator.json')));
let worst = {}, fails = 0;
const cmp = (name, a, b) => { const d = Math.abs(Number(a) - Number(b)); worst[name] = Math.max(worst[name] || 0, d); };
const same = (name, a, b) => { if (a !== b) { fails++; if (fails < 5) console.log('mismatch', name, a, b); } };

// 1) closed loop with the same sensor noise and events
const box = new IDT.SimBox({ T0: 22, Troom: 22, noise: ref.noise }), tw = new IDT.Twin();
for (let k = 0; k < ref.obs.length; k++) {
  box.lidOpen = 25 * 60 <= box.t && box.t < 26 * 60;
  if (k === 800) box.act(35);
  const o = box.observe(), s = tw.update(o), ro = ref.obs[k], rs = ref.est[k];
  ['t', 'T', 'true_T', 'true_T_heater', 't3'].forEach(f => cmp('obs.' + f, o[f], ro[f]));
  same('heater_on', o.heater_on, ro.heater_on);
  ['T_est', 'T_heater_est', 'T_heater_sd', 'innov'].forEach(f => cmp('est.' + f, s[f], rs[f]));
  same('anomaly', s.anomaly, rs.anomaly);
  if (k === ref.whatif.k) {
    const w = tw.predict(1800, box.ctrl, 35);
    w.T.forEach((v, i) => cmp('whatif.T', v, ref.whatif.T[i]));
    w.heater.forEach((v, i) => same('whatif.heater', v, ref.whatif.heater[i]));
    cmp('whatif.energy', w.energy_Wh, ref.whatif.energy_Wh);
  }
  box.advance();
}
// 2) replay of the real experiment
global.window = {};
eval(fs.readFileSync(path.join(webDir, 'incubator_replay.js'), 'utf8'));
const R = window.INCUBATOR_REPLAY, rr = JSON.parse(fs.readFileSync(path.join(refDir, 'ref_web_incubator_replay.json')));
const tw2 = new IDT.Twin();
for (let i = 0; i < R.t.length; i++) {
  const s = tw2.update({ t: R.t[i], T: R.T[i], T_room: R.T_room[i], heater_on: R.heater_on[i] });
  ['T_est', 'T_heater_est', 'T_heater_sd', 'innov'].forEach(f => cmp('replay.' + f, s[f], rr[i][f]));
  same('replay.anomaly', s.anomaly, rr[i].anomaly);
}
for (const [k, v] of Object.entries(worst)) console.log(k.padEnd(22), 'max |JS - Python| =', v.toExponential(2));
const ok = fails === 0 && Object.values(worst).every(v => v < 1e-6);
console.log(fails ? `${fails} exact mismatches` : 'no exact mismatches (heater, anomaly)');
console.log(ok ? 'INCUBATOR WEB CHECK OK' : 'INCUBATOR WEB CHECK FAILED');
