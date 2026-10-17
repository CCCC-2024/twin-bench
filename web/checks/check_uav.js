// Check the page's UAV twin against Python (twins/uav/scripts/export_web.py).
// Usage: node web/checks/check_uav.js twins/uav/results web
const fs = require('fs'), path = require('path');
const [refDir, webDir] = process.argv.slice(2);
global.window = {};
eval(fs.readFileSync(path.join(webDir, 'uav_model.js'), 'utf8'));
const U = require(path.resolve(webDir, 'uav_core.js')).make(window.UAV_MODEL);
const ref = JSON.parse(fs.readFileSync(path.join(refDir, 'ref_web_uav.json')));
let fails = 0, diff = 0;
const pol25 = U.policy(2.5); if (pol25.join() !== window.UAV_MODEL.policy.join()) { fails++; console.log('default policy differs'); }
for (const [a, p] of Object.entries(ref.policies)) if (U.policy(Number(a)).join() !== p.join()) { fails++; console.log('policy differs at alpha2', a); }
const asset = new U.Asset({ noise: ref.noise, growth: ref.growth }), tw = new U.Twin();
asset.act(tw.decide());
const beliefs = {};
ref.rows.forEach((r, k) => {
  if (k === 40) { asset.damage(0); asset.damage(0); }
  if (k === 70) asset.damage(1);
  if (k === 90) tw.setPolicy(0.5);
  const o = asset.observe(), e = tw.update(o), d = tw.decide();
  if (o.u !== r.u || o.true1 !== r.true1 || o.true2 !== r.true2 || d.u !== r.next_u) { fails++; if (fails < 5) console.log('mismatch at', k, o.u, r.u, d.u, r.next_u); }
  diff = Math.max(diff, Math.abs(e.est1 - r.est1), Math.abs(e.est2 - r.est2), Math.abs(e.pmax - r.pmax));
  beliefs[k] = e.belief.slice();
  asset.act(d); asset.advance();
});
let rdiff = 0;
for (const r of ref.rollouts) { const o = U.rollout(beliefs[r.k], U.policy(r.alpha2), 30); for (const key of ['n3g', 'mean_d1', 'mean_d2', 'p_d1_60']) rdiff = Math.max(rdiff, Math.abs(o[key] - r[key])); }
console.log(`${ref.rows.length} turns: exact mismatches (maneuver, true damage, next maneuver, policies) ${fails}; max |estimate diff| ${diff.toExponential(2)}; max |rollout diff| ${rdiff.toExponential(2)}`);
console.log(fails === 0 && diff < 1e-9 && rdiff < 1e-9 ? 'UAV WEB CHECK OK' : 'UAV WEB CHECK FAILED');
