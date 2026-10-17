/* Battery twin: the deterministic parts the page computes itself, ported from
 * twins/battery/bdt (pf.power_law, decision._exp_utility / utility_curve / decide_online).
 * The particle filter itself runs in Python on the shared skeleton; its states are in battery_cells.js.
 * web/check_battery.js compares this file with Python.
 */
(function (root) {
  'use strict';
  const AH_BOUNDS = [300, 1000], MTBC_BOUNDS = [0.210, 0.250], HORIZON_CAP = 10000, DECIDE_AT = 0.95, FPT = 0.985;

  const powerLaw = (k, la, b) => 1 - Math.pow(10, la) * Math.pow(k, b);

  function expUtility(v, lb, ub, R) {
    if (v < lb) return 0.0;
    if (v > ub) return 1.0;
    const a = Math.exp(-lb / R) / (Math.exp(-lb / R) - Math.exp(-ub / R));
    const b = 1 / (Math.exp(-lb / R) - Math.exp(-ub / R));
    return a - b * Math.exp(-v / R);
  }

  function utilityCurve(capAh, xs, w) {
    const csum = new Float64Array(capAh.length + 1);
    for (let i = 0; i < capAh.length; i++) csum[i + 1] = csum[i] + capAh[i];
    return xs.map(x => {
      const n = x - 2, total = csum[n], meanH = csum[n] / n / 4.4;
      return w[0] * expUtility(total, AH_BOUNDS[0], AH_BOUNDS[1], 200) + w[1] * expUtility(meanH, MTBC_BOUNDS[0], MTBC_BOUNDS[1], 0.015);
    });
  }

  /* Retirement decision using only what the twin knows now (decision.decide_online). */
  function decideOnline(qHist, q0, la, b, eolPred, w = [0.5, 0.5]) {
    const kNow = qHist.length + 1;
    const horizon = Math.trunc(Math.min(eolPred, HORIZON_CAP));
    if (horizon - 1 <= kNow) return { x: kNow, horizon, umax: null, xs: [kNow], u: [NaN] };
    const capAh = qHist.map(q => q * q0);
    for (let k = kNow + 1; k < horizon; k++) capAh.push(powerLaw(k, la, b) * q0);
    const xs = []; for (let x = kNow; x < horizon - 1; x++) xs.push(x);
    const u = utilityCurve(capAh, xs, w);
    let best = 0; for (let i = 1; i < u.length; i++) if (u[i] > u[best]) best = i;
    return { x: xs[best], horizon, umax: u[best], xs, u, capAh };
  }

  /* Energy delivered and mean discharge time if retired at cycle x (the two utility attributes). */
  function attributes(capAh, x) {
    let s = 0; for (let i = 0; i < x - 2; i++) s += capAh[i];
    return { ah: s, meanH: s / (x - 2) / 4.4 };
  }

  const BAT = { powerLaw, expUtility, utilityCurve, decideOnline, attributes, DECIDE_AT, FPT };
  if (typeof module !== 'undefined' && module.exports) module.exports = BAT; else root.BAT = BAT;
})(typeof self !== 'undefined' ? self : this);
