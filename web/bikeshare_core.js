/* Bike-share station twin for the page, ported from twins/bikeshare (model.py, twin.py, rates.py, sources.py).
 * Needs window.BIKESHARE_MODEL (bikeshare_model.js) for the learned totals and the simulated demand.
 * check_bikeshare.js compares this file with Python.
 */
(function (root) {
  'use strict';

  // ---------- dense matrices: row-major Float64Array, n x n ----------
  function mm(A, B, n, m) {            // (n x n) @ (n x m)
    m = m || n; const C = new Float64Array(n * m);
    for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) { const a = A[i * n + k]; if (a === 0) continue;
      for (let j = 0; j < m; j++) C[i * m + j] += a * B[k * m + j]; }
    return C;
  }
  function solve(A, B, n) {            // A X = B by LU with partial pivoting; A, B are n x n and are overwritten
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r * n + c]) > Math.abs(A[p * n + c])) p = r;
      if (p !== c) for (let j = 0; j < n; j++) { let t = A[c * n + j]; A[c * n + j] = A[p * n + j]; A[p * n + j] = t;
                                                  t = B[c * n + j]; B[c * n + j] = B[p * n + j]; B[p * n + j] = t; }
      const d = A[c * n + c];
      for (let r = c + 1; r < n; r++) { const f = A[r * n + c] / d; if (f === 0) continue;
        for (let j = c; j < n; j++) A[r * n + j] -= f * A[c * n + j];
        for (let j = 0; j < n; j++) B[r * n + j] -= f * B[c * n + j]; }
    }
    for (let c = n - 1; c >= 0; c--) { const d = A[c * n + c];
      for (let j = 0; j < n; j++) { let s = B[c * n + j]; for (let k = c + 1; k < n; k++) s -= A[c * n + k] * B[k * n + j]; B[c * n + j] = s / d; } }
    return B;
  }
  /* matrix exponential: scaling and squaring with the degree-13 Pade approximant (Higham 2005) */
  const B13 = [64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000,
               10559470521600, 670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1];
  function expm(A, n) {
    let nrm = 0;
    for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < n; i++) s += Math.abs(A[i * n + j]); if (s > nrm) nrm = s; }
    const s = nrm > 5.371920351148152 ? Math.ceil(Math.log2(nrm / 5.371920351148152)) : 0, sc = Math.pow(2, -s);
    const X = A.map(v => v * sc), X2 = mm(X, X, n), X4 = mm(X2, X2, n), X6 = mm(X2, X4, n);
    const W1 = new Float64Array(n * n), Z1 = new Float64Array(n * n), W = new Float64Array(n * n), V = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) {
      W1[i] = B13[13] * X6[i] + B13[11] * X4[i] + B13[9] * X2[i];
      Z1[i] = B13[12] * X6[i] + B13[10] * X4[i] + B13[8] * X2[i];
    }
    const W1b = mm(X6, W1, n), Z1b = mm(X6, Z1, n);
    for (let i = 0; i < n * n; i++) {
      W[i] = W1b[i] + B13[7] * X6[i] + B13[5] * X4[i] + B13[3] * X2[i];
      V[i] = Z1b[i] + B13[6] * X6[i] + B13[4] * X4[i] + B13[2] * X2[i];
    }
    for (let i = 0; i < n; i++) { W[i * n + i] += B13[1]; V[i * n + i] += B13[0]; }
    const U = mm(X, W, n), P = new Float64Array(n * n), Q = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) { P[i] = V[i] + U[i]; Q[i] = V[i] - U[i]; }
    let R = solve(Q, P, n);
    for (let k = 0; k < s; k++) R = mm(R, R, n);
    return R;
  }

  // ---------- station model (model.py) ----------
  function generator(lp, lr, C) {
    const n = C + 1, Q = new Float64Array(n * n);
    for (let i = 0; i < C; i++) { Q[i * n + i + 1] = lr; Q[(i + 1) * n + i] = lp; }
    for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += Q[i * n + j]; Q[i * n + i] = -s; }
    return Q;
  }
  /* one hour: P = expm(Q); M = integral_0^1 expm(Q s) ds @ [e_0, e_C] (fraction of the hour empty, full) */
  function hour(lp, lr, C) {
    const n = C + 1, m = n + 2, Q = generator(lp, lr, C), A = new Float64Array(m * m);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i * m + j] = Q[i * n + j];
    A[0 * m + n] = 1; A[C * m + n + 1] = 1;
    const E = expm(A, m), P = new Float64Array(n * n), M = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) { for (let j = 0; j < n; j++) P[i * n + j] = E[i * m + j]; M[i * 2] = E[i * m + n]; M[i * 2 + 1] = E[i * m + n + 1]; }
    return { P, M };
  }
  /* starts: array of distributions over 0..C. Per hour: dist (end of hour), lost_rent, lost_return, p_empty, p_full */
  function forecast(lp, lr, C, starts) {
    const n = C + 1, k = starts.length;
    let pi = new Float64Array(k * n); starts.forEach((s, r) => { for (let j = 0; j < n; j++) pi[r * n + j] = s[j]; });
    const out = { dist: [], lost_rent: [], lost_return: [], p_empty: [], p_full: [] };
    for (let h = 0; h < lp.length; h++) {
      const { P, M } = hour(lp[h], lr[h], C), a = lp[h], b = lr[h];
      const e = [], f = [], le = [], lf = [];
      for (let r = 0; r < k; r++) { let s0 = 0, s1 = 0;
        for (let j = 0; j < n; j++) { s0 += pi[r * n + j] * M[j * 2]; s1 += pi[r * n + j] * M[j * 2 + 1]; }
        e.push(s0); f.push(s1); le.push(a * s0); lf.push(b * s1); }
      const nx = new Float64Array(k * n);
      for (let r = 0; r < k; r++) for (let i = 0; i < n; i++) { const v = pi[r * n + i]; if (v === 0) continue;
        for (let j = 0; j < n; j++) nx[r * n + j] += v * P[i * n + j]; }
      pi = nx;
      const d = []; for (let r = 0; r < k; r++) d.push(Array.from(pi.subarray(r * n, (r + 1) * n)));
      out.dist.push(d); out.lost_rent.push(le); out.lost_return.push(lf); out.p_empty.push(e); out.p_full.push(f);
    }
    return out;
  }
  const eye = n => Array.from({ length: n }, (_, i) => { const r = new Array(n).fill(0); r[i] = 1; return r; });
  function udf(lp, lr, C) {
    const f = forecast(lp, lr, C, eye(C + 1)), u = new Array(C + 1).fill(0), v = new Array(C + 1).fill(0);
    for (let h = 0; h < lp.length; h++) for (let s = 0; s <= C; s++) { u[s] += f.lost_rent[h][s]; v[s] += f.lost_return[h][s]; }
    return u.map((x, s) => x + v[s]);
  }
  function bestStart(lp, lr, C) { const f = udf(lp, lr, C); let s = 0; for (let i = 1; i <= C; i++) if (f[i] < f[s]) s = i; return [s, f[s]]; }
  function replay(C, start, changes) {
    let bikes = start, lp = 0, lr = 0;
    for (const c of changes) { if (c < 0) { if (bikes === 0) lp++; else bikes--; } else if (c > 0) { if (bikes === C) lr++; else bikes++; } }
    return [lp, lr, bikes];
  }

  // ---------- twin (twin.py, with HistoricalRates from rates.py) ----------
  const weekday = day => day % 7 < 5;                 // day index from 1 Jan 2018, a Monday
  class Twin {
    /* base: totals per day type and hour learned before the page starts: [n days, rentals, returns] */
    constructor(C, base) { this.C = C; this.base = base; this.hist = {}; this.bikes = null; this.day = null; this.hour = null; }
    update(o) {
      const key = (o.weekday ? 'w' : 'e') + o.hour, H = this.hist[key] || (this.hist[key] = { days: [], cp: [0], cr: [0] });
      H.days.push(o.day); H.cp.push(H.cp[H.cp.length - 1] + o.pickups); H.cr.push(H.cr[H.cr.length - 1] + o.returns);
      this.bikes = o.bikes; this.day = o.day; this.hour = o.hour;
      const [nd, nh] = this.next(), r = this.rate(nd, nh);
      return { t: o.t, bikes: this.bikes, next_rent_rate: r.mean[0], next_return_rate: r.mean[1],
               next_rent_sd: r.sd[0], next_return_sd: r.sd[1], days_seen: r.n };
    }
    rate(day, hour) {                                  // Gamma posterior (flat prior) from days before `day`
      const wk = weekday(day), b = this.base[wk ? 'weekday' : 'weekend'][hour], H = this.hist[(wk ? 'w' : 'e') + hour];
      let end = 0; if (H) while (end < H.days.length && H.days[end] < day) end++;
      const n = b[0] + end; if (n === 0) return { mean: [NaN, NaN], sd: [NaN, NaN], n: 0 };
      const cp = b[1] + (H ? H.cp[end] : 0), cr = b[2] + (H ? H.cr[end] : 0);
      return { mean: [cp / n, cr / n], sd: [Math.sqrt(cp) / n, Math.sqrt(cr) / n], n };
    }
    next() { return this.hour < 23 ? [this.day, this.hour + 1] : [this.day + 1, 0]; }
    ratesAhead(h) {
      let [day, hr] = this.next(); const lp = [], lr = [];
      for (let i = 0; i < h; i++) { const m = this.rate(day, hr).mean; lp.push(m[0]); lr.push(m[1]); if (hr < 23) hr++; else { day++; hr = 0; } }
      return [lp, lr];
    }
    predict(h, setBikes = null) {
      const C = this.C, [lp, lr] = this.ratesAhead(h), start = new Array(C + 1).fill(0);
      start[setBikes === null ? this.bikes : setBikes] = 1;
      const f = forecast(lp, lr, C, [start]), out = { rent_rate: lp, return_rate: lr, mean_bikes: [], p10: [], p90: [],
        p_empty: [], p_full: [], lost_rent: [], lost_return: [], expected_lost: 0 };
      let sr = 0, st = 0;
      for (let i = 0; i < h; i++) {
        const d = f.dist[i][0]; let m = 0, c = 0, a = 0, b = 0;
        for (let j = 0; j <= C; j++) { m += d[j] * j; c += d[j]; if (c < 0.1) a++; if (c < 0.9) b++; }
        out.mean_bikes.push(m); out.p10.push(a); out.p90.push(b);
        out.p_empty.push(f.p_empty[i][0]); out.p_full.push(f.p_full[i][0]);
        out.lost_rent.push(f.lost_rent[i][0]); out.lost_return.push(f.lost_return[i][0]);
        sr += f.lost_rent[i][0]; st += f.lost_return[i][0];
      }
      out.expected_lost = sr + st;
      return out;
    }
    decide() {                                         // at 23:00: bikes to start tomorrow with
      if (this.hour !== 23) return null;
      const [lp, lr] = this.ratesAhead(24);
      if (lp.some(v => !isFinite(v)) || lr.some(v => !isFinite(v))) return null;
      const [s, cost] = bestStart(lp, lr, this.C);
      return { set_bikes: s, expected_lost: cost };
    }
  }

  // ---------- physical side (sources.py): simulated demand, or given events per hour ----------
  function rng(seed) {
    let a = seed >>> 0;
    return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  class Station {
    /* rates(day, hour) -> [rentals/h, returns/h] for simulated demand; events(k) -> "+-..." string for hour k
       (recorded or given), or null/undefined when there is no more data */
    constructor({ C, day0, bikes, rates = null, events = null, seed = 1 }) {
      Object.assign(this, { C, day0, bikes, rates, events, k: 0, moved: 0, surge: 1, lastChanges: [] });
      this.rand = rng(seed);
    }
    when() { return [this.day0 + Math.floor(this.k / 24), this.k % 24]; }
    act(a) { if (a && a.set_bikes !== undefined) { const t = Math.round(Math.min(Math.max(a.set_bikes, 0), this.C)); this.moved += t - this.bikes; this.bikes = t; } }
    poisson(lam) { let n = 0; while (lam > 30) { n += this.poissonSmall(30); lam -= 30; } return n + this.poissonSmall(lam); }
    poissonSmall(l) { const L = Math.exp(-l); let k = 0, p = 1; do { k++; p *= this.rand(); } while (p > L); return k - 1; }
    observe() {
      const [day, hour] = this.when(); let changes;
      if (this.events) { const s = this.events(this.k); if (s === null || s === undefined) return null; changes = Array.from(s, c => c === '+' ? 1 : -1); }
      else { const [a, b] = this.rates(day, hour).map(v => v * this.surge), n = a + b > 0 ? this.poisson(a + b) : 0;
             changes = []; for (let i = 0; i < n; i++) changes.push(this.rand() < a / (a + b) ? -1 : 1); }
      this.lastChanges = changes;
      const [lp, lr, bikes] = replay(this.C, this.bikes, changes); this.bikes = bikes;
      return { t: this.k, day, hour, weekday: weekday(day), pickups: changes.filter(c => c < 0).length,
               returns: changes.filter(c => c > 0).length, bikes, lost_rent: lp, lost_return: lr };
    }
    advance() { this.k += 1; }
  }

  const BIKE = { expm, generator, hour, forecast, udf, bestStart, replay, weekday, Twin, Station };
  if (typeof module !== 'undefined' && module.exports) module.exports = BIKE; else root.BIKE = BIKE;
})(typeof self !== 'undefined' ? self : this);
