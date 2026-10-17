/* UAV twin for the page, ported from twins/uav/udt (model.py, twin.py, assets.ControlledUAV).
 * Needs window.UAV_MODEL (uav_model.js). check_uav.js compares this file with Python.
 */
(function (root) {
  'use strict';
  function make(M) {
    const LOAD = { '2g': 2, '3g': 3 }, NS = 25;
    const means = M.C.map(c => M.raw[c].map(row => row.map(v => v / LOAD[c])));
    const LOGC = Math.log(Math.sqrt(2 * Math.PI));
    function T(ci) {
      const p = M.p[M.C[ci]], S = M.S, T = [];
      for (let i = 0; i < NS; i++) { T.push(new Array(NS).fill(0)); for (let j = 0; j < NS; j++) {
        const d1 = S[j][0] - S[i][0], d2 = S[j][1] - S[i][1];
        if (S[i][0] === 80 && S[i][1] === 80 && S[j][0] === 80 && S[j][1] === 80) T[i][j] = 1;
        else if (d1 === 0 && d2 === 0) T[i][j] = (1 - p) * (1 - p);
        else if (d1 === 20 && d2 === 20) T[i][j] = p * p;
        else if (d1 === 20 && d2 === 0) T[i][j] = p * (1 - p);
        else if (d1 === 0 && d2 === 20) T[i][j] = p * (1 - p); } }
      return T;
    }
    const TM = [T(0), T(1)];
    const vecT = (b, Tm) => { const o = new Array(NS).fill(0); for (let i = 0; i < NS; i++) if (b[i]) for (let j = 0; j < NS; j++) o[j] += b[i] * Tm[i][j]; return o; };
    const norm = b => { const s = b.reduce((a, v) => a + v, 0); return b.map(v => v / s); };
    function likelihood(m, ci) {
      const sd = M.sigTwin / Math.sqrt(LOAD[M.C[ci]]), ll = [];
      for (let s = 0; s < NS; s++) { let a = 0; for (let k = 0; k < 24; k++) { const z = (m[k] - means[ci][s][k]) / sd; a += -z * z / 2 - LOGC - Math.log(sd); } ll.push(a); }
      const mx = Math.max(...ll); return norm(ll.map(v => Math.exp(v - mx)));
    }
    function policy(alpha2) {
      const R = M.SR.map(r => [r[0] + alpha2 * -0.1, r[1] + alpha2 * 0.1]);
      let V = new Array(NS).fill(0);
      const q = (s, ci, V) => { let a = 0; for (let j = 0; j < NS; j++) a += TM[ci][s][j] * V[j]; return R[s][ci] + a; };
      for (let it = 2; it < 100; it++) { const nv = []; for (let s = 0; s < NS; s++) nv.push(M.gamma * Math.max(q(s, 0, V), q(s, 1, V))); V = nv; }
      const pol = []; for (let s = 0; s < NS; s++) pol.push(q(s, 1, V) > q(s, 0, V) ? 1 : 0);
      return pol;
    }
    function rollout(belief, pol, horizon = 30) {
      let b = belief.slice(), n3g = 0;
      for (let h = 0; h < horizon; h++) {
        for (let s = 0; s < NS; s++) if (pol[s] === 1) n3g += b[s];
        const o = new Array(NS).fill(0);
        for (let i = 0; i < NS; i++) if (b[i]) { const Tm = TM[pol[i]]; for (let j = 0; j < NS; j++) o[j] += b[i] * Tm[i][j]; }
        b = norm(o);
      }
      let d1 = 0, d2 = 0, p60 = 0; for (let s = 0; s < NS; s++) { d1 += b[s] * M.S[s][0]; d2 += b[s] * M.S[s][1]; if (M.S[s][0] >= 60) p60 += b[s]; }
      return { n3g, mean_d1: d1, mean_d2: d2, p_d1_60: p60 };
    }
    class Twin {
      constructor(alpha2 = 2.5) { this.setPolicy(alpha2); this.bPred = new Array(NS).fill(0); this.bPred[0] = 1; this.bPost = this.bPred.slice(); }
      setPolicy(a) { this.alpha2 = a; this.pol = policy(a); }
      decide() { let p = [0, 0]; for (let s = 0; s < NS; s++) p[this.pol[s]] += this.bPred[s]; return { u: p[1] > p[0] ? 1 : 0, p3g: p[1] }; }
      update(o) {
        const lik = likelihood(o.m, o.u);
        this.bPost = norm(this.bPred.map((v, i) => v * lik[i]));
        this.bPred = norm(vecT(this.bPost, TM[o.u]));
        let e1 = 0, e2 = 0, im = 0; for (let s = 0; s < NS; s++) { e1 += this.bPost[s] * M.S[s][0]; e2 += this.bPost[s] * M.S[s][1]; if (this.bPost[s] > this.bPost[im]) im = s; }
        return { t: o.t, est1: e1, est2: e2, map: M.S[im], pmax: this.bPost[im], belief: this.bPost };
      }
      predict(h = 30, alpha2 = null) { const pol = alpha2 === null ? this.pol : policy(alpha2); return Object.assign({ policy: pol }, rollout(this.bPost, pol, h)); }
    }
    function gauss(seed) {
      let a = seed >>> 0;
      const rand = () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
      return { rand, normal: sd => { let u = 0; while (u === 0) u = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand()) * sd; } };
    }
    class Asset {     // ControlledUAV: damage on the 20 % grid, grows with the maneuver
      constructor({ seed = 11, noise = null, growth = null } = {}) {
        this.g = [0, 0]; this.u = 1; this.k = 0; this.grows = true;
        const r = gauss(seed);
        this.noiseAt = noise ? (k => noise[k]) : (() => Array.from({ length: 24 }, () => r.normal(M.sigAsset)));
        this.growthAt = growth ? (k => growth[k]) : (() => [r.rand(), r.rand()]);
      }
      damage(region) { this.g[region] = Math.min(4, this.g[region] + 1); }
      repair() { this.g = [0, 0]; }
      observe() {
        const s = this.g[0] * 5 + this.g[1], c = M.C[this.u], n = this.noiseAt(this.k);
        const m = M.raw[c][s].map((v, i) => Math.max(1, v + n[i]) / LOAD[c]);
        return { t: this.k, m, u: this.u, true1: 20 * this.g[0], true2: 20 * this.g[1] };
      }
      act(a) { this.u = a.u; }
      advance() {
        const p = M.p[M.C[this.u]], gr = this.growthAt(this.k), grew = [];
        if (this.grows) for (let r = 0; r < 2; r++) if (gr[r] < p && this.g[r] < 4) { this.g[r] += 1; grew.push(r); }
        this.k += 1; return grew;
      }
    }
    return { M, means, TM, likelihood, policy, rollout, Twin, Asset, LOAD };
  }
  const UAV = { make };
  if (typeof module !== 'undefined' && module.exports) module.exports = UAV; else root.UAV = UAV;
})(typeof self !== 'undefined' ? self : this);
