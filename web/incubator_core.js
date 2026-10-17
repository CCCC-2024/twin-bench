/* Incubator digital twin: computing core for the page.
 * A line-by-line port of twins/incubator/idt (model.py, controller.py, kalman.py,
 * plant_sim.py, twin.py). Python is the reference; scripts/check_web.js compares the two.
 */
(function (root) {
  'use strict';

  const PARAMS_2023 = { C_air: 267.55929458, G_box: 0.5763498, C_heater: 329.25376821,
                        G_heater: 1.67053237, V: 12.15579391, I: 1.53551347 };
  // Another parameter set that fits the real 2023 data almost equally well (step 1a profile,
  // C_heater fixed at 100 J/K): used to simulate a box that differs from the twin's model.
  const PARAMS_ALT = Object.assign({}, PARAMS_2023,
                        { C_air: 535.356014, G_box: 0.576171, C_heater: 100.0, G_heater: 1.047667 });
  const G_OPEN_LID = 14.3873949;
  const DT = 3.0;
  const QSTEP = 0.0625;
  const ANOMALY_C = 0.5, ANOMALY_N = 3;

  // ---------- small linear algebra ----------
  function matmul(A, B) {
    const n = A.length, m = B[0].length, k = B.length, C = [];
    for (let i = 0; i < n; i++) {
      C.push(new Array(m).fill(0));
      for (let j = 0; j < m; j++) { let s = 0; for (let l = 0; l < k; l++) s += A[i][l] * B[l][j]; C[i][j] = s; }
    }
    return C;
  }
  function expm(M) {                       // scaling and squaring + Taylor series
    const n = M.length;
    let norm = 0;
    for (const r of M) norm = Math.max(norm, r.reduce((a, v) => a + Math.abs(v), 0));
    const s = Math.max(0, Math.ceil(Math.log2(norm / 0.25)));
    const A = M.map(r => r.map(v => v / Math.pow(2, s)));
    let E = A.map((r, i) => r.map((_, j) => (i === j ? 1 : 0)));
    let term = E.map(r => r.slice());
    for (let k = 1; k <= 20; k++) {
      term = matmul(term, A).map(r => r.map(v => v / k));
      E = E.map((r, i) => r.map((v, j) => v + term[i][j]));
    }
    for (let i = 0; i < s; i++) E = matmul(E, E);
    return E;
  }

  // ---------- model.py ----------
  function continuous(p) {
    const Ca = p.C_air, Gb = p.G_box, Ch = p.C_heater, Gh = p.G_heater, P = p.V * p.I;
    return { A: [[-(Gh + Gb) / Ca, Gh / Ca], [Gh / Ch, -Gh / Ch]],
             B: [[0, Gb / Ca], [P / Ch, 0]] };
  }
  function discretize(p, dt) {
    const { A, B } = continuous(p);
    const M = [[A[0][0], A[0][1], B[0][0], B[0][1]], [A[1][0], A[1][1], B[1][0], B[1][1]],
               [0, 0, 0, 0], [0, 0, 0, 0]].map(r => r.map(v => v * dt));
    const E = expm(M);
    return { Ad: [[E[0][0], E[0][1]], [E[1][0], E[1][1]]], Bd: [[E[0][2], E[0][3]], [E[1][2], E[1][3]]] };
  }
  class Stepper {
    constructor(p) { this.p = Object.assign({}, p); this.cache = new Map(); }
    mats(dt) {
      const key = Math.round(dt * 1000) / 1000;
      if (!this.cache.has(key)) this.cache.set(key, discretize(this.p, key));
      return this.cache.get(key);
    }
    step(x, heaterOn, Troom, dt) {
      const { Ad, Bd } = this.mats(dt), u = [Number(heaterOn), Troom];
      return [Ad[0][0] * x[0] + Ad[0][1] * x[1] + Bd[0][0] * u[0] + Bd[0][1] * u[1],
              Ad[1][0] * x[0] + Ad[1][1] * x[1] + Bd[1][0] * u[0] + Bd[1][1] * u[1]];
    }
  }

  // ---------- controller.py ----------
  class Controller {
    constructor(desired = 38, lowerBound = 2, heatingTime = 60000, heatingGap = 5) {
      Object.assign(this, { desired, lowerBound, heatingTime, heatingGap,
                            state: 'CoolingDown', nextTime: -1, heaterOn: false });
    }
    set(desired) { if (desired !== undefined && desired !== null) this.desired = Number(desired); }
    copy() {
      const c = new Controller(this.desired, this.lowerBound, this.heatingTime, this.heatingGap);
      c.state = this.state; c.nextTime = this.nextTime; c.heaterOn = this.heaterOn; return c;
    }
    step(t, T) {
      if (this.state === 'CoolingDown') {
        if (T <= this.desired - this.lowerBound) { this.state = 'Heating'; this.heaterOn = true; this.nextTime = t + this.heatingTime; }
      } else if (this.state === 'Heating') {
        if (0 < this.nextTime && this.nextTime <= t) { this.state = 'Waiting'; this.heaterOn = false; this.nextTime = t + this.heatingGap; }
        else if (T > this.desired) { this.state = 'CoolingDown'; this.heaterOn = false; this.nextTime = -1; }
      } else if (this.state === 'Waiting') {
        if (0 < this.nextTime && this.nextTime <= t) {
          if (T <= this.desired) { this.state = 'Heating'; this.heaterOn = true; this.nextTime = t + this.heatingTime; }
          else { this.state = 'CoolingDown'; this.heaterOn = false; this.nextTime = -1; }
        }
      }
      return this.heaterOn;
    }
  }

  // ---------- kalman.py ----------
  class Kalman {
    constructor(p, x0, P0 = [0.5, 5.0], q = [0.002, 0.002], r = 0.05) {
      this.st = new Stepper(p);
      this.x = x0.slice();
      this.P = [[P0[0] ** 2, 0], [0, P0[1] ** 2]];
      this.Qrate = [q[0] ** 2, q[1] ** 2];
      this.R = r * r;
    }
    predict(heaterOn, Troom, dt) {
      const { Ad } = this.st.mats(dt);
      this.x = this.st.step(this.x, heaterOn, Troom, dt);
      const AP = matmul(Ad, this.P);
      const At = [[Ad[0][0], Ad[1][0]], [Ad[0][1], Ad[1][1]]];
      const P = matmul(AP, At);
      P[0][0] += this.Qrate[0] * dt; P[1][1] += this.Qrate[1] * dt;
      this.P = P;
    }
    update(y) {
      const innov = y - this.x[0];
      const S = this.P[0][0] + this.R;
      const K = [this.P[0][0] / S, this.P[1][0] / S];
      this.x = [this.x[0] + K[0] * innov, this.x[1] + K[1] * innov];
      const P0 = this.P[0];
      this.P = [[this.P[0][0] - K[0] * P0[0], this.P[0][1] - K[0] * P0[1]],
                [this.P[1][0] - K[1] * P0[0], this.P[1][1] - K[1] * P0[1]]];
      return [innov, S];
    }
  }

  // ---------- plant_sim.py ----------
  function roundHalfEven(x) {
    const r = Math.round(x);
    return (Math.abs(x % 1) === 0.5) ? 2 * Math.round(x / 2) : r;
  }
  const quantize = v => roundHalfEven(v / QSTEP) * QSTEP;
  function gaussianSource(seed, sd) {      // mulberry32 + Box-Muller
    let a = seed >>> 0;
    const rand = () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
                         t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const g = () => { let u = 0; while (u === 0) u = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand()) * sd; };
    return () => [g(), g(), g()];
  }
  class SimBox {
    constructor({ p = PARAMS_2023, T0 = 22, Troom = 22, controller = null, noise = null, seed = 7, sd = 0.05 } = {}) {
      this.setParams(p);
      this.x = [T0, T0]; this.Troom = Troom; this.lidOpen = false; this.t = 0; this.k = 0;
      this.ctrl = controller || new Controller();
      this.noiseAt = noise ? (k => noise[k]) : (() => { const g = gaussianSource(seed, sd); return () => g(); })();
      this.heaterOn = false;
    }
    setParams(p) {
      this.p = Object.assign({}, p);
      this.stClosed = new Stepper(this.p);
      this.stOpen = new Stepper(Object.assign({}, this.p, { G_box: G_OPEN_LID }));
    }
    observe() {
      const n = this.noiseAt(this.k);
      const t1 = quantize(this.x[0] + 0.5 + n[0]), t2 = quantize(this.x[0] - 0.5 + n[1]), t3 = quantize(this.Troom + n[2]);
      const Tm = (t1 + t2) / 2;
      this.heaterOn = this.ctrl.step(this.t, Tm);
      return { t: this.t, t1, t2, t3, T: Tm, T_room: t3, heater_on: this.heaterOn ? 1 : 0,
               true_T: this.x[0], true_T_heater: this.x[1], lid_open: this.lidOpen };
    }
    act(desired) { this.ctrl.set(desired); }
    advance(dt = DT) {
      const st = this.lidOpen ? this.stOpen : this.stClosed;
      this.x = st.step(this.x, this.heaterOn, this.Troom, dt);
      this.t += dt; this.k += 1;
    }
  }

  // ---------- twin.py ----------
  class Twin {
    constructor(p = PARAMS_2023) { this.p = Object.assign({}, p); this.kf = null; this.model = new Stepper(this.p);
                                   this.last = null; this.streak = 0; this.anomaly = false; }
    update(obs) {
      let innov, S;
      if (this.kf === null) { this.kf = new Kalman(this.p, [obs.T, obs.T]); innov = 0; S = this.kf.P[0][0] + this.kf.R; }
      else { this.kf.predict(this.last.heater_on, this.last.T_room, obs.t - this.last.t); [innov, S] = this.kf.update(obs.T); }
      this.streak = Math.abs(innov) > ANOMALY_C ? this.streak + 1 : 0;
      this.anomaly = this.streak >= ANOMALY_N;
      this.last = obs;
      return { t: obs.t, T_est: this.kf.x[0], T_heater_est: this.kf.x[1], T_heater_sd: Math.sqrt(this.kf.P[1][1]),
               innov, anomaly: this.anomaly };
    }
    predict(horizon, controller, desired = null, Troom = null, dt = DT) {
      const c = controller.copy(); c.set(desired);
      let x = this.kf.x.slice();
      const Tr = Troom === null ? this.last.T_room : Troom;
      const n = Math.floor(horizon / dt);
      const T = new Array(n + 1), H = new Array(n + 1).fill(0);
      T[0] = x[0]; let t = this.last.t;
      for (let i = 0; i < n; i++) { H[i] = c.step(t, x[0]) ? 1 : 0; x = this.model.step(x, H[i], Tr, dt); t += dt; T[i + 1] = x[0]; }
      let on = 0; for (let i = 0; i < n; i++) on += H[i];
      return { T, heater: H, energy_Wh: on * dt * this.p.V * this.p.I / 3600 };
    }
  }

  const IDT = { PARAMS_2023, PARAMS_ALT, G_OPEN_LID, DT, ANOMALY_C, ANOMALY_N,
                expm, discretize, Stepper, Controller, Kalman, SimBox, Twin, quantize, gaussianSource };
  if (typeof module !== 'undefined' && module.exports) module.exports = IDT; else root.IDT = IDT;
})(typeof self !== 'undefined' ? self : this);
