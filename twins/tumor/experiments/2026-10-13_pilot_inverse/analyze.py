"""Pilot: can a small network read (d,k,aRT,aCT) from per-visit summaries of the first K MRIs?"""
import glob, json, sys, numpy as np, warnings
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
Ps, Fs = [], []
for f in sorted(glob.glob(sys.argv[1] + "/vp_*.npz")):
    z = np.load(f); Ps.append(z["P"]); Fs.append(z["F"])
P0, F0 = Ps[0][0], Fs[0][0]            # measured HGG_demo_001 (truth 0.1,0.05,0.05,0.2)
P = np.vstack([Ps[0][1:]] + Ps[1:]); F = np.vstack([Fs[0][1:]] + Fs[1:])
n = len(P); names = ["d", "k", "aRT", "aCT"]
print(f"virtual patients: {n}")
def X(F, K, cols):  # log of features for first K visits
    return np.log1p(F[:, 1:K, cols].reshape(len(F), -1))  # visit 0 is identical for all (same initial map) -> drop
def model():
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 32), alpha=1e-3, max_iter=5000, random_state=0))
def cv(K, cols, noise=0.0, seed=0, ntrain=None):
    Y = np.log(P); pred = np.zeros_like(Y); rng = np.random.default_rng(seed)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(P):
        tr = tr[:ntrain] if ntrain else tr
        Ftr, Ytr = F[tr], Y[tr]
        if noise > 0:   # train on noisy copies too
            Ftr = np.concatenate([Ftr] + [Ftr * (1 + noise * rng.standard_normal(Ftr.shape)) for _ in range(5)]); Ytr = np.tile(Ytr, (6, 1))
        m = model().fit(X(Ftr, K, cols), Ytr)
        Ft = F[te] * (1 + noise * rng.standard_normal(F[te].shape))
        pred[te] = m.predict(X(Ft, K, cols))
    rel = np.abs(np.exp(pred) - P) / P * 100
    return np.median(rel, 0)
res = {}
for label, cols in [("TCC only", [0]), ("TCC+volumes+radius", [0, 1, 2, 3, 4])]:
    for K in [2, 3, 4, 5]:
        for noise in [0.0, 0.05]:
            r = cv(K, cols, noise); res[f"{label}|K={K}|noise={noise}"] = r.round(1).tolist()
            print(f"{label:20s} K={K} noise={noise:4.2f}  median rel err % (d,k,aRT,aCT): {np.round(r,1)}")
for nt in [18, 36, 54, 72]:
    r = cv(5, [0, 1, 2, 3, 4], ntrain=nt); res[f"learning|ntrain={nt}"] = r.round(1).tolist()
    print(f"learning curve ntrain={nt}: {np.round(r,1)}")
# apply to the measured patient
for label, cols in [("TCC only", [0]), ("TCC+volumes+radius", [0, 1, 2, 3, 4])]:
    m = model().fit(X(F, 5, cols), np.log(P))
    p = np.exp(m.predict(X(F0[None], 5, cols))[0])
    res[f"HGG_demo_001|{label}"] = p.round(4).tolist()
    print(f"HGG_demo_001 {label}: AI={np.round(p,4)} truth={P0} relerr%={np.round(np.abs(p-P0)/P0*100,1)}")
json.dump(res, open(sys.argv[1] + "/pilot_results.json", "w"), indent=1)
