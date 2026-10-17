# Run from twins/uav/experiments/2026-10-02_noros_repro: python metrics_1007.py  -> metrics_1007.json (20 seeds x 60 steps; twin transition prob x1.0/0.2/0.0)
import sys, numpy as np, json, contextlib, io, warnings
warnings.filterwarnings('ignore')
sys.argv=['x','--repo','../../upstream/UAV-Digital-Twin','--out','tmp_an']
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('kapteyn_noros_driver.py').read().split("rows = run(args.seed")[0])
def stats(runs):
    e1=np.array([[abs(r['est1']-r['true1']) for r in R] for R in runs]); e2=np.array([[abs(r['est2']-r['true2']) for r in R] for R in runs])
    mis=np.array([[r['u']!=r['oracle_u'] for r in R] for R in runs])
    return dict(any=float(((e1>10)|(e2>10)).mean()), r1=float((e1>10).mean()), r2=float((e2>10).mean()), dec=float(mis.mean()), n=int(mis.size))
out={}
orig=dict(cfg['transition_probabilities'])
for f in [1.0,0.2,0.0]:
    for c in C: cfg['transition_probabilities'][c]=orig[c]*f
    TM[:]=[T_matrix(c) for c in C]
    out[f]=stats([run(s,60) for s in range(1,21)])
    print(f, out[f])
json.dump({str(k):v for k,v in out.items()}, open('metrics_1007.json','w'), indent=1)
