import sys, numpy as np, json, contextlib, io
sys.argv=['x','--repo','../../upstream/UAV-Digital-Twin','--out','tmp_an']
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('kapteyn_noros_driver.py').read().split("rows = run(args.seed")[0])
# 1) sensor discriminability per component: mean |Δobs|/sigma between neighbouring grid states
for c in C:
    sig = SIG/np.sqrt(LOAD[c]); d1=[];d2=[]
    for i,s in enumerate(S):
        if s[0]<80: j=S.index((s[0]+20,s[1])); d1.append(np.linalg.norm(MEANS[c][j]-MEANS[c][i])/sig)
        if s[1]<80: j=S.index((s[0],s[1]+20)); d2.append(np.linalg.norm(MEANS[c][j]-MEANS[c][i])/sig)
    print(c,'separation (Mahalanobis, 24ch) one grid step: comp1 median %.1f  comp2 median %.1f'%(np.median(d1),np.median(d2)))
s=S.index((60,60)); 
for c in C:
    sig=SIG/np.sqrt(LOAD[c]); print(c,'(60,60)->(60,80) sep %.1f ; (60,60)->(80,60) sep %.1f'%(np.linalg.norm(MEANS[c][S.index((60,80))]-MEANS[c][s])/sig, np.linalg.norm(MEANS[c][S.index((80,60))]-MEANS[c][s])/sig))
# 2) frequency of estimation error vs decision error over 20 seeds
def stats(runs):
    e1=np.array([[abs(r['est1']-r['true1']) for r in R] for R in runs]); e2=np.array([[abs(r['est2']-r['true2']) for r in R] for R in runs])
    mis=np.array([[r['u']!=r['oracle_u'] for r in R] for R in runs])
    return dict(p_err1_gt10=float((e1>10).mean()), p_err2_gt10=float((e2>10).mean()), p_decision_mismatch=float(mis.mean()),
                late_err2_mean=float(e2[:,40:].mean()), late_bias2=float(np.mean([[r['est2']-r['true2'] for r in R[40:]] for R in runs])))
base=[run(s,60) for s in range(1,21)]
print('default twin:',stats(base))
# 3) attribution test: same asset, twin transition probs scaled down (closer to scripted truth that stops growing)
orig=dict(cfg['transition_probabilities'])
for f in [0.2,0.0]:
    for c in C: cfg['transition_probabilities'][c]=orig[c]*f
    TM[:]=[T_matrix(c) for c in C]
    print('twin p x%.1f:'%f, stats([run(s,60) for s in range(1,21)]))
for c in C: cfg['transition_probabilities'][c]=orig[c]
TM[:]=[T_matrix(c) for c in C]
# 4) when do decision mismatches happen
ks=[r['k'] for R in base for r in R if r['u']!=r['oracle_u']]
print('mismatch timesteps:',sorted(set(ks)), 'count',len(ks))
