"""Lock-step cross-check: author's pomegranate GraphicalModel vs hand-written Bayes filter.
Both twins see the same observations (generated under the author's chosen action).
Run from UAV-Digital-Twin repo root (author code uses relative config path)."""
import sys, json, time, numpy as np
from scipy.stats import norm
sys.path.insert(0, 'src/digitaltwin')
import contextlib, io
from digitaltwin.graphicalmodel import GraphicalModel
from digitaltwin.planner import Planner
from digitaltwin.UAV import UAV
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
np.random.seed(0)
q = io.StringIO()
with contextlib.redirect_stdout(q):
    gm = GraphicalModel("DT"); planner = Planner(gm); gm.policy = planner.getPolicy()
    gm.process_new_state(); gm.compute_marginals()
cfg = gm.config; S = cfg['flat_states']; C = cfg['controls']; LOAD = {'2g':2.,'3g':3.}
POL = np.array([gm.policy(s) for s in S])
def Tm(c):
    p = cfg['transition_probabilities'][c]; T = np.zeros((25,25))
    for i,a in enumerate(S):
        for j,b in enumerate(S):
            d1,d2 = b[0]-a[0], b[1]-a[1]
            if a==(80,80) and b==(80,80): T[i,j]=1
            elif d1==d2==0: T[i,j]=(1-p)**2
            elif d1==20 and d2==20: T[i,j]=p*p
            elif (d1,d2) in [(20,0),(0,20)]: T[i,j]=p*(1-p)
    return T
TM=[Tm(c) for c in C]
MEANS={c: np.array([np.array(cfg['observations'][str(s[0])][str(s[1])][c]['mean'])/LOAD[c] for s in S]) for c in C}
def lik(m,u):
    ll = norm.logpdf(np.asarray(m)[None,:], MEANS[C[u]], 125/np.sqrt(LOAD[C[u]])).sum(1); w=np.exp(ll-ll.max()); return w/w.sum()
def author_post(t):  # joint over 25 states of Damage t
    mat = gm.state_joints['Damage {}'.format(t)]
    return np.array([mat[cfg['states'][0].index(s[0]), cfg['states'][1].index(s[1])] for s in S])
def author_action(t): return int(np.argmax(gm.marginals['ControlP {}'.format(t)][0]))
asset = UAV(); x=[0.,0.]; ta=-1
b_pred = np.zeros(25); b_pred[0]=1
rows=[]
for t in range(STEPS):
    ua = author_action(t)
    mine_u = int(np.argmax([b_pred[POL==u].sum() for u in range(2)]))
    with contextlib.redirect_stdout(q):
        gm.process_new_control([ua]); gm.prepare_prediction(t+10)
    m = asset.measurementGenerator.getMeasurement(x,[ua])
    ta+=1
    if ta in [20,30]: x[0]+=1
    if 10<=ta<20: x[0]+=0.1
    if ta in [5,25]: x[1]+=1
    if 5<=ta<15: x[1]+=0.1
    t0=time.time()
    with contextlib.redirect_stdout(q):
        gm.process_new_observation(m); gm.process_new_state(); gm.compute_marginals()
    dt=time.time()-t0
    pa = author_post(t)                      # author's posterior of D_t
    b_post = b_pred*lik(m,ua); b_post/=b_post.sum()
    tv = 0.5*np.abs(pa/pa.sum()-b_post).sum()
    rows.append(dict(t=t, action_author=C[ua], action_mine=C[mine_u], tv_distance=float(tv), sec=round(dt,2)))
    b_pred = b_post@TM[ua]; b_pred/=b_pred.sum()
    print(rows[-1], flush=True)
json.dump(rows, open(sys.argv[2] if len(sys.argv)>2 else 'crosscheck.json','w'), indent=1)
print('max TV distance', max(r['tv_distance'] for r in rows), '| action disagreements', sum(r['action_author']!=r['action_mine'] for r in rows))
