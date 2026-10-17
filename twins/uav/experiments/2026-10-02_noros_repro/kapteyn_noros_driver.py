"""Minimal no-ROS driver for Kapteyn UAV-Digital-Twin (commit b51e42d).

Reuses unmodified UAV.py (asset measurement generator), planner.py (value-iteration
policy) and UAVconfig.json. Replaces graphicalmodel.py (pomegranate PGM) with a
hand-written 25-state discrete Bayes filter that mirrors its factors:
  prior D0=(0,0); transition p(D_t|D_t-1,U_t-1) with 2g=0.05 / 3g=0.1 per component;
  observation likelihood N(m | mean/load, 125/sqrt(load)) per channel;
  action = argmax_u sum_s b_pred(s) * 1[policy(s)=u]  (author's ControlP marginal argmax).
Asset follows nodes/asset.py 'custom' scripted damage (not driven by control).
Usage: python kapteyn_noros_driver.py --repo <UAV-Digital-Twin> --steps 60 --seed 0
"""
import argparse, json, os, sys
import numpy as np
from scipy.stats import norm

ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True)
ap.add_argument('--steps', type=int, default=60)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--seeds', type=int, default=0, help='extra Monte Carlo seeds for summary')
ap.add_argument('--out', default='.')
args = ap.parse_args()

sys.path.insert(0, os.path.join(args.repo, 'src', 'digitaltwin'))
from digitaltwin.UAV import UAV            # unmodified
from digitaltwin.planner import Planner    # unmodified
from digitaltwin.utils import read_json_file, flatten_list

CFG_PATH = os.path.join(args.repo, 'src/digitaltwin/inputfiles/UAVconfig.json')

class GMShim:
    """Only what Planner needs from GraphicalModel (copied semantics)."""
    def __init__(self):
        self.config = read_json_file(CFG_PATH)
        self.config['flat_states'] = flatten_list(self.config['states'])
        self.n_samples = 30
    def health_reward_function(self, ref_obs):
        return 1.0 - (np.max(ref_obs) / 2500)
    def control_reward_function(self, control):
        return 0.2 * control - 0.1

gm = GMShim()
cfg = gm.config
S = cfg['flat_states']                      # 25 tuples in percent
C = cfg['controls']                         # ['2g','3g']
LOAD = {'2g': 2.0, '3g': 3.0}
policy = Planner(gm).getPolicy()
POL = np.array([policy(s) for s in S])      # action index per state

def T_matrix(control):
    p = cfg['transition_probabilities'][control]
    T = np.zeros((25, 25))
    for i, s1 in enumerate(S):
        for j, s2 in enumerate(S):
            d1, d2 = s2[0]-s1[0], s2[1]-s1[1]
            if s1 == (80, 80) and s2 == (80, 80): T[i, j] = 1.0
            elif d1 == d2 == 0: T[i, j] = (1-p)*(1-p)
            elif d1 == 20 and d2 == 20: T[i, j] = p*p
            elif d1 == 20 and d2 == 0: T[i, j] = p*(1-p)
            elif d1 == 0 and d2 == 20: T[i, j] = p*(1-p)
    return T
TM = [T_matrix(c) for c in C]
# NB: author factor gives states at 80 on one axis no "stay" mass beyond (1-p)^2 -> rows may not sum to 1; normalise like pomegranate would not. Keep raw but renormalise belief.
MEANS = {c: np.array([np.array(cfg['observations'][str(s[0])][str(s[1])][c]['mean'])/LOAD[c] for s in S]) for c in C}
SIG = 125.0

def likelihood(m, cidx):
    c = C[cidx]
    ll = norm.logpdf(np.asarray(m)[None, :], MEANS[c], SIG/np.sqrt(LOAD[c])).sum(1)
    w = np.exp(ll - ll.max())
    return w / w.sum()

def asset_custom_step(x, t):           # nodes/asset.py, state_transition='custom'
    x = list(x)
    if t in [20, 30]: x[0] += 1
    if 10 <= t < 20: x[0] += 0.1
    if t in [5, 25]: x[1] += 1
    if 5 <= t < 15: x[1] += 0.1
    return x

def run(seed, steps, log=False):
    np.random.seed(seed)
    asset = UAV(config_fpath=CFG_PATH)
    x = [0.0, 0.0]; t_asset = -1
    b_pred = np.zeros(25); b_pred[0] = 1.0           # prior D0=(0,0)
    rows = []
    for k in range(steps):
        # twin chooses action from predicted belief (ControlP marginal argmax)
        pu = np.array([b_pred[POL == u].sum() for u in range(2)])
        u = int(np.argmax(pu))
        # asset: measurement at current true state under u, then advance truth
        m = asset.measurementGenerator.getMeasurement(x, [u])
        x_true = list(x)
        t_asset += 1; x = asset_custom_step(x, t_asset)
        # twin: Bayes update of D_k, then predict D_{k+1}
        b_post = b_pred * likelihood(m, u); b_post /= b_post.sum()
        b_next = b_post @ TM[u]; b_next /= b_next.sum()
        # diagnostics
        true_pct = [20*x_true[0], 20*x_true[1]]
        mean_est = [float(sum(b_post[i]*S[i][d] for i in range(25))) for d in range(2)]
        map_s = S[int(np.argmax(b_post))]
        near = tuple(int(20*round(v)) for v in x_true)       # nearest grid state to truth
        oracle_u = int(POL[S.index(near)])
        rows.append(dict(k=k, u=C[u], oracle_u=C[oracle_u], true1=true_pct[0], true2=true_pct[1],
                         est1=mean_est[0], est2=mean_est[1], map1=map_s[0], map2=map_s[1],
                         pmax=float(b_post.max()), pu3g_pred=float(pu[1])))
        b_pred = b_next
    return rows

rows = run(args.seed, args.steps)
os.makedirs(args.out, exist_ok=True)
json.dump(dict(seed=args.seed, steps=args.steps, policy={str(s): C[POL[i]] for i, s in enumerate(S)}, rows=rows),
          open(os.path.join(args.out, f'run_seed{args.seed}.json'), 'w'), indent=1)

def summary(rows):
    e1 = np.array([abs(r['est1']-r['true1']) for r in rows]); e2 = np.array([abs(r['est2']-r['true2']) for r in rows])
    mis = np.mean([r['u'] != r['oracle_u'] for r in rows])
    mapok = np.mean([(r['map1'], r['map2']) == (int(20*round(r['true1']/20)), int(20*round(r['true2']/20))) for r in rows])
    return dict(mae1=float(e1.mean()), mae2=float(e2.mean()), map_match=float(mapok), action_mismatch=float(mis),
                n3g=int(sum(r['u']=='3g' for r in rows)))
print('seed', args.seed, summary(rows))
if args.seeds:
    S_ = [summary(run(s, args.steps)) for s in range(1, args.seeds+1)]
    agg = {k: (float(np.mean([d[k] for d in S_])), float(np.std([d[k] for d in S_]))) for k in S_[0]}
    print(f'{args.seeds} seeds mean/std:', agg)
    json.dump(dict(per_seed=S_, agg=agg), open(os.path.join(args.out, 'mc_summary.json'), 'w'), indent=1)

# plot
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
k = [r['k'] for r in rows]
fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True, gridspec_kw=dict(height_ratios=[3, 3, 1]))
for i, (tk, ek, mk) in enumerate([('true1', 'est1', 'map1'), ('true2', 'est2', 'map2')]):
    ax[i].plot(k, [r[tk] for r in rows], color='#222', lw=2, label='asset truth (scripted)')
    ax[i].plot(k, [r[ek] for r in rows], color='#2a6fdb', lw=1.5, label='twin posterior mean')
    ax[i].step(k, [r[mk] for r in rows], where='mid', color='#e07b00', lw=1, ls='--', label='twin MAP')
    ax[i].set_ylabel(f'component {i+1} damage (%)'); ax[i].set_ylim(-5, 85); ax[i].grid(alpha=.3)
ax[0].legend(loc='upper left', fontsize=8)
ax[2].step(k, [1 if r['u']=='3g' else 0 for r in rows], where='mid', color='#2a6fdb', label='twin action')
ax[2].step(k, [1.05 if r['oracle_u']=='3g' else 0.05 for r in rows], where='mid', color='#222', ls=':', label='policy(true state)')
ax[2].set_yticks([0, 1]); ax[2].set_yticklabels(['2g', '3g']); ax[2].set_xlabel('timestep'); ax[2].legend(fontsize=8, loc='center right')
fig.suptitle(f'Kapteyn UAV DT core loop without ROS (seed {args.seed}, default "custom" asset)', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(args.out, f'belief_vs_truth_seed{args.seed}.png'), dpi=150)
print('saved')
