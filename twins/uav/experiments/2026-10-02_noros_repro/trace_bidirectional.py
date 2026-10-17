"""Make both directions of the twin<->asset loop visible.
A) step trace: action sent (twin->asset), readings returned (asset->twin), belief before/after.
B) does the action change the asset's true damage? default scripted asset vs action-dependent asset."""
import sys, numpy as np, contextlib, io
REPO = sys.argv[1] if len(sys.argv) > 1 else '../../upstream/UAV-Digital-Twin'
sys.argv = ['x', '--repo', REPO, '--out', 'tmp_tr']
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('kapteyn_noros_driver.py').read().split("rows = run(args.seed")[0])
idx1 = lambda lvl: np.array([s[0] >= lvl for s in S])

def loop(seed, steps, asset_mode='custom', force=None, trace=()):
    np.random.seed(seed); asset = UAV(config_fpath=CFG_PATH); rng = np.random.RandomState(seed + 999)
    x = [0., 0.]; ta = -1; b_pred = np.zeros(25); b_pred[0] = 1; hist = []
    for k in range(steps):
        u = force if force is not None else int(np.argmax([b_pred[POL == a].sum() for a in range(2)]))
        m = np.array(asset.measurementGenerator.getMeasurement(x, [u]))        # asset -> twin
        b_post = b_pred * likelihood(m, u); b_post /= b_post.sum()
        if k in trace:
            top = lambda b: ', '.join('%s:%.2f' % (S[i], b[i]) for i in np.argsort(-b)[:2])
            print(f'k={k:2d} | twin->asset: {C[u]} | asset true=({20*x[0]:.0f},{20*x[1]:.0f}) | '
                  f'asset->twin: 24 strains, mean={m.mean():.0f} | belief before [{top(b_pred)}] '
                  f'-> after [{top(b_post)}] | P(comp1>=60) {b_pred[idx1(60)].sum():.2f}->{b_post[idx1(60)].sum():.2f}')
        hist.append((k, C[u], 20*x[0], 20*x[1]))
        ta += 1
        if asset_mode == 'custom':
            x = asset_custom_step(x, ta)                                          # ignores u
        else:  # action-dependent: same rule the twin assumes (2g p=.05, 3g p=.1 per component)
            p = cfg['transition_probabilities'][C[u]]
            for d in range(2):
                if x[d] < 4 and rng.rand() < p: x[d] += 1
        b_pred = b_post @ TM[u]; b_pred /= b_pred.sum()
    return hist

print('=== A) step trace, default asset, seed 0 (around the 3g->2g switch) ===')
loop(0, 34, trace=range(28, 34))

print('\n=== B1) default scripted asset: does the action change true damage? ===')
h3 = loop(0, 60, force=1); h2 = loop(0, 60, force=0)
print('always-3g vs always-2g identical true damage path:', [r[2:] for r in h3] == [r[2:] for r in h2])

print('\n=== B2) action-dependent asset (damage growth depends on 2g/3g), 200 seeds x 60 steps ===')
def summarise(policy):
    out = []
    for s in range(200):
        h = loop(s, 60, asset_mode='dep', force=policy)
        t60 = next((r[0] for r in h if r[2] >= 60), 60); f = h[-1]
        out.append((t60, f[2], f[3], sum(r[1] == '3g' for r in h)))
    a = np.array(out, float)
    return 'steps until comp1>=60: %.1f | final damage comp1 %.0f%% comp2 %.0f%% | 3g steps %.1f' % tuple(a.mean(0))
print('twin closed loop :', summarise(None))
print('always 3g        :', summarise(1))
print('always 2g        :', summarise(0))
