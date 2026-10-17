# Running the tumor twin on a Slurm cluster (HiPerGator example)

The training simulations for the AI surrogate are independent runs, so a CPU job array gives the biggest speed-up:
256 runs take about 2 h on a 2-core laptop and about 5 min as a 32-task array. The scripts here are written for
the University of Florida's HiPerGator; on another Slurm cluster change the paths (`/blue/...`) and `module load`.
Items marked **untested** have not been run on the cluster yet.

## What runs
- **Simulator:** TumorTwin, a 3D reaction–diffusion tumor model on a 77 × 116 × 36 voxel grid (PyTorch, RK4 with
  0.5-day steps). One 270-day simulation ≈ 30 s on 2 CPU cores.
- **What we need many of:** simulations at different parameter values (diffusion d, proliferation k, radiotherapy
  sensitivity α_RT, chemo sensitivity α_CT, adjuvant dose multiplier). They train the Gaussian-process surrogate
  in `scripts/surrogate_experiments.py`.
- **GPU (untested):** the code accepts `--device cuda`, but part of each time step is a Python loop over chemo doses,
  which a GPU does not speed up. Run `hpg/sample_gpu.sbatch` once to measure before relying on GPUs.

## HiPerGator facts (docs.rc.ufl.edu, checked 2026-10-13)
- GPUs: NVIDIA L4 (default), RTX PRO 6000 (`--partition=hpg-rtx6000`), B200 (`--partition=hpg-b200` plus
  `--gpus=N`; ~14 CPU cores per B200). Every GPU job needs at least one CPU core per GPU.
- The home directory is limited to 40 GB: keep environments and caches in `/blue`.
- Conda: `module load conda`, create environments with `-p /blue/...`; in job scripts put the environment's `bin`
  on `PATH` instead of activating it.

## Steps
Replace `<user>` with your login and `<group>` with your HiPerGator group (`sacctmgr show assoc user=$USER format=account,qos`).

1. Copy the folder up (from a clone of this repository):
   ```
   rsync -av --exclude upstream --exclude '*.log' twins/tumor/ <user>@hpg.rc.ufl.edu:/blue/<group>/<user>/twin-bench-tumor/
   ```
   `fetch_upstream.sh` calls `../../tools/fetch_github.sh`; copy `tools/` next to it as well, or run the whole repository there.
2. Set up once, on a login node:
   ```
   cd /blue/<group>/$USER/twin-bench-tumor && mkdir -p logs
   export GROUP=<group>
   bash hpg/setup_env.sh                 # ends with: ok <torch version> ...
   ```
3. Sanity check (one run at the true parameters; must match the data):
   ```
   srun --account=$GROUP --qos=$GROUP --cpus-per-task=2 --mem=8gb --time=00:10:00 \
     /blue/$GROUP/$USER/conda/envs/tumortwin/bin/python scripts/sample_sims.py --check
   ```
   Every 3rd value of `Y[0]` in `results/HGG_demo_001/sims_check.npz` should equal `measured_tcc` in
   `results/HGG_demo_001/repro.json` within 0.1 %.
4. Training simulations (256 runs, 32 tasks):
   ```
   sbatch --account=$GROUP --qos=$GROUP hpg/sample_array.sbatch
   sbatch --account=$GROUP --qos=$GROUP --cpus-per-task=2 --mem=8gb --time=00:30:00 --output=logs/truth_%j.log \
     --wrap "PATH=/blue/$GROUP/$USER/conda/envs/tumortwin/bin:\$PATH python scripts/sample_sims.py --truth"
   ```
   More runs: `sbatch --account=$GROUP --qos=$GROUP --export=ALL,N=1024,PER=8 --array=0-127 hpg/sample_array.sbatch`.
   Another patient: add `PATIENT=HGG_demo_002`.
5. Merge and run the experiments:
   ```
   python scripts/merge_sims.py --seed 0
   python scripts/surrogate_experiments.py      # -> results/HGG_demo_001/surrogate.json
   ```
6. Bring the results back: `rsync -av <user>@hpg.rc.ufl.edu:/blue/<group>/<user>/twin-bench-tumor/results/ twins/tumor/results/`
7. Optional GPU timing test: `sbatch --account=$GROUP --qos=$GROUP hpg/sample_gpu.sbatch`, then compare the time per
   run in `logs/gpu_test_*.log` with ~30 s per run on CPU.

Do not change the parameter ranges or the seed in `sample_sims.py` between batches: the surrogate and the
experiments assume one design.
