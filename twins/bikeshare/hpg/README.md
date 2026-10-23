# Running the LLM agent experiments on a Slurm cluster (HiPerGator example)

`scripts/compare_policy.py` and `scripts/new_conditions.py` talk to a language model served by Ollama. On an
Apple-silicon laptop one model takes 10 min to 2 h per run; on a cluster GPU each reply is faster. The scripts here
use the same server (Ollama) and the same model files as the laptop, so results from either can be compared; another
server (vLLM) or other weights would need every model rerun. Written for the University of Florida's HiPerGator
(paths `/blue/...`, `module load conda`); **untested** on the cluster so far.

## Steps
Replace `<group>` with your HiPerGator group (`sacctmgr -n show assoc user=$USER format=account,qos`; `slurmInfo`
shows whether the group has GPUs).

1. On a login node, clone the repository into `/blue` and set up once (a few minutes):
   ```
   cd /blue/<group>/$USER && git clone https://github.com/CCCC-2024/twin-bench.git && cd twin-bench
   export GROUP=<group>
   bash twins/bikeshare/hpg/setup.sh          # ends with the tests passing and "ready: ..."
   ```
2. Get a GPU node: an interactive session (Open OnDemand "VScode" or "Hipergator Desktop" app with `--gres=gpu:1`,
   32 GB memory), or `srun --account=<group> --qos=<group> --gres=gpu:1 --mem=32gb --time=06:00:00 --pty bash -i`.
   Never run a model on a login node. B200 GPUs (`--partition=hpg-b200`) can wait long in the queue; both models fit
   on the default L4 (24 GB).
3. On that node, from the repository root:
   ```
   export GROUP=<group>
   bash twins/bikeshare/hpg/run_agents.sh new_conditions.py qwen3.5:9b --reasoning none --check
   bash twins/bikeshare/hpg/run_agents.sh new_conditions.py gpt-oss:20b --reasoning low --check
   ```
   The first run of a model downloads it into `/blue/<group>/$USER/ollama/models`. Results are written to
   `twins/bikeshare/results/` in the clone; commit them from there or copy them back.
