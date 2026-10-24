# Running the LLM agent experiments on a Slurm cluster (HiPerGator example)

`scripts/compare_policy.py` and `scripts/new_conditions.py` talk to a language model served by Ollama. On an
Apple-silicon laptop one model takes 10 min to 2 h per run; on a cluster GPU a run takes 15–54 min. The results in
`results/compare_policy.json` and `results/new_conditions.json` were made this way, on one B200 at the University of
Florida's HiPerGator (paths `/blue/...`, `module load conda`).

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
   bash twins/bikeshare/hpg/run_agents.sh new_conditions.py laqmwe --reasoning none --check
   bash twins/bikeshare/hpg/run_agents.sh new_conditions.py gpt-oss:20b --reasoning low --check
   ```
   Installed models are reused. A missing registry model is downloaded into `/blue/<group>/$USER/ollama/models`;
   the local alias `laqmwe` must already exist and is never pulled from the registry. Results are written to
   `twins/bikeshare/results/` in the clone; commit them from there or copy them back.

Speed: Ollama starts about as many threads as the node has cores, far more than a job gets, and the GPU then mostly
waits. Give each model two threads once (same replies, about ten times faster), and ask for 16 or more CPUs:
```
printf 'FROM gpt-oss:20b\nPARAMETER num_thread 2\n' > Modelfile && ollama create gpt-oss:20b -f Modelfile
```
(the same for the installed alias `laqmwe`, with a server running).

`laqmwe` is the local deployment name for `qwen3.5:9b`; published results retain the original model name. The custom
deployment also uses `laqmwe` for its architecture, tokenizer selector, renderer and parser. It requires the matching
custom Ollama runtime; stock Ollama cannot load that architecture. See [the build and verification notes](laqmwe.md).
The model weights, vocabulary, chat template and upstream license are preserved. Naming does not change provenance.
