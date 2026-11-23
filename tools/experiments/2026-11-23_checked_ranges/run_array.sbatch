#!/bin/bash
#SBATCH --job-name=advise3
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --time=06:00:00
# One run of tools/advise_events.py per array task, each on its own GPU, its own Ollama server and its own copy of the
# code. Submit from the run folder OUT, which holds tasks.txt (one line per task: CODE MODEL REASONING ARM RUN) and,
# for every CODE named there, a folder of that name with the code to run (upstream fetched, no
# results/advise_events.json). ARM is number or range; which range method runs is the code's: `frozen` is this
# repository as at 6363443 with today's tools/advise_events.py, `checked` is it as it is.
#   SPLIT=test3 sbatch --account=<group> --qos=<group> --array=1-$(wc -l < tasks.txt)%8 run_array.sbatch
# Needs GROUP exported (setup as in twins/bikeshare/hpg/README.md). Output: OUT/<code>_<model>_<reasoning>_<arm>_<run>/
# and its .txt summary; progress in OUT/progress.log.
set -u
BASE=/blue/${GROUP:?export GROUP=<your HiPerGator group> first}/$USER
OUT=$SLURM_SUBMIT_DIR
read code model level arm i < <(sed -n "${SLURM_ARRAY_TASK_ID}p" $OUT/tasks.txt)
name=${code}_${model#gpt-oss:}_${level}_${arm}_$i      # the names collect.py reads
# Two tasks can land on the same node: each needs its own port, or one server answers both and stops with its task.
export OLLAMA_MODELS=$BASE/ollama/models OLLAMA_HOST=127.0.0.1:$((11500 + SLURM_JOB_ID % 5000))
export OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1
$BASE/ollama/bin/ollama serve > $OUT/ollama_$name.log 2>&1 &
trap 'kill $!' EXIT
until curl -s $OLLAMA_HOST/api/version > /dev/null; do sleep 1; done
dir=$OUT/$name; rm -rf $dir; cp -r $OUT/$code $dir; cd $dir
extra=""; [ $arm = range ] && extra="--ranges"
echo "$(date +%T) start $name (task $SLURM_ARRAY_TASK_ID, $(hostname))" >> $OUT/progress.log
# The editable install would import the clone's packages, not this copy's: PYTHONPATH makes it this copy.
PYTHONPATH=$dir $BASE/conda/envs/twin-bench/bin/python tools/advise_events.py $model --reasoning $level \
  --max-tokens 16384 --split ${SPLIT:?} $extra --twins incubator bikeshare --url http://$OLLAMA_HOST/v1 \
  > $OUT/$name.txt 2>&1
rc=$?                                       # before any $(...), which would reset $?
echo "$(date +%T) done $name, exit $rc" >> $OUT/progress.log
