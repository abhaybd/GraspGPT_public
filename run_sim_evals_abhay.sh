set -euxo pipefail

cp -r /weka/prior/abhayd/semantic-grasping-datasets/graspgpt_checkpoints checkpoints
mkdir -p data
ln -s /weka/prior/abhayd/semantic-grasping-datasets/LA-TaskGrasp data/taskgrasp

python gcngrasp/run_sim_eval.py
