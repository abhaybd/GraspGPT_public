set -euxo pipefail

cp -r /weka/prior/abhayd/semantic-grasping-datasets/graspgpt_checkpoints checkpoints
mkdir -p data
ln -s /weka/prior/abhayd/semantic-grasping-datasets/LA-TaskGrasp data/taskgrasp

python gcngrasp/run_tg_eval.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_t_split_idx_0_.yml
python gcngrasp/run_tg_eval.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_t_split_idx_1_.yml
python gcngrasp/run_tg_eval.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_t_split_idx_2_.yml
python gcngrasp/run_tg_eval.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_t_split_idx_3_.yml
python gcngrasp/run_tg_eval.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_o_split_idx_0_.yml
python gcngrasp/run_tg_eval.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_o_split_idx_1_.yml
python gcngrasp/run_tg_eval.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_o_split_idx_2_.yml
python gcngrasp/run_tg_eval.py --cfg_file cfg/eval/gcngrasp/gcngrasp_split_mode_o_split_idx_3_.yml
python gcngrasp/results/aggregate_results.py