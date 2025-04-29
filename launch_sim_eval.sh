TIMESTAMP=$(date +%m%d_%H%M)

gantry run -w ai2/abhayd -b ai2/prior \
    --name "tg_eval_graspgpt_${TIMESTAMP}" \
    --task-name "tg_eval_graspgpt_${TIMESTAMP}" \
    --env-secret WANDB_API_KEY=WANDB_API_KEY \
    --env-secret GITHUB_TOKEN=GITHUB_TOKEN \
    --dataset-secret SSH_KEY:/root/.ssh/id_ed25519 \
    --beaker-image ai2/cuda11.8-dev-ubuntu20.04 \
    --gpus 1 \
    --weka prior-default:/weka/prior \
    --weka oe-training-default:/weka/oe-training-default \
    --priority high \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/saturn-cirrascale \
    --cluster ai2/ceres-cirrascale \
    --install "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 && pip install -r new_requirements.txt" \
    --allow-dirty \
    -- \
    python gcngrasp/run_sim_eval.py
