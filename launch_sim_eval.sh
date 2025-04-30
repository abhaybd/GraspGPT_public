TIMESTAMP=$(date +%m%d_%H%M)

gantry run -w ai2/molmo-act -b ai2/prior \
    --name "sim_eval_graspgpt_${TIMESTAMP}" \
    --task-name "sim_eval_graspgpt_${TIMESTAMP}" \
    --env-secret WANDB_API_KEY=ABHAYD_WANDB_API_KEY \
    --env-secret OPENAI_API_KEY=ABHAYD_OPENAI_API_KEY \
    --env-secret GITHUB_TOKEN=ABHAYD_GITHUB_TOKEN \
    --dataset-secret ABHAYD_SSH_KEY:/root/.ssh/id_ed25519 \
    --beaker-image ai2/cuda11.8-dev-ubuntu20.04 \
    --gpus 1 \
    --weka prior-default:/weka/prior \
    --weka oe-training-default:/weka/oe-training-default \
    --priority urgent \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/saturn-cirrascale \
    --cluster ai2/ceres-cirrascale \
    --install "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 && pip install -r new_requirements.txt && pip install git+https://github.com/facebookresearch/sam2.git" \
    --allow-dirty \
    -- \
    bash run_sim_evals_abhay.sh
