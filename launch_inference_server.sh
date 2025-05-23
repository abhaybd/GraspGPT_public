#!/bin/bash

beaker session create -w ai2/abhayd --budget ai2/prior \
    --name "graspgpt_tmp_inference_server" \
    --mount src=weka,ref=prior-default,dst=/weka/prior \
    --mount src=weka,ref=oe-training-default,dst=/weka/oe-training-default \
    --secret-env OPENAI_API_KEY=OPENAI_API_KEY \
    --workdir /weka/prior/abhayd/GraspGPT_public \
    --detach \
    --bare \
    --image beaker://ai2/cuda11.8-dev-ubuntu20.04 \
    --priority high \
    --gpus 1 \
    --cluster ai2/neptune-cirrascale \
    --port 8080 \
    -- \
    /bin/bash -c "/weka/prior/abhayd/envs/graspgpt/bin/python gcngrasp/inference_server.py ; /bin/bash"
