import os
import numpy as np
from transformers import logging
from config import get_cfg_defaults
from PIL import Image
from io import BytesIO
from base64 import b64decode
logging.set_verbosity_error()

from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

from inference import GraspGPTPredictor, load_cfg

class GraspPredictionRequest(BaseModel):
    image_str: str
    depth: list[list[float]]
    cam_K: list[list[float]]
    grasps: list[list[list[float]]]
    task: str

MODEL_CFG = "cfg/eval/gcngrasp/gcngrasp_split_mode_t_split_idx_3_.yml"
cfg = load_cfg(MODEL_CFG)
predictor = GraspGPTPredictor(cfg)

app = FastAPI()

@app.post("/api/predict_grasp")
def predict(request: GraspPredictionRequest):
    image = Image.open(BytesIO(b64decode(request.image_str.encode("utf-8")))).convert("RGB")
    depth = np.array(request.depth)
    cam_K = np.array(request.cam_K)
    grasps = np.array(request.grasps)
    task = request.task
    probs, preds = predictor.predict(image, depth, cam_K, grasps, task, verbosity=5)
    if probs is None:
        print("WARN: No object detected in the image")
        return {"grasp_idx": np.random.randint(0, len(grasps)), "probs": [], "preds": []}
    grasp_idx = np.argmax(probs).item()
    if not preds[grasp_idx]:
        print("WARN: All grasps are rejected")
    return {"grasp_idx": grasp_idx, "probs": probs, "preds": preds}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
