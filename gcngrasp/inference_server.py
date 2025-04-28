import argparse
import os
import tqdm
import time
import random
import sys
from openai import OpenAI
import torch
import numpy as np
import torch.nn.functional as F
from models.graspgpt_plain import GraspGPT_plain
from transformers import BertTokenizer, BertModel, logging
from data.SGNLoader import pc_normalize
from config import get_cfg_defaults
from geometry_utils import farthest_grasps, regularize_pc_point_count
from visualize import save_scene, get_gripper_control_points
from PIL import Image
from io import BytesIO
logging.set_verbosity_error()

from base64 import b64encode, b64decode

from mask_detection import MaskDetector

DEVICE = "cuda"
CODE_DIR = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(CODE_DIR)
from data_specification import OBJ_PROMPTS, TASK_PROMPTS

SYS_PROMPT = """
You are a helpful robotic assistant. You are given an image, and a description of a task to be performed, that involves grasping an object.
You should identify the object to be grasped in a few words, and also a single verb that describes the task to be performed.
You should respond in JSON format, with the following fields:
- obj_class: the object to be grasped in at most a few words without any articles, e.g. "apple".
- task_verb: a single verb that describes the task to be performed, e.g. "cut".
"""

openai_client = OpenAI()

def gpt(text):
    """
    OpenAI GPT API
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": text}],
        max_tokens=256,
    )

    return response.choices[0].message.content.strip()

def encode_text(text, tokenizer, model, device, type=None):
    """
    Language data encoding with a Google pre-trained BERT
    """
    if type == 'od':
        encoded_input = tokenizer(text, return_tensors='pt', padding="max_length", max_length=300).to(device)
    elif type == 'td':
        encoded_input = tokenizer(text, return_tensors='pt', padding="max_length", max_length=200).to(device)
    elif type == 'li':
        encoded_input = tokenizer(text, return_tensors='pt', padding="max_length", max_length=21).to(device)
    else:
         raise ValueError(f'No such language embedding type: {type}')
    
    with torch.no_grad():
        output = model(**encoded_input)
        word_embedding = output[0]
        sentence_embedding = torch.mean(output[0], dim=1)
    
    return word_embedding, sentence_embedding, encoded_input['attention_mask']

def gen_gpt_desc(class_label, task_label):
    """
    Generate object class and task descriptions
    """        
    class_keys = [random.choice(['shape', 'geometry']), random.choice(["use", "func"]), 
    random.choice(["sim_shape", "sim_geo"]), random.choice(["sim_use", "sim_func"])]
    task_keys = [random.choice(['func', 'use']), "sim_effect", random.choice(['sem_verb', 'sim_verb'])]

    print("\nGenerating object class description ......\n")
    class_desc = []
    for c_key in class_keys:
        prompt = OBJ_PROMPTS[c_key]
        prompt = prompt.replace('OBJ_CLASS', class_label)
        temp_ans = gpt(prompt)
        print(f"[{c_key}] "+temp_ans)
        class_desc.append(temp_ans)
    class_desc = ' '.join(item for item in class_desc)
    
    print("\nGenerating task description ......\n")
    task_desc = []
    for t_key in task_keys:
        prompt = TASK_PROMPTS[t_key]
        prompt = prompt.replace('TASK_CLASS', task_label)
        temp_ans = gpt(prompt)
        print(f"[{t_key}] "+temp_ans)
        task_desc.append(temp_ans)
    task_desc = ' '.join(item for item in task_desc)

    return class_desc, task_desc

def load_model(cfg):
    """
    Load GraspGPT pre-trained weight from checkpoint
    """

    model = GraspGPT_plain(cfg)
    model_weights = torch.load(
        cfg.weight_file,
        map_location=DEVICE)['state_dict']
    
    model.load_state_dict(model_weights)
    model = model.to(DEVICE)
    model.eval()

    return model
    
def test(model, pc, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask):   
    pc = pc.type(torch.cuda.FloatTensor)
    with torch.no_grad():
        logits = model(pc, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask)
    logits = logits.squeeze()
    probs = torch.sigmoid(logits)
    preds = torch.round(probs)
    return probs, preds

def img_to_pc(rgb: np.ndarray, depth: np.ndarray, cam_info: np.ndarray, mask: np.ndarray | None = None):
    h, w = rgb.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    depth_mask = (depth > 0)
    if mask is None:
        mask = np.ones_like(depth, dtype=bool)
    uvd = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)
    uvd *= np.expand_dims(depth, axis=-1)
    uvd = uvd[depth_mask & mask]
    xyz = np.linalg.solve(cam_info, uvd.T).T
    return np.concatenate([xyz, rgb[depth_mask & mask]], axis=-1)

class GraspGPTPredictor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = load_model(cfg)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_model = self.bert_model.to(DEVICE)
        self.bert_model.eval()

        self.mask_detector = MaskDetector()

    def _get_pc(self, image: Image.Image, depth: np.ndarray, cam_K: np.ndarray, obj_class: str):
        mask = self.mask_detector.detect_mask(obj_class, image)
        if mask is None:
            return None
        rgb = np.asarray(image)
        pc = img_to_pc(rgb, depth, cam_K, mask)
        return pc

    def preprocess_grasps(self, grasps: np.ndarray):
        grasp_trf = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0.09],
            [0, 0, 0, 1]
        ])
        grasps = grasps @ grasp_trf[None]
        return grasps

    def _get_task_metadata(self, task: str, image: Image.Image):
        class TaskMetadata(BaseModel):
            obj_class: str
            task_verb: str

        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        enc_image = b64encode(image_bytes.getvalue()).decode("utf-8")
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The task is: {task}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{enc_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            response_format=TaskMetadata
        )
        metadata: TaskMetadata = response.choices[0].message.parsed
        print(f"Task metadata: {metadata.model_dump_json()}")
        return metadata.obj_class, metadata.task_verb

    def predict(self, image: Image.Image, depth: np.ndarray, cam_K: np.ndarray, grasps: np.ndarray, task_ins_txt: str):
        obj_class, task_verb = self._get_task_metadata(task_ins_txt, image)
        pc_input = self._get_pc(image, depth, cam_K, obj_class)
        if pc_input is None:
            return None, None
        grasps = self.preprocess_grasps(grasps)
        pc_input = regularize_pc_point_count(pc_input, self.cfg.num_points, use_farthest_point=False)

        # language descriptions
        obj_desc_txt, task_desc_txt = gen_gpt_desc(obj_class, task_verb)
        print(f"Object description: {obj_desc_txt}")
        print(f"Task description: {task_desc_txt}")
        obj_desc, _, obj_desc_mask = encode_text(obj_desc_txt, self.tokenizer, self.bert_model, DEVICE, type='od')
        task_desc, _, task_desc_mask = encode_text(task_desc_txt, self.tokenizer, self.bert_model, DEVICE, type='td')
        # language instruciton
        task_ins, _, task_ins_mask = encode_text(task_ins_txt, self.tokenizer, self.bert_model, DEVICE, type='li')

        probs = []
        preds = []
        for i in range(len(grasps)):
            grasp = grasps[i]
            pc = pc_input[:, :3]
            grasp_pc = get_gripper_control_points()
            grasp_pc = np.matmul(grasp, grasp_pc.T).T  # transform grasps
            grasp_pc = grasp_pc[:, :3]  # remove latent indicator

            latent = np.concatenate(
                [np.zeros(pc.shape[0]), np.ones(grasp_pc.shape[0])])  # create latent indicator
            latent = np.expand_dims(latent, axis=1)
            pc = np.concatenate([pc, grasp_pc], axis=0)  # [*, 3]

            pc, grasp = pc_normalize(pc, grasp, pc_scaling=self.cfg.pc_scaling)
            pc = np.concatenate([pc, latent], axis=1)  # add back latent indicator

            pc = torch.as_tensor(pc[None])
            prob, pred = test(self.model, pc, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask)
            probs.append(prob.tolist())
            preds.append(pred.tolist())
        probs = np.array(probs).flatten().tolist()
        preds = np.array(preds).flatten().tolist()
        assert len(probs) == len(grasps) and len(preds) == len(grasps)
        return probs, preds

def load_cfg(cfg_path: str):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    if cfg.base_dir != '':
        if not os.path.exists(cfg.base_dir):
            raise FileNotFoundError(
                'Provided base dir {} not found'.format(
                    cfg.base_dir))
    else:
        assert cfg.base_dir == ''
        cfg.base_dir = os.path.join(os.path.dirname(__file__), '../data')
    experiment_dir = os.path.join(cfg.log_dir, cfg.weight_file)

    weight_files = os.listdir(os.path.join(experiment_dir, 'weights'))
    assert len(weight_files) == 1
    cfg.weight_file = os.path.join(experiment_dir, 'weights', weight_files[0])
    return cfg


from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

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
    probs, preds = predictor.predict(image, depth, cam_K, grasps, task)
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
