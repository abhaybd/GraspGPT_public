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
import pdb
logging.set_verbosity_error()

DEVICE = "cuda"
CODE_DIR = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(CODE_DIR)
from data_specification import TASKS, OPENAI_API_KEY, OBJ_PROMPTS, TASK_PROMPTS

from utils.data_utils import TaskGraspDataset

openai_client = OpenAI(api_key=OPENAI_API_KEY)

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

    obj_desc = torch.from_numpy(obj_desc).unsqueeze(0).to(DEVICE)
    obj_desc_mask = torch.from_numpy(obj_desc_mask).unsqueeze(0).to(DEVICE)
    task_desc = torch.from_numpy(task_desc).unsqueeze(0).to(DEVICE)
    task_desc_mask = torch.from_numpy(task_desc_mask).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(pc, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask)
    logits = logits.squeeze()
    probs = torch.sigmoid(logits)
    preds = torch.round(probs)
    
    return probs, preds

def load_pc_and_grasps(data_dir, obj_name, view_idx: int):
    obj_dir = os.path.join(data_dir, "scans", obj_name)

    pc_file = os.path.join(obj_dir, f"{view_idx}_segmented_pc.npy")
    grasps_file = os.path.join(obj_dir, f"{view_idx}_registered_grasps.npy")

    if not os.path.exists(pc_file):
        return None

    pc = np.load(pc_file)
    grasps = np.load(grasps_file)

    return pc, grasps

def run_eval(
    dataset: TaskGraspDataset,
    data_dir: str,
    obj_id: str,
    view_idx: str,
    task_verb: str,
    model: GraspGPT_plain,
    bert_tokenizer: BertTokenizer,
    bert_model: BertModel,
    train_obj_grasp_tasks: dict
):
    try:
        pc, grasps = load_pc_and_grasps(data_dir, obj_id, view_idx)
    except:
        print(f"no point cloud and grasp")
        return None
    pc_input = regularize_pc_point_count(
        pc, cfg.num_points, use_farthest_point=False)

    object_name = obj_id.split("_", 1)[1].replace("_", " ")
    task_text = f"grasp the {object_name} to {task_verb}"

    object_class = obj_id.split("_", 1)[1]
    
    obj_desc_dir = os.path.join(data_dir, "obj_gpt_v2", object_class)
    task_desc_dir = os.path.join(data_dir, "task_gpt_v2", task_verb)


    preds = []
    probs = []

    # language descriptions
    #obj_desc_txt, task_desc_txt = gen_gpt_desc(object_name, task_verb)
    #obj_desc, _, obj_desc_mask = encode_text(obj_desc_txt, bert_tokenizer, bert_model, DEVICE, type='od')
    #task_desc, _, task_desc_mask = encode_text(task_desc_txt, bert_tokenizer, bert_model, DEVICE, type='td')
    # language instruciton
    task_ins, _, task_ins_mask = encode_text(task_text, bert_tokenizer, bert_model, DEVICE, type='li')

    skip_count = 0
    # eval each grasp in a loop
    for i in tqdm.trange(len(grasps)):

        # unique indicator
        unique_id = f"{obj_id}-{i}-{task_verb}"

        if unique_id in train_obj_grasp_tasks:
            # skip if the grasp is in the training set
            skip_count +=1
            continue

        grasp = grasps[i]

        pc = pc_input[:, :3]

        grasp_pc = get_gripper_control_points()
        grasp_pc = np.matmul(grasp, grasp_pc.T).T  # transform grasps
        grasp_pc = grasp_pc[:, :3]  # remove latent indicator

        latent = np.concatenate(
            [np.zeros(pc.shape[0]), np.ones(grasp_pc.shape[0])])  # create latent indicator
        latent = np.expand_dims(latent, axis=1)
        pc = np.concatenate([pc, grasp_pc], axis=0)  # [4103, 3]

        pc, grasp = pc_normalize(pc, grasp, pc_scaling=cfg.pc_scaling)
        pc = np.concatenate([pc, latent], axis=1)  # add back latent indicator

        # load language embeddings
        pc = torch.tensor([pc])

        # object class description embeddings
        obj_desc_path =  os.path.join(obj_desc_dir, 'descriptions', str(np.random.randint(0, 10)))
        if not os.path.exists(obj_desc_path):
            print(f"no description found for {object_name}")
            return None
            # raise ValueError(f"No such object description path: {obj_desc_path}")
        obj_desc_txt = open(os.path.join(obj_desc_path, 'all.txt')).readlines()[0]
        obj_desc = np.load(os.path.join(obj_desc_path, 'word_embed.npy'))[0]
        obj_desc_mask = np.load(os.path.join(obj_desc_path, 'attn_mask.npy'))[0]
        # task description embeddings 
        task_desc_path = os.path.join(task_desc_dir, 'descriptions', str(np.random.randint(0, 10)))
        if not os.path.exists(task_desc_path):
            print(f"no description found for {task_verb}")
            return None
            # raise ValueError(f"No such task description dir: {task_desc_path}")
        task_desc_txt = open(os.path.join(task_desc_path, 'all.txt')).readlines()[0]
        task_desc = np.load(os.path.join(task_desc_path, 'word_embed.npy'))[0]
        task_desc_mask = np.load(os.path.join(task_desc_path, 'attn_mask.npy'))[0]

        prob, pred = test(model, pc, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask)

        preds.append(pred.tolist())
        probs.append(prob.tolist())

    if len(preds) == 0:
        return None
    
    print(f"Skipped {skip_count} grasps in {obj_id} {view_idx} {task_verb} since in train")
    preds = np.array(preds).flatten().astype(bool)
    probs = np.array(probs).flatten()

    label_mask = dataset.get_grasp_label_mask(obj_id, task_verb)
    gt = dataset.get_grasp_labels(obj_id, task_verb)

    masked_preds = preds[label_mask]
    tp = np.sum(masked_preds & gt)
    fp = np.sum(masked_preds & ~gt)
    tn = np.sum(~masked_preds & ~gt)
    fn = np.sum(~masked_preds & gt)
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "top-1": preds[np.argmax(probs[label_mask])]
    }

def main(args, cfg):
    data_dir = args.data_dir

    # load train ids
    target_folder = "/net/nfs2.prior/arijitr/research/semantic_grasping/GraspGPT_public/data/taskgrasp/splits_final/"
    split_idx = cfg.split_idx
    split_mode = cfg.split_mode
    train_obj_grasp_tasks = {}
    
    train_split_file = os.path.join(target_folder, split_mode, str(split_idx), "train_split.txt")

    # open the file and read the lines
    with open(train_split_file, 'r') as f:
        lines = f.readlines()
        # remove the newline characters
        lines = [line.strip() for line in lines]
    
    # each line is formatted as <object_id>-<grasp_id>-<task_verb>:<label>. Get the object_id, grasp_id, and task_verb
    # format as dictionary and append to train_obj_grasp_tasks
    for line in lines:
        obj_grasp_task = line.split(":")
        train_obj_grasp_tasks[obj_grasp_task[0]] = obj_grasp_task[1]

    print(f"Total number of training samples: {len(train_obj_grasp_tasks)}")

    # load GraspGPT
    model = load_model(cfg)

    # load BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model = bert_model.to(DEVICE)
    bert_model.eval()

    tg_dataset = TaskGraspDataset(data_dir)
    top_1_results = []
    for object_id in tg_dataset.get_objects():
        for view_idx in tg_dataset.get_object_views(object_id):
            for task_verb in tg_dataset.get_object_tasks(object_id):
                results = run_eval(tg_dataset, data_dir, object_id, view_idx, task_verb, model, tokenizer, bert_model, train_obj_grasp_tasks)
                if results is not None:
                    top_1_results.append(results["top-1"])
    print(f"Top-1 Accuracy: {np.mean(top_1_results):.2%}")

    #save to csv
    run_name = args.cfg_file.split("/")[-1].split(".")[0]
    log_entry_name = f"{cfg.split_mode}_{cfg.split_idx}"
    with open(os.path.join("gcngrasp", "results", f"{run_name}_top_1_results.csv"), "w") as f:
        f.write(f"{log_entry_name},{np.mean(top_1_results):.2%}\n")
    

if __name__ == '__main__':
    """
    python gcngrasp/run_tg_eval_ver_ar.py --data_dir data/retargeted_taskgrasp
    """
    parser = argparse.ArgumentParser(description="visualize data and stuff")
    parser.add_argument('--task', help='', default='scoop')
    parser.add_argument('--obj_class', help='', default='spatula')
    parser.add_argument('--data_dir', help='location of sample data', default='')
    parser.add_argument('--obj_name', help='', default='spatula')
    parser.add_argument(
        '--cfg_file',
        help='yaml file in YACS config format to override default configs',
        default='cfg/eval/gcngrasp/gcngrasp_split_mode_t_split_idx_3_.yml',
        type=str)

    args = parser.parse_args()

    cfg = get_cfg_defaults()

    if args.cfg_file != '':
        if os.path.exists(args.cfg_file):
            cfg.merge_from_file(args.cfg_file)
        else:
            raise ValueError('Please provide a valid config file for the --cfg_file arg')

    if cfg.base_dir != '':
        if not os.path.exists(cfg.base_dir):
            raise FileNotFoundError(
                'Provided base dir {} not found'.format(
                    cfg.base_dir))
    else:
        assert cfg.base_dir == ''
        cfg.base_dir = os.path.join(os.path.dirname(__file__), '../data')

    cfg.batch_size = 16

    if len(cfg.gpus) == 1:
        torch.cuda.set_device(cfg.gpus[0])

    experiment_dir = os.path.join(cfg.log_dir, cfg.weight_file)

    weight_files = os.listdir(os.path.join(experiment_dir, 'weights'))
    assert len(weight_files) == 1
    cfg.weight_file = os.path.join(experiment_dir, 'weights', weight_files[0])

    if args.data_dir == '':
        args.data_dir = os.path.join(cfg.base_dir, 'sample_data/pcs')

    cfg.freeze()
    print(cfg)

    main(args, cfg)
