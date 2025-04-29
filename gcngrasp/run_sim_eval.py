import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

from pydantic import BaseModel
from PIL import Image
import wandb
import h5py
import pandas as pd
import numpy as np
import numpy.typing as npt
import yaml
from tqdm import tqdm

from inference import GraspGPTPredictor, load_cfg

class SimEvalConfig(BaseModel):
    obs_dir: str
    task_dir: str
    batch_size: int
    model_config_path: str
    far_clip: float
    dataloader_workers: int

class Batch(BaseModel, arbitrary_types_allowed=True):
    images: list[Image.Image]
    depths: list[npt.NDArray[np.floating]]
    grasps: list[npt.NDArray[np.floating]]
    cam_Ks: list[npt.NDArray[np.floating]]
    tasks: list[str]
    labels: list[int]

def load_data(scene_path: str, view_id: str, far_clip: float):
    with h5py.File(scene_path, "r") as f:
        rgb = f[view_id]["rgb"][:]
        image = Image.fromarray(rgb)
        xyz: npt.NDArray[np.floating] = f[view_id]["xyz"][:]
        cam_K: npt.NDArray[np.floating] = f[view_id]["cam_params"][:]

        depth = xyz[:, :, 2]
        depth[depth >= far_clip] = 0

        grasps_list: list[npt.NDArray[np.floating]] = []
        n_obs = sum(1 for obs_id in f[view_id].keys() if obs_id.startswith("obs_"))
        for i in range(n_obs):
            obs_id = f"obs_{i}"
            assert obs_id in f[view_id].keys()
            grasp = f[view_id][obs_id]["grasp_pose"][:]
            grasps_list.append(grasp)

    grasps = np.stack(grasps_list, axis=0)
    return image, depth, grasps, cam_K

def load_batch(config: SimEvalConfig, batch_df: pd.DataFrame):
    images = []
    depths = []
    tasks = []
    grasps = []
    cam_Ks = []
    labels = []
    obs_dir = config.obs_dir
    for _, row in batch_df.iterrows():
        scene_path = os.path.join(obs_dir, row["scene_path"])
        image, depth, view_grasps, cam_K = load_data(scene_path, row["view_id"], config.far_clip)
        images.append(image)
        depths.append(depth)
        grasps.append(view_grasps)
        cam_Ks.append(cam_K)
        tasks.append(row["task"])
        labels.append(int(row["obs_id"].split("_")[-1]))
    return Batch(
        images=images,
        depths=depths,
        grasps=grasps,
        cam_Ks=cam_Ks,
        tasks=tasks,
        labels=labels
    )

def eval_batch(predictor: GraspGPTPredictor, batch: Batch):
    preds = []
    for i in range(len(batch.images)):
        preds.append(predictor.predict_grasp(batch.images[i], batch.depths[i], batch.cam_Ks[i], batch.grasps[i], batch.tasks[i], verbosity=3))
    succ_pred_viz = []
    fail_pred_viz = []
    n_succ = 0
    n_samples = 0
    for i in range(len(batch.images)):
        if preds[i] is not None and preds[i] == batch.labels[i]:
            succ_pred_viz.append((batch.images[i], batch.tasks[i]))
            n_succ += 1
        else:
            fail_pred_viz.append((batch.images[i], batch.tasks[i]))
        n_samples += 1
    results = {
        "n_samples": n_samples,
        "n_succ": n_succ,
    }
    return results, succ_pred_viz, fail_pred_viz


def build_wandb_config(wandb_config: dict):
    wandb_config = {**wandb_config}
    wandb_config["env"] = {
        **{k: v for k, v in os.environ.items() if k.startswith("GANTRY_")},
        **{k: v for k, v in os.environ.items() if k.startswith("BEAKER_")},
    }
    return wandb_config

def main():
    with open("cfg/eval_sim.yaml", "r") as f:
        config = SimEvalConfig.model_validate(yaml.safe_load(f))
    print(config.model_dump_json(indent=4))

    df = pd.read_csv(os.path.join(config.task_dir, "matched_tasks.csv"))

    cfg = load_cfg(config.model_config_path)
    predictor = GraspGPTPredictor(cfg)

    out_dir = "/results"
    task_name = os.getenv("GANTRY_TASK_NAME", None)
    run = wandb.init(
        entity="prior-ai2",
        project="semantic-grasping",
        config=build_wandb_config(config.model_dump()),
        name=task_name,
        dir=out_dir,
        job_type="eval_sim"
    )

    succ_viz: list[wandb.Image] = []
    fail_viz: list[wandb.Image] = []
    results = {
        "n_samples": 0,
        "n_succ": 0,
    }

    with tqdm(total=len(df), desc="Evaluating") as pbar:
        with ThreadPoolExecutor(config.dataloader_workers) as executor:
            for i in range(0, len(df), config.batch_size):
                batches = [load_batch(config, df.iloc[j:j+1]) for j in range(i, min(i+config.batch_size, len(df)))]
                futures = [executor.submit(eval_batch, predictor, batch) for batch in batches]

                for future in as_completed(futures):
                    batch_results, succ_pred_viz, fail_pred_viz = future.result()
                    succ_viz.extend([wandb.Image(image, caption=task) for image, task in succ_pred_viz])
                    fail_viz.extend([wandb.Image(image, caption=task) for image, task in fail_pred_viz])
                    for k, v in batch_results.items():
                        if k in results:
                            results[k] += v
                        pbar.update(1)
                    pbar.set_description(f"Evaluating, accuracy: {results['n_succ']}/{results['n_samples']} ({results['n_succ'] / results['n_samples']:.1%})")
    run.summary["results"] = results
    run.summary["accuracy"] = results["n_succ"] / results["n_samples"]
    print(f"Average top-1 accuracy: {results['n_succ'] / results['n_samples']:.1%}")

    if len(succ_viz) > 0:
        if len(succ_viz) > 100:
            random.shuffle(succ_viz)
            succ_viz = succ_viz[:100]
        run.log({"succ_predictions": succ_viz})
    if len(fail_viz) > 0:
        if len(fail_viz) > 100:
            random.shuffle(fail_viz)
            fail_viz = fail_viz[:100]
        run.log({"fail_predictions": fail_viz})

if __name__ == "__main__":
    main()
