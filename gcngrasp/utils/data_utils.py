import os

import numpy as np

class TaskGraspDataset:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.grasp_labels: dict = {}  # obj -> task -> grasp_labels

        object_tasks_raw: dict = {}
        with open(f"{data_dir}/task2_results.txt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                part1, part2 = line.split(":")
                obj_id, grasp_id, task = part1.split("-")
                grasp_id = int(grasp_id)
                label = int(part2)
                if obj_id not in object_tasks_raw:
                    object_tasks_raw[obj_id] = {}
                if task not in object_tasks_raw[obj_id]:
                    object_tasks_raw[obj_id][task] = {}
                object_tasks_raw[obj_id][task][grasp_id] = label

        for obj_id in object_tasks_raw:
            for task in object_tasks_raw[obj_id]:
                grasp_labels = object_tasks_raw[obj_id][task]
                if not any(label == 1 for label in grasp_labels.values()):
                    continue
                assert len(grasp_labels) == max(grasp_labels.keys()) - min(grasp_labels.keys()) + 1
                assert min(grasp_labels.keys()) == 0
                grasp_labels_arr = np.zeros(len(grasp_labels), dtype=int)
                for grasp_id, label in grasp_labels.items():
                    grasp_labels_arr[grasp_id] = label
                if obj_id not in self.grasp_labels:
                    self.grasp_labels[obj_id] = {}
                self.grasp_labels[obj_id][task] = grasp_labels_arr

    def object_has_task(self, obj_id: str, task: str) -> bool:
        return obj_id in self.grasp_labels and task in self.grasp_labels[obj_id]

    def get_all_grasp_labels(self, obj_id: str, task: str) -> np.ndarray:
        """Returns array of ints, where 1 is positive, -1 is negative, 0 is unclear, and -2 is unsure"""
        return self.grasp_labels[obj_id][task]

    def get_grasp_label_mask(self, obj_id: str, task: str) -> np.ndarray:
        """Returns a mask of all_grasp_labels where definitive answers are known"""
        labels = self.get_all_grasp_labels(obj_id, task)
        return (labels == 1) | (labels == -1)

    def get_grasp_labels(self, obj_id: str, task: str) -> np.ndarray:
        """Returns a boolean array of grasp labels where True is positive and False is negative, of the same shape as mask"""
        labels = self.get_all_grasp_labels(obj_id, task)
        mask = self.get_grasp_label_mask(obj_id, task)
        return labels[mask] == 1

    def get_objects(self):
        return sorted(self.grasp_labels.keys())

    def get_object_tasks(self, obj_id: str):
        return sorted(self.grasp_labels[obj_id].keys())

    def get_object_views(self, obj_id: str):
        object_dir = f"{self.data_dir}/scans/{obj_id}"
        views = []
        for fn in os.listdir(object_dir):
            if fn.endswith("_segmented_pc.npy"):
                views.append(int(fn.split("_")[0]))
        return sorted(views)

if __name__ == "__main__":
    dataset = TaskGraspDataset("../semantic-grasping/data/taskgrasp")
    breakpoint()
    pass
