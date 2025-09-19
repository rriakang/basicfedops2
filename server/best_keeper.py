# server/best_keeper.py
import os, json, torch
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

class BestKeeper:
    def __init__(self, save_dir="./gl_best", metric_key="accuracy"):
        self.best = None  # {"metric": float, "round": int, "params": Parameters}
        self.metric_key = metric_key
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def update(self, server_round, parameters, metrics: dict):
        m = float(metrics.get(self.metric_key, -1.0))
        if (self.best is None) or (m > self.best["metric"]):
            self.best = {"metric": m, "round": server_round, "params": parameters}
            # 파일로도 저장(원하면 npy로 저장 가능)
            torch.save(parameters_to_ndarrays(parameters), os.path.join(self.save_dir, "best_params.pt"))
            with open(os.path.join(self.save_dir, "best_meta.json"), "w") as f:
                json.dump({"round": server_round, "metric": m}, f)

    def load_params(self):
        import numpy as np
        path = os.path.join(self.save_dir, "best_params.pt")
        if not os.path.exists(path):
            return None
        nds = torch.load(path, map_location="cpu")
        return ndarrays_to_parameters(nds)
