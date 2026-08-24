import time
import json
import os
import dataclasses


class experimentTracker:
    def __init__(self, run_dir, config):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)

        self.metric_path = os.path.join(self.run_dir, "metrics.jsonl")
        self.config_path = os.path.join(self.run_dir, "config.json")

        self.t0 = time.time()

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(config), f, indent=2, sort_keys=True)

    def wall_time(self):
        return time.time() - self.t0

    def log(self, step, metrics):
        records = {"step": int(step), "wall_time_s": float(self.wall_time())}
        for k, v in metrics.items():
            records[k] = float(v) if isinstance(v, (int, float)) else v
        with open(self.metric_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(records) + "\n")

    def close(self):
        pass
