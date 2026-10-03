import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any

class ExperimentTracker:
    """
    Local experiment tracking to avoid heavy MLflow dependencies if unnecessary.
    """
    def __init__(self, storage_dir: str = "artifacts/experiments"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        
    def create_experiment(self, config: Dict[str, Any]) -> str:
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        exp_path = os.path.join(self.storage_dir, exp_id)
        os.makedirs(exp_path, exist_ok=True)
        
        # Save config
        with open(os.path.join(exp_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
            
        return exp_id
        
    def log_metrics(self, exp_id: str, epoch: int, metrics: Dict[str, float]):
        exp_path = os.path.join(self.storage_dir, exp_id)
        metrics_file = os.path.join(exp_path, "metrics.jsonl")
        
        entry = {"epoch": epoch, "timestamp": datetime.now().isoformat(), **metrics}
        with open(metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
            
    def save_calibration(self, exp_id: str, calibration_data: Dict[str, Any]):
        exp_path = os.path.join(self.storage_dir, exp_id)
        with open(os.path.join(exp_path, "calibration.json"), "w") as f:
            json.dump(calibration_data, f, indent=2)
            
    def log_cross_dataset_result(self, exp_id: str, target_dataset: str, results: Dict[str, float]):
        exp_path = os.path.join(self.storage_dir, exp_id)
        file_path = os.path.join(exp_path, f"cross_eval_{target_dataset}.json")
        with open(file_path, "w") as f:
            json.dump(results, f, indent=2)
