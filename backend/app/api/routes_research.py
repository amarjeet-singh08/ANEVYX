from fastapi import APIRouter, BackgroundTasks
from backend.app.services.experiments import ExperimentTracker
from backend.app.services.training import train_pipeline
import traceback

router = APIRouter()
tracker = ExperimentTracker()

def run_training_task(exp_id: str):
    try:
        train_pipeline(epochs=20) # 20 for speed in demo, but it's real data
        print(f"Experiment {exp_id} completed successfully.")
    except Exception as e:
        print(f"Experiment {exp_id} failed: {e}")
        traceback.print_exc()

@router.post("/experiments")
async def start_experiment(config: dict, background_tasks: BackgroundTasks):
    exp_id = tracker.create_experiment(config)
    background_tasks.add_task(run_training_task, exp_id)
    return {"status": "started", "experiment_id": exp_id, "message": "Real training pipeline started in background."}

@router.get("/experiments/{exp_id}")
async def get_experiment(exp_id: str):
    import os, json
    exp_path = os.path.join(tracker.storage_dir, exp_id)
    if not os.path.exists(exp_path):
        return {"error": "Not found"}
        
    with open(os.path.join(exp_path, "config.json")) as f:
        config = json.load(f)
        
    return {"id": exp_id, "config": config}
