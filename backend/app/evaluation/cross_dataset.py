from typing import Dict
import numpy as np
from backend.app.services.experiments import ExperimentTracker
from backend.app.services.inference import InferenceService

def run_cross_dataset_evaluation(
    exp_tracker: ExperimentTracker, 
    exp_id: str, 
    inference_service: InferenceService, 
    target_dataset_name: str, 
    test_images: list, 
    ground_truth: list
) -> Dict[str, float]:
    """
    Evaluates the model on a secondary dataset to test generalization.
    """
    if not test_images or not ground_truth or len(test_images) != len(ground_truth):
        results = {
            "auroc": "Not evaluated",
            "auprc": "Not evaluated",
            "fpr_at_95_tpr": "Not evaluated"
        }
        exp_tracker.log_cross_dataset_result(exp_id, target_dataset_name, results)
        return results
        
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
        
        preds = []
        for img in test_images:
            res = inference_service.analyze(img)
            # Use raw score for AUC calculation
            score = res["anomaly"]["score"]
            preds.append(score)
            
        preds = np.array(preds)
        gt = np.array(ground_truth)
        
        # Ground truth: 1 for anomaly, 0 for normal
        auroc = float(roc_auc_score(gt, preds))
        auprc = float(average_precision_score(gt, preds))
        
        fpr, tpr, _ = roc_curve(gt, preds)
        # Find FPR at 95% TPR
        idx = np.where(tpr >= 0.95)[0]
        fpr_95 = float(fpr[idx[0]]) if len(idx) > 0 else 1.0
        
        results = {
            "auroc": auroc,
            "auprc": auprc,
            "fpr_at_95_tpr": fpr_95
        }
    except Exception as e:
        results = {
            "auroc": "Not evaluated",
            "auprc": "Not evaluated",
            "fpr_at_95_tpr": "Not evaluated",
            "error": str(e)
        }
        
    exp_tracker.log_cross_dataset_result(exp_id, target_dataset_name, results)
    return results
