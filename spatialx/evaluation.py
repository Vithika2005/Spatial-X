"""Small metrics helpers. Use only on aligned predictions and ground truth."""
import numpy as np

def depth_metrics(predicted_depth, ground_truth_depth, valid_mask=None):
    pred=np.asarray(predicted_depth,dtype=float); gt=np.asarray(ground_truth_depth,dtype=float)
    if pred.shape != gt.shape: raise ValueError("Depth arrays must have the same aligned shape")
    valid=np.isfinite(pred)&np.isfinite(gt)&(pred>0)&(gt>0)
    if valid_mask is not None: valid &= np.asarray(valid_mask,dtype=bool)
    if not valid.any(): raise ValueError("No valid aligned depth pixels")
    p,g=pred[valid],gt[valid]; ratio=np.maximum(p/g,g/p)
    return {"valid_pixels":int(valid.sum()),"abs_rel":float(np.mean(np.abs(p-g)/g)),
            "rmse":float(np.sqrt(np.mean((p-g)**2))),"delta1":float(np.mean(ratio<1.25)),
            "delta2":float(np.mean(ratio<1.25**2)),"delta3":float(np.mean(ratio<1.25**3))}

def binary_mask_iou(predicted_mask, ground_truth_mask):
    p=np.asarray(predicted_mask,dtype=bool); g=np.asarray(ground_truth_mask,dtype=bool)
    if p.shape != g.shape: raise ValueError("Masks must have the same shape")
    union=np.logical_or(p,g).sum()
    return 1.0 if union==0 else float(np.logical_and(p,g).sum()/union)

def retrieval_recall_at_k(rankings, relevant_instance_ids, k=1):
    if len(rankings)!=len(relevant_instance_ids): raise ValueError("Each query needs relevant instance IDs")
    if not rankings: return None
    return float(np.mean([any(x in set(rel) for x in rank[:k]) for rank,rel in zip(rankings,relevant_instance_ids)]))
