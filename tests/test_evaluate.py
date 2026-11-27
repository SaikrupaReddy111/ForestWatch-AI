import torch
import numpy as np
from src.evaluate import compute_segmentation_metrics

def test_compute_segmentation_metrics_known_confusion():
    # Create a batch of size 1, 2x2 mask for simplicity
    # logits before sigmoid
    logits = torch.tensor([[[[10.0, -10.0],
                             [-10.0, 10.0]]]])  # shape (1,1,2,2)
    # Targets: 1 where logits high, 0 where low, ignore one pixel (255)
    targets = torch.tensor([[[[1, 0],
                              [0, 255]]]])
    metrics = compute_segmentation_metrics(logits, targets, threshold=0.5, ignore_index=255)
    # Expected: TP=1 (first pixel), FP=0, FN=0, TN=2
    assert metrics['confusion_matrix']['tp'] == 1
    assert metrics['confusion_matrix']['fp'] == 0
    assert metrics['confusion_matrix']['fn'] == 0
    assert metrics['confusion_matrix']['tn'] == 2
    # IoU = 1.0 (perfect overlap)
    assert np.isclose(metrics['iou'], 1.0)
    # Dice = 1.0 (perfect overlap)
    assert np.isclose(metrics['dice'], 1.0)
    # Precision = 1.0
    assert np.isclose(metrics['precision'], 1.0)
    # Recall = 1.0
    assert np.isclose(metrics['recall'], 1.0)
