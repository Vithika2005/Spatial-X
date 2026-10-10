import unittest
import numpy as np
from spatialx.evaluation import depth_metrics, binary_mask_iou, retrieval_recall_at_k

class EvaluationHelperTests(unittest.TestCase):
    def test_depth_metrics_perfect_prediction(self):
        depth = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = depth_metrics(depth, depth)
        self.assertAlmostEqual(result["abs_rel"], 0.0)
        self.assertAlmostEqual(result["rmse"], 0.0)
        self.assertEqual(result["valid_pixels"], 4)

    def test_mask_iou(self):
        pred = np.array([[1, 1], [0, 0]], dtype=bool)
        gt = np.array([[1, 0], [1, 0]], dtype=bool)
        self.assertAlmostEqual(binary_mask_iou(pred, gt), 1 / 3)

    def test_recall_at_k(self):
        score = retrieval_recall_at_k([[4, 2, 1], [7, 8, 9]], [{2}, {8}], k=2)
        self.assertEqual(score, 1.0)

if __name__ == "__main__":
    unittest.main()
