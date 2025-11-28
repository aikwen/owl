import unittest
import math
import torch

from owl.utils.metrics import auc_single, auc_batch


class TestAUCSingle(unittest.TestCase):
    def test_auc_single_perfect_and_worst(self):
        """
        y_true:  [1, 0, 1, 0]  (两正两负)
        情况一：完美预测 -> AUC=1
        情况二：完全相反 -> AUC=0
        """
        gt = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
        ])  # [H, W]

        # 完美预测：正样本分数都比负样本高
        pred_perfect = torch.tensor([
            [0.9, 0.1],
            [0.8, 0.2],
        ])
        auc_p = auc_single(gt, pred_perfect)
        self.assertTrue(math.isfinite(auc_p))
        self.assertAlmostEqual(auc_p, 1.0, places=6)

        # 完全相反：正样本分数都比负样本低
        pred_worst = torch.tensor([
            [0.1, 0.9],
            [0.2, 0.8],
        ])
        auc_w = auc_single(gt, pred_worst)
        self.assertTrue(math.isfinite(auc_w))
        self.assertAlmostEqual(auc_w, 0.0, places=6)

    def test_auc_single_nan_when_single_class(self):
        """
        当 gt_prob 全 0 或全 1 时，预期返回 NaN。
        """
        all_zero = torch.zeros((2, 2), dtype=torch.float32)
        all_one = torch.ones((2, 2), dtype=torch.float32)
        pred = torch.rand((2, 2), dtype=torch.float32)

        auc_zero = auc_single(all_zero, pred)
        auc_one = auc_single(all_one, pred)

        self.assertTrue(math.isnan(auc_zero))
        self.assertTrue(math.isnan(auc_one))


class TestAUCBatch(unittest.TestCase):
    def test_auc_batch_mixed(self):
        """
        batch 里包含：
        - 一张有正负样本的图 -> 返回正常 AUC
        - 一张全 0 的图 -> 返回 NaN，并在 auc_batch 中被过滤掉
        """
        # 第一张：有正负样本，且预测接近完美
        gt1 = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
        ])
        pred1 = torch.tensor([
            [0.9, 0.1],
            [0.8, 0.2],
        ])

        # 第二张：全 0
        gt2 = torch.zeros((2, 2), dtype=torch.float32)
        pred2 = torch.rand((2, 2), dtype=torch.float32)

        # 组成 batch 为 [N, 1, H, W]
        gt_batch = torch.stack([gt1, gt2], dim=0).unsqueeze(1)
        pred_batch = torch.stack([pred1, pred2], dim=0).unsqueeze(1)

        res = auc_batch(gt_batch, pred_batch, auc_single_func=auc_single)

        # 只会有第一张的 AUC，长度应为 1，且约为 1.0
        self.assertEqual(len(res), 1)
        self.assertAlmostEqual(res[0], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
