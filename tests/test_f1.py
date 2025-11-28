import unittest
import torch

from owl.utils.metrics import (
    ConfuseMatrix,
    f1_score,
    to_binary,
    confuse_matrix,
)


class TestToBinary(unittest.TestCase):
    def test_to_binary_basic(self):
        x = torch.tensor([0.2, 0.5, 0.8])
        y = to_binary(x, threshold=0.5)
        # > 0.5 -> 1, <= 0.5 -> 0
        expected = torch.tensor([0, 0, 1], dtype=y.dtype)
        self.assertTrue(torch.equal(y, expected))

    def test_to_binary_inplace_flag(self):
        x = torch.tensor([0.2, 0.8])

        # inplace=False 不修改原 tensor
        y = to_binary(x, threshold=0.5, inplace=False)
        self.assertTrue(torch.equal(x, torch.tensor([0.2, 0.8])))
        self.assertTrue(torch.equal(y, torch.tensor([0, 1], dtype=y.dtype)))

        # inplace=True 会修改 x 自身
        z = to_binary(x, threshold=0.5, inplace=True)
        self.assertTrue(torch.equal(x, torch.tensor([0, 1], dtype=x.dtype)))
        self.assertTrue(torch.equal(z, x))


class TestConfuseMatrix(unittest.TestCase):
    def test_confuse_matrix_single_image(self):
        """
        y_true:
        1 0
        0 1

        y_pred:
        1 1
        0 0
        """
        y_true = torch.tensor(
            [[
                [1, 0],
                [0, 1],
            ]], dtype=torch.float32
        )  # [1, 2, 2]
        y_pred = torch.tensor(
            [[
                [1, 1],
                [0, 0],
            ]], dtype=torch.float32
        )

        mat = confuse_matrix(y_pred, y_true)

        # 只输入了一张图，TP/TN/FP/FN shape 应该是 [1]
        self.assertEqual(mat.TP.shape, torch.Size([1]))
        self.assertEqual(mat.TN.shape, torch.Size([1]))
        self.assertEqual(mat.FP.shape, torch.Size([1]))
        self.assertEqual(mat.FN.shape, torch.Size([1]))

        # 手算：
        # TP: (0,0) = 1
        # TN: (1,0) = 1
        # FP: (0,1) = 1
        # FN: (1,1) = 1
        self.assertEqual(mat.TP.item(), 1)
        self.assertEqual(mat.TN.item(), 1)
        self.assertEqual(mat.FP.item(), 1)
        self.assertEqual(mat.FN.item(), 1)

    def test_confuse_matrix_batch(self):
        """
        batch 输入 [N, 1, 2, 2]
        第1张：完美预测
        第2张：全部错误
        """
        # 第一张：完美预测 -> TP=2, TN=2, FP=0, FN=0
        y_true1 = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32)
        y_pred1 = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32)

        # 第二张：全错 -> TP=0, TN=0, FP=2, FN=2
        y_true2 = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32)
        y_pred2 = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.float32)

        y_true = torch.stack([y_true1, y_true2], dim=0)  # [2, 1, 2, 2]
        y_pred = torch.stack([y_pred1, y_pred2], dim=0)

        mat = confuse_matrix(y_pred, y_true)

        self.assertTrue(torch.equal(mat.TP, torch.tensor([2.0, 0.0])))
        self.assertTrue(torch.equal(mat.TN, torch.tensor([2.0, 0.0])))
        self.assertTrue(torch.equal(mat.FP, torch.tensor([0.0, 2.0])))
        self.assertTrue(torch.equal(mat.FN, torch.tensor([0.0, 2.0])))


class TestF1Score(unittest.TestCase):
    def test_f1_score_single(self):
        """
        precision = 2 / (2+1) = 2/3
        recall    = 2 / (2+1) = 2/3
        f1        = 2 * p * r / (p + r) = 2/3
        """
        mat = ConfuseMatrix(
            TP=torch.tensor([2.0]),
            TN=torch.tensor([3.0]),
            FP=torch.tensor([1.0]),
            FN=torch.tensor([1.0]),
        )
        f1 = f1_score(mat)
        self.assertEqual(f1.shape, torch.Size([1]))
        self.assertTrue(torch.allclose(f1, torch.tensor([2.0 / 3.0]), atol=1e-6))

    def test_f1_score_batch(self):
        # 两张图：
        # 1) 完美预测 -> F1=1
        # 2) 全错 -> F1=0
        mat = ConfuseMatrix(
            TP=torch.tensor([4.0, 0.0]),
            TN=torch.tensor([0.0, 0.0]),
            FP=torch.tensor([0.0, 4.0]),
            FN=torch.tensor([0.0, 4.0]),
        )
        f1 = f1_score(mat)
        self.assertEqual(f1.shape, torch.Size([2]))
        self.assertTrue(torch.allclose(f1, torch.tensor([1.0, 0.0]), atol=1e-6))


class TestConfuseMatrixAdd(unittest.TestCase):
    def test_add_and_iadd(self):
        mat1 = ConfuseMatrix(
            TP=torch.tensor([1.0]),
            TN=torch.tensor([1.0]),
            FP=torch.tensor([0.0]),
            FN=torch.tensor([0.0]),
        )
        mat2 = ConfuseMatrix(
            TP=torch.tensor([2.0]),
            TN=torch.tensor([0.0]),
            FP=torch.tensor([1.0]),
            FN=torch.tensor([1.0]),
        )

        mat_sum = mat1 + mat2
        self.assertTrue(torch.equal(mat_sum.TP, torch.tensor([3.0])))
        self.assertTrue(torch.equal(mat_sum.TN, torch.tensor([1.0])))
        self.assertTrue(torch.equal(mat_sum.FP, torch.tensor([1.0])))
        self.assertTrue(torch.equal(mat_sum.FN, torch.tensor([1.0])))

        mat1 += mat2
        self.assertTrue(torch.equal(mat1.TP, torch.tensor([3.0])))
        self.assertTrue(torch.equal(mat1.TN, torch.tensor([1.0])))
        self.assertTrue(torch.equal(mat1.FP, torch.tensor([1.0])))
        self.assertTrue(torch.equal(mat1.FN, torch.tensor([1.0])))


if __name__ == "__main__":
    unittest.main()
