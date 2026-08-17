import unittest

try:
    from func import WeightedKNNClassifier
except ModuleNotFoundError as exc:
    if exc.name != "torchmetrics":
        raise
    WeightedKNNClassifier = None


class TorchMetricsCompatibilityTests(unittest.TestCase):
    @unittest.skipIf(WeightedKNNClassifier is None, "torchmetrics is not installed")
    def test_weighted_knn_constructs_with_installed_torchmetrics(self):
        metric = WeightedKNNClassifier()

        self.assertEqual(metric.k, 20)
        self.assertEqual(metric.distance_fx, "cosine")


if __name__ == "__main__":
    unittest.main()
