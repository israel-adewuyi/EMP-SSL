import unittest

import torch.nn as nn

from model.model import encoder


class ModelNormalizationTests(unittest.TestCase):
    def test_batch_option_contains_batch_norm_and_no_layer_norm(self):
        model = encoder(z_dim=8, hidden_dim=16, norm="batch")

        self.assertTrue(any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()))
        self.assertFalse(any(isinstance(module, nn.LayerNorm) for module in model.modules()))

    def test_layer_option_contains_layer_norm_and_no_batch_norm(self):
        model = encoder(z_dim=8, hidden_dim=16, norm="layer")

        self.assertTrue(any(isinstance(module, nn.LayerNorm) for module in model.modules()))
        self.assertFalse(any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()))


if __name__ == "__main__":
    unittest.main()
