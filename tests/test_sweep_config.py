import unittest

from scripts.run_patchsketch_sweep import (
    DEFAULT_CONFIG_PATH,
    build_train_args,
    iter_sweep_configs,
    load_sweep_config,
)


class SweepConfigTests(unittest.TestCase):
    def test_default_toml_config_loads(self):
        config = load_sweep_config(DEFAULT_CONFIG_PATH)

        self.assertEqual(config["name"], "cifar10_patchsketch_grid")
        self.assertEqual(config["train_args"]["num_patches"], 200)
        self.assertEqual(config["train_args"]["num_workers"], 0)
        self.assertEqual(config["train_grid"]["selected_patches"], [25, 50, 100])
        self.assertEqual(config["train_grid"]["norm"], ["batch", "layer"])
        self.assertEqual(config["eval_args"]["test_patches"], 128)
        self.assertEqual(config["eval_args"]["lr"], 0.03)
        self.assertTrue(config["eval_args"]["linear"])
        self.assertEqual(config["eval_args"]["device"], "auto")
        self.assertTrue(config["output_root"].is_absolute())

    def test_grid_expands_to_one_config_per_combination(self):
        combinations = list(
            iter_sweep_configs({"sketch_size": [5, 10], "selected_patches": [10, 20]})
        )

        self.assertEqual(len(combinations), 4)
        self.assertIn({"sketch_size": 10, "selected_patches": 20}, combinations)

    def test_grid_values_override_base_training_args(self):
        train_args, run_slug = build_train_args(
            {
                "data": "cifar10",
                "arch": "resnet18-cifar",
                "num_patches": 100,
                "bs": 16,
                "lr": 0.3,
                "cov_weight": 1.0,
                "msg": "TEST",
            },
            {"sketch_size": 8, "selected_patches": 20},
        )

        self.assertEqual(train_args["sketch_size"], 8)
        self.assertEqual(train_args["selected_patches"], 20)
        self.assertIn("sk8_sel20", run_slug)


if __name__ == "__main__":
    unittest.main()
