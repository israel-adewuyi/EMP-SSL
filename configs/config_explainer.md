# PatchSketch configuration explainer

Every setting in `patchsketch_cifar10.toml` is summarized below; values under
`train.grid` are combined as a Cartesian-product sweep.

## Sweep

| Setting | Summary |
| --- | --- |
| `sweep.name` | Names the sweep in logs and console output. |
| `sweep.output_root` | Sets the repository-relative directory for run logs, commands, evaluation results, and the summary CSV. |
| `sweep.train_cuda_visible_devices` | Selects the GPU IDs visible to each training subprocess. |
| `sweep.eval_cuda_visible_devices` | Selects the GPU IDs visible to each evaluation subprocess. |
| `sweep.stop_on_failure` | Stops the entire sweep when a run fails if set to `true`. |
| `sweep.skip_completed_runs` | Reuses runs that already have an evaluation-results JSON file if set to `true`. |

## Training arguments

| Setting | Summary |
| --- | --- |
| `train.args.data` | Selects the training dataset. |
| `train.args.arch` | Selects the encoder architecture. |
| `train.args.num_patches` | Sets the number of candidate augmented patches generated per image. |
| `train.args.bs` | Sets the number of source images in each training batch. |
| `train.args.epoch` | Sets the number of training epochs. |
| `train.args.num_workers` | Sets training DataLoader workers; use `0` in shared-memory-limited containers. |
| `train.args.lr` | Sets the training optimizer learning rate. |
| `train.args.cov_weight` | Weights the covariance-decorrelation term in the PatchSketch loss. |
| `train.args.dir` | Names the parent directory created under `logs/`. |
| `train.args.msg` | Adds a label to the run directory name; the runner appends a unique signature. |
| `train.args.hist_every_n_steps` | Logs TensorBoard histograms every N steps, with `0` disabling them. |
| `train.args.image_every_n_steps` | Logs patch-preview images every N steps, with `0` disabling them. |
| `train.grid.norm` | Selects `batch` or `layer` normalization globally for both the ResNet backbone and projection MLP. |
| `train.grid.sketch_size` | Lists the Frequent Directions sketch capacities to sweep over. |
| `train.grid.selected_patches` | Lists the numbers of top-scoring patches retained per image. |
| `train.flags.disable_tensorboard` | Adds `--disable_tensorboard` to training when set to `true`. |

## Evaluation arguments

| Setting | Summary |
| --- | --- |
| `evaluate.args.data` | Selects the evaluation dataset. |
| `evaluate.args.arch` | Selects the architecture used to load the trained checkpoint. |
| `evaluate.args.test_patches` | Sets the number of augmented patches aggregated for each evaluation image. |
| `evaluate.args.lr` | Sets the linear-evaluation classifier learning rate. |
| `evaluate.args.linear` | Enables linear evaluation when set to `true`. |
| `evaluate.args.device` | Selects `auto`, `cuda`, or `cpu` for evaluation. |
| `evaluate.args.num_workers` | Sets the evaluation DataLoader worker count. |
| `evaluate.flags.knn` | Adds `--knn` and runs cosine-similarity k-NN evaluation when set to `true`. |
| `norm` | Is copied from each training combination so evaluation always reconstructs the matching encoder. |
| `model_path` | Is generated per run from its expected checkpoint path and is not manually configured. |
| `results_json` | Is generated per run inside its sweep-output directory and is not manually configured. |
