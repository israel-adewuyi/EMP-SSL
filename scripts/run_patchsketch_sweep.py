import csv
import hashlib
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


# Edit this block for your sweep.
SWEEP_NAME = "cifar10_patchsketch_grid"
OUTPUT_ROOT = REPO_ROOT / "sweeps" / SWEEP_NAME
TRAIN_CUDA_VISIBLE_DEVICES = "3"
EVAL_CUDA_VISIBLE_DEVICES = "2"
STOP_ON_FAILURE = True
SKIP_COMPLETED_RUNS = True

TRAIN_BASE_ARGS = {
    "data": "cifar10",
    "arch": "resnet18-cifar",
    "num_patches": 200,
    "bs": 32,
    "epoch": 1,
    "lr": 0.3,
    "cov_weight": 1.0,
    "dir": "PatchSketch-Training",
    "msg": "SWEEP",
    "hist_every_n_steps": 100,
    "image_every_n_steps": 0,
}

TRAIN_SWEEP_GRID = {
    "sketch_size": [10],
    "selected_patches": [25, 50, 100],
}

TRAIN_FLAGS = {
    "disable_tensorboard": False,
}

EVAL_ARGS = {
    "data": "cifar10",
    "arch": "resnet18-cifar",
    "test_patches": 128,
    "num_workers": 0,
}

EVAL_FLAGS = {
    "knn": False,
}


def sanitize_value(value):
    text = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in text)


def iter_sweep_configs():
    keys = list(TRAIN_SWEEP_GRID.keys())
    values = [TRAIN_SWEEP_GRID[key] for key in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def build_run_slug(train_args):
    signature = hashlib.sha1(
        json.dumps(train_args, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    data = sanitize_value(train_args.get("data", "data"))
    arch = sanitize_value(train_args.get("arch", "arch"))
    return (
        f"sk{train_args['sketch_size']}_sel{train_args['selected_patches']}"
        f"_np{train_args['num_patches']}_bs{train_args['bs']}"
        f"_lr{sanitize_value(train_args['lr'])}_cov{sanitize_value(train_args['cov_weight'])}"
        f"_{data}_{arch}_{signature}"
    )


def build_train_args(combo_args):
    train_args = dict(TRAIN_BASE_ARGS)
    train_args.update(combo_args)
    run_slug = build_run_slug(train_args)
    base_msg = sanitize_value(train_args.get("msg", "SWEEP"))
    signature = run_slug.rsplit("_", 1)[-1]
    train_args["msg"] = f"{base_msg}_{signature}"
    return train_args, run_slug


def patchsketch_run_dir(train_args):
    return REPO_ROOT / "logs" / str(train_args["dir"]) / (
        f"sketch{train_args['sketch_size']}_topk{train_args['selected_patches']}"
        f"_numpatch{train_args['num_patches']}_bs{train_args['bs']}_lr{train_args['lr']}"
        f"_{train_args['msg']}"
    )


def checkpoint_path_for(train_args):
    epoch_index = int(train_args["epoch"]) - 1
    return patchsketch_run_dir(train_args) / "save_models" / f"{epoch_index}.pt"


def build_cli_args(arg_map, flag_map=None):
    cli_args = []
    for key, value in arg_map.items():
        cli_args.extend([f"--{key}", str(value)])
    if flag_map is not None:
        for key, enabled in flag_map.items():
            if enabled:
                cli_args.append(f"--{key}")
    return cli_args


def run_and_tee(command, env_overrides, log_path):
    env = os.environ.copy()
    env.update(env_overrides)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def load_eval_metrics(results_json_path):
    with open(results_json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results", {})
    linear_results = results.get("linear", {})
    knn_results = results.get("knn", {})
    return {
        "best_linear_top1": linear_results.get("best_linear_top1"),
        "last_linear_top1": linear_results.get("last_linear_top1"),
        "best_linear_top5": linear_results.get("best_linear_top5"),
        "last_linear_top5": linear_results.get("last_linear_top5"),
        "best_linear_epoch": linear_results.get("best_linear_epoch"),
        "knn_top1": knn_results.get("top1"),
        "knn_top5": knn_results.get("top5"),
    }


def write_summary_csv(rows, summary_csv_path):
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_csv_path = OUTPUT_ROOT / "summary.csv"
    summary_rows = []
    started_at = datetime.now().isoformat(timespec="seconds")
    print(f"Starting PatchSketch sweep '{SWEEP_NAME}' at {started_at}")

    for run_index, combo_args in enumerate(iter_sweep_configs(), start=1):
        train_args, run_slug = build_train_args(combo_args)
        run_output_dir = OUTPUT_ROOT / run_slug
        run_output_dir.mkdir(parents=True, exist_ok=True)
        train_log_path = run_output_dir / "train.log"
        eval_log_path = run_output_dir / "eval.log"
        eval_results_json_path = run_output_dir / "eval_results.json"
        checkpoint_path = checkpoint_path_for(train_args)

        row = {
            "run_index": run_index,
            "run_slug": run_slug,
            "status": "pending",
            "train_log": str(train_log_path),
            "eval_log": str(eval_log_path),
            "eval_results_json": str(eval_results_json_path),
            "checkpoint_path": str(checkpoint_path),
        }
        row.update(train_args)
        row.update(EVAL_ARGS)
        summary_rows.append(row)

        if SKIP_COMPLETED_RUNS and eval_results_json_path.exists():
            row["status"] = "skipped_existing"
            row.update(load_eval_metrics(eval_results_json_path))
            print(f"[{run_index}] Skipping existing run {run_slug}")
            write_summary_csv(summary_rows, summary_csv_path)
            continue

        train_command = [sys.executable, "main_patchsketch.py"]
        train_command.extend(build_cli_args(train_args, TRAIN_FLAGS))
        eval_command = [sys.executable, "evaluate.py"]
        eval_arg_map = dict(EVAL_ARGS)
        eval_arg_map["model_path"] = str(checkpoint_path)
        eval_arg_map["results_json"] = str(eval_results_json_path)
        eval_command.extend(build_cli_args(eval_arg_map, EVAL_FLAGS))

        (run_output_dir / "train_command.txt").write_text(
            " ".join(train_command), encoding="utf-8"
        )
        (run_output_dir / "eval_command.txt").write_text(
            " ".join(eval_command), encoding="utf-8"
        )

        try:
            print(f"[{run_index}] Training {run_slug}")
            run_and_tee(
                train_command,
                {"CUDA_VISIBLE_DEVICES": TRAIN_CUDA_VISIBLE_DEVICES},
                train_log_path,
            )

            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Expected checkpoint not found: {checkpoint_path}")

            print(f"[{run_index}] Evaluating {run_slug}")
            run_and_tee(
                eval_command,
                {"CUDA_VISIBLE_DEVICES": EVAL_CUDA_VISIBLE_DEVICES},
                eval_log_path,
            )

            row["status"] = "completed"
            row.update(load_eval_metrics(eval_results_json_path))
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            write_summary_csv(summary_rows, summary_csv_path)
            if STOP_ON_FAILURE:
                raise
        else:
            write_summary_csv(summary_rows, summary_csv_path)

    finished_at = datetime.now().isoformat(timespec="seconds")
    print(f"Finished PatchSketch sweep '{SWEEP_NAME}' at {finished_at}")
    print(f"Summary written to {summary_csv_path}")


if __name__ == "__main__":
    main()
