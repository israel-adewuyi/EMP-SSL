import argparse
import csv
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "patchsketch_cifar10.toml"


def parse_args():
    parser = argparse.ArgumentParser(description="Run a PatchSketch TOML experiment sweep")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"experiment TOML file (default: {DEFAULT_CONFIG_PATH.relative_to(REPO_ROOT)})",
    )
    return parser.parse_args()


def require_table(parent, key, context="root"):
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected TOML table [{key}] under {context}")
    return value


def load_sweep_config(config_path):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config_path = config_path.resolve()

    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)

    sweep = require_table(payload, "sweep")
    train = require_table(payload, "train")
    evaluate = require_table(payload, "evaluate")

    name = sweep.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("[sweep].name must be a non-empty string")

    output_root = Path(sweep.get("output_root", f"sweeps/{name}"))
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    train_grid = require_table(train, "grid", context="[train]")
    for key, values in train_grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"[train.grid].{key} must be a non-empty array")

    return {
        "path": config_path,
        "name": name,
        "output_root": output_root.resolve(),
        "train_cuda_visible_devices": str(
            sweep.get("train_cuda_visible_devices", "0")
        ),
        "eval_cuda_visible_devices": str(
            sweep.get("eval_cuda_visible_devices", "0")
        ),
        "stop_on_failure": bool(sweep.get("stop_on_failure", True)),
        "skip_completed_runs": bool(sweep.get("skip_completed_runs", True)),
        "train_args": require_table(train, "args", context="[train]"),
        "train_grid": train_grid,
        "train_flags": train.get("flags", {}),
        "eval_args": require_table(evaluate, "args", context="[evaluate]"),
        "eval_flags": evaluate.get("flags", {}),
    }


def sanitize_value(value):
    text = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in text)


def iter_sweep_configs(sweep_grid):
    keys = list(sweep_grid.keys())
    values = [sweep_grid[key] for key in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def build_run_slug(train_args):
    signature = hashlib.sha1(
        json.dumps(train_args, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    data = sanitize_value(train_args.get("data", "data"))
    arch = sanitize_value(train_args.get("arch", "arch"))
    norm = sanitize_value(train_args.get("norm", "batch"))
    return (
        f"sk{train_args['sketch_size']}_sel{train_args['selected_patches']}"
        f"_np{train_args['num_patches']}_bs{train_args['bs']}"
        f"_lr{sanitize_value(train_args['lr'])}_cov{sanitize_value(train_args['cov_weight'])}"
        f"_{data}_{arch}_{norm}_{signature}"
    )


def build_train_args(base_args, combo_args):
    train_args = dict(base_args)
    train_args.update(combo_args)
    run_slug = build_run_slug(train_args)
    base_msg = sanitize_value(train_args.get("msg", "SWEEP"))
    signature = run_slug.rsplit("_", 1)[-1]
    train_args["msg"] = f"{base_msg}_{signature}"
    return train_args, run_slug


def patchsketch_run_dir(train_args):
    return REPO_ROOT / "logs" / str(train_args["dir"]) / (
        f"sketch{train_args['sketch_size']}_topk{train_args['selected_patches']}"
        f"_norm{train_args['norm']}_numpatch{train_args['num_patches']}"
        f"_bs{train_args['bs']}_lr{train_args['lr']}"
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
    cli_args = parse_args()
    config = load_sweep_config(cli_args.config)
    output_root = config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_root / "summary.csv"
    summary_rows = []
    started_at = datetime.now().isoformat(timespec="seconds")
    print(
        f"Starting PatchSketch sweep '{config['name']}' at {started_at} "
        f"from {config['path']}"
    )

    for run_index, combo_args in enumerate(
        iter_sweep_configs(config["train_grid"]), start=1
    ):
        train_args, run_slug = build_train_args(config["train_args"], combo_args)
        run_output_dir = output_root / run_slug
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
        row.update(config["eval_args"])
        summary_rows.append(row)

        if config["skip_completed_runs"] and eval_results_json_path.exists():
            row["status"] = "skipped_existing"
            row.update(load_eval_metrics(eval_results_json_path))
            print(f"[{run_index}] Skipping existing run {run_slug}")
            write_summary_csv(summary_rows, summary_csv_path)
            continue

        train_command = [sys.executable, "main_patchsketch.py"]
        train_command.extend(build_cli_args(train_args, config["train_flags"]))
        eval_command = [sys.executable, "evaluate.py"]
        eval_arg_map = dict(config["eval_args"])
        eval_arg_map["norm"] = train_args["norm"]
        eval_arg_map["model_path"] = str(checkpoint_path)
        eval_arg_map["results_json"] = str(eval_results_json_path)
        eval_command.extend(build_cli_args(eval_arg_map, config["eval_flags"]))

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
                {"CUDA_VISIBLE_DEVICES": config["train_cuda_visible_devices"]},
                train_log_path,
            )

            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Expected checkpoint not found: {checkpoint_path}")

            print(f"[{run_index}] Evaluating {run_slug}")
            run_and_tee(
                eval_command,
                {"CUDA_VISIBLE_DEVICES": config["eval_cuda_visible_devices"]},
                eval_log_path,
            )

            row["status"] = "completed"
            row.update(load_eval_metrics(eval_results_json_path))
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            write_summary_csv(summary_rows, summary_csv_path)
            if config["stop_on_failure"]:
                raise
        else:
            write_summary_csv(summary_rows, summary_csv_path)

    finished_at = datetime.now().isoformat(timespec="seconds")
    print(f"Finished PatchSketch sweep '{config['name']}' at {finished_at}")
    print(f"Summary written to {summary_csv_path}")


if __name__ == "__main__":
    main()
