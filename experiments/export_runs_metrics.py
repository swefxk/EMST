import argparse
import os
import subprocess
import sys


def run_cmd(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--ckpt_root",
        default="checkpoints/ablation_a012",
        help="Checkpoint directory with A0/A1/A2_seedX.pt files.",
    )
    parser.add_argument("--out_root", default="runs")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    settings = [
        ("A0", "off"),
        ("A1", "zero_dt"),
        ("A2", "on"),
    ]

    python = sys.executable
    for seed in seeds:
        for name, prev_event in settings:
            ckpt = os.path.join(args.ckpt_root, f"{name}_seed{seed}.pt")
            out_dir = os.path.join(args.out_root, name, f"seed_{seed}")
            os.makedirs(out_dir, exist_ok=True)
            out_metrics = os.path.join(out_dir, "metrics.json")
            cmd = [
                python,
                "eval.py",
                "--data",
                args.data,
                "--ckpt",
                ckpt,
                "--config",
                args.config,
                "--seed",
                str(seed),
                "--prev_event",
                prev_event,
                "--out_metrics",
                out_metrics,
            ]
            print(f"export {name} seed {seed} -> {out_metrics}")
            run_cmd(cmd)


if __name__ == "__main__":
    main()
