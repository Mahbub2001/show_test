"""One-shot orchestrator: Show-o2 inference on GapEval -> vision-LLM judging -> Gap Score.

Thin wrapper around showo2_runner.py, judge.py, gap_score.py so the whole pipeline is a single
command on the 3090 box. Each stage is also runnable standalone (see README.md) -- useful since
inference (GPU-bound) and judging (API-bound, needs GEMINI_API_KEY or OPENAI_API_KEY) are
naturally separate steps you may want to re-run independently (e.g. re-judge without
re-generating images).

Usage:
    python run_eval.py --config configs/showo2_1.5b_demo_432x432.yaml \
                       --name showo2_1.5b --limit 20   # smoke test
    python run_eval.py --config configs/showo2_1.5b_demo_432x432.yaml \
                       --name showo2_1.5b               # full run, Gemini 2.5 Flash judge (default)
    python run_eval.py --config configs/showo2_1.5b_demo_432x432.yaml \
                       --name showo2_1.5b --judge_provider openai   # judge with GPT-5-mini instead
"""
import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHOWO2_ROOT = os.path.dirname(SCRIPT_DIR)

from dotenv import load_dotenv  # noqa: E402
load_dotenv()  # picks up show-o2/.env (searches this dir and parents)
# Keep HF downloads inside show-o2/ -- propagates to the showo2_runner.py subprocess below
# since subprocess.run() inherits the parent's os.environ by default.
os.environ.setdefault("HF_HOME", os.path.join(SHOWO2_ROOT, "hf_cache"))


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a showo2_*.yaml config (relative to show-o2/)")
    parser.add_argument("--name", required=True, help="Run name, used as outputs/<name>/")
    parser.add_argument("--data_dir", default=os.path.join(SCRIPT_DIR, "data"))
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N items (smoke test)")
    parser.add_argument("--weight_type", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--judge_provider", choices=["gemini", "openai"], default="gemini")
    parser.add_argument("--judge_model", default=None, help="Defaults to gemini-2.5-flash / gpt-5-mini per provider")
    parser.add_argument("--judge_samples", type=int, default=1, help="Judge repeats per item (paper uses 10)")
    parser.add_argument("--skip_inference", action="store_true", help="Reuse existing outputs, only (re-)judge")
    parser.add_argument("--skip_judge", action="store_true", help="Only run inference, skip judging/gap score")
    args = parser.parse_args()

    output_dir = os.path.join(SCRIPT_DIR, "outputs", args.name)
    os.makedirs(output_dir, exist_ok=True)

    if not args.skip_inference:
        cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "showo2_runner.py"),
            "--config", args.config, "--data_dir", args.data_dir, "--output_dir", output_dir,
            "--direction", "both", "--weight_type", args.weight_type,
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        run(cmd)

    if not args.skip_judge:
        required_key = "GEMINI_API_KEY" if args.judge_provider == "gemini" else "OPENAI_API_KEY"
        if required_key not in os.environ:
            raise SystemExit(f"{required_key} not set -- required for judge.py --provider {args.judge_provider}.")
        cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "judge.py"),
            "--data_dir", args.data_dir, "--output_dir", output_dir, "--provider", args.judge_provider,
            "--direction", "both", "--samples", str(args.judge_samples),
        ]
        if args.judge_model:
            cmd += ["--model", args.judge_model]
        run(cmd)
        run([
            sys.executable, os.path.join(SCRIPT_DIR, "gap_score.py"),
            "--model", f"{args.name}={output_dir}",
        ])


if __name__ == "__main__":
    main()
