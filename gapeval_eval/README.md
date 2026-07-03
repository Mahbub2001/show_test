# Evaluating Show-o2 on GapEval

Pipeline built here on a 4GB laptop GPU (setup only); intended to actually run on the
RTX 3090 24GB box. Evaluates Show-o2 with the [GapEval](https://huggingface.co/datasets/FrancisChen1/GapEval)
bidirectional benchmark from `gapEval.pdf`, reproducing the paper's Succ/Und/Gen/Gap metrics
(Table 2) for Show-o2 specifically.

## Layout

```
gapeval_eval/
  data.py            # loads+normalizes data/data_<id>/prompt.json into GapEvalItem
  data/               # downloaded GapEval repo (621 items; see "Data" below)
  showo2_runner.py    # runs Show-o2: mmu_generate (und) + t2i_generate (gen) per item
  judge_prompts.py    # verbatim per-category judge prompt templates (Tables 4-11)
  judge.py            # Gemini 2.5 Flash (default) or GPT-5-mini judge -> und_judged.jsonl / gen_judged.jsonl
  gap_score.py        # MIRT-MAP Gap Score (Appendix A) from judged outputs
  run_eval.py          # orchestrates the three stages
  outputs/<name>/      # per-run artifacts (created on first run)
```

## Data

Already downloaded into `data/` via:
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('FrancisChen1/GapEval', repo_type='dataset', local_dir='data')"
```
621 items (not the paper's 646 -- the public HF snapshot differs slightly), each a
`data_<id>/` folder with `prompt.json`, an `image/` question image, and sometimes a `ref/`
ground-truth image. Category taxonomy in the raw data (`reasoning`, `physics`, `Multi Hop`,
`counting`, `rule_based`) is mapped to the paper's 4 categories in `data.py:CATEGORY_MAP`
(`physics` folds into `Reasoning`, matching the paper's "Real-world Reasoning" subtype).

Run `python data.py` to re-verify the loader against whatever's in `data/`.

## Setup (on the 3090 box)

1. Create/activate a venv in `show-o2/` (`python -m venv venv`, then activate it).
2. **Install CUDA torch first, separately, before anything else:**
   ```bash
   pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
   ```
   Do NOT skip this step or let a later `pip install` pull torch from plain PyPI -- PyPI's
   default index only serves the CPU-only build, which silently gives you a working-but-CPU-bound
   setup with no error (verify with `python -c "import torch; print(torch.cuda.is_available())"`
   -- must print `True` before proceeding).
3. `pip install -r gapeval_eval/requirements.txt` (this also pins `transformers==4.47.0`,
   `diffusers==0.31.0` etc. to match `../build_env.sh` -- show-o2's `models/qwen2.py` and other
   files are hand-modified copies of upstream code frozen to that API snapshot; a newer
   `transformers` breaks them, e.g. `Qwen2Config` losing `rope_theta`).
4. Download the Wan2.1 3D Causal VAE weights into `show-o2/` (not `gapeval_eval/`):
   ```bash
   curl -L -o Wan2.1_VAE.pth https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/Wan2.1_VAE.pth
   ```
5. Copy `../.env.example` to `../.env` and fill in `GEMINI_API_KEY` (default judge provider,
   `gemini-2.5-flash`) or `OPENAI_API_KEY` if you'll pass `--judge_provider openai`
   (`gpt-5-mini`). Both are vision-capable and validated against each other in the paper's
   Appendix B (Pearson r=0.9656 between GPT-5-mini and Gemini3-Flash judgments) -- either is a
   faithful choice.
6. The show-o2 checkpoint (`showlab/show-o2-1.5B` by default, see the config's
   `model.showo.pretrained_model_path`) downloads automatically from HF on first run.

## Running

Smoke test first (20 items, cheap):
```bash
python run_eval.py --config configs/showo2_1.5b_demo_432x432.yaml --name showo2_1.5b_smoke --limit 20
```

Full run (Gemini 2.5 Flash judge by default):
```bash
python run_eval.py --config configs/showo2_1.5b_demo_432x432.yaml --name showo2_1.5b
```
Use GPT-5-mini instead:
```bash
python run_eval.py --config configs/showo2_1.5b_demo_432x432.yaml --name showo2_1.5b --judge_provider openai
```
For the 7B checkpoint, swap in `configs/showo2_7b_demo_432x432.yaml` (fits comfortably in 24GB
for inference).

This runs, in order:
1. `showo2_runner.py` -- for every item: `und_prompt` + question image -> text (mmu_generate);
   `gen_prompt` -> image (t2i_generate + flow-matching ODE sampler). Writes
   `outputs/<name>/und_outputs.jsonl`, `outputs/<name>/gen_outputs.jsonl` and
   `outputs/<name>/gen_images/<id>.png`.
2. `judge.py` -- the judge model scores each answer against the category-specific rubric from
   the paper. Writes `outputs/<name>/und_judged.jsonl`, `outputs/<name>/gen_judged.jsonl`.
3. `gap_score.py` -- fits the MIRT-MAP model and prints Succ/Und/Gen/Gap per category, matching
   the columns in the paper's Table 2.

Each stage is independently re-runnable, e.g. re-judge without re-generating:
```bash
python run_eval.py --config ... --name showo2_1.5b --skip_inference
```
Or run `judge.py` directly for finer control (provider/model/samples):
```bash
python judge.py --output_dir outputs/showo2_1.5b --provider gemini --model gemini-2.5-flash --samples 10
```

## Known limitations / deviations from the paper

- **No image-conditioned generation.** The publicly released Show-o2 checkpoints only expose
  unconditional text-to-image via `t2i_generate` (mixed-modality/image-conditioned generation is
  still unreleased per the show-o2 README TODO list). So for GapEval items whose `gen_prompt`
  asks to edit/transform the *given* question image (most of Instruction Following, some
  Reasoning/Physics items), `showo2_runner.py` generates from the text prompt alone -- the
  question image is not fed into the generation path. This will likely depress Show-o2's
  Instruction-Following Gen score relative to what a true image-editing pipeline would achieve;
  it reflects a real capability gap of the released model, not a bug in this harness.
- **621 vs 646 items.** The public HF dataset snapshot has 621 items; category proportions are
  broadly similar to the paper's Table listing but not identical.
- **Gap Score with N=1.** The paper's MIRT-MAP fits a *shared* Gaussian prior across several
  models' ability vectors (Eq. 10-11), which needs multiple models to be non-degenerate.
  Evaluating Show-o2 alone, `gap_score.py` automatically falls back to a fixed N(0, I) prior
  (flagged in its output). To reproduce the paper's exact cross-model regularization, run this
  harness against additional baselines and pass all of them to `gap_score.py --model name=dir`.
- **Judge sampling.** The paper averages 10 independent judge calls per item
  (`--judge_samples 10` here); this defaults to 1 for cost, bump it up for a paper-faithful run.
