"""MIRT-MAP Gap Score, implementing gapEval.pdf Appendix A ("Detailed Metric Implementation").

Given judged und/gen outputs for one or more models, fits a 2D multidimensional IRT model
(text ability, image ability per model; one difficulty per direction) via Bayesian MAP with a
learned shared Gaussian prior coupling the two ability dimensions (Eq. 1, 6-11), then reports
the normalized capability gap with the co-success/co-failure reward-penalty adjustment
(Eq. 12-16).

Caveat on N=1 (evaluating only Show-o2, no baselines): the shared covariance Sigma in Eq. 10-11
is only informative when fit jointly across several models. With a single model there is nothing
to share, so this script falls back to a fixed identity-covariance prior N(0, I) in that case
(flagged in the output) -- the resulting Gap Score is still a valid MAP fit of that one model's
theta_text/theta_image, just without the paper's cross-model regularization benefit. Add more
models' judged/ directories via --model to get the full paper behavior.

Usage:
    python gap_score.py --model showo2_1.5b=outputs/showo2_1.5b [--model other=outputs/other ...]
"""
import argparse
import json
import os
from collections import defaultdict

import torch

CATEGORIES = ["World Knowledge", "Numerical Perception", "Instruction Following", "Reasoning"]


def load_scores(path: str) -> dict:
    """id -> averaged score in [0,1]"""
    scores = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            scores[row["id"]] = (row["score"], row["category"])
    return scores


def compute_counts(und_scores: dict, gen_scores: dict):
    """Per-category expected 2x2 counts (n_T✓I×, n_T×I✓, n_T✓I✓, n_T×I×), Eq. 2-5 precursor.

    Scores are treated as P(correct); for --samples=1 in judge.py these are exactly 0/1 and the
    expected counts reduce to the paper's exact binary counts. For --samples>1 this is a smooth
    generalization (expected joint counts under independence of the two judge draws)."""
    counts = defaultdict(lambda: {"TI": 0.0, "Ti": 0.0, "tI": 0.0, "ti": 0.0, "n": 0})
    ids = set(und_scores) & set(gen_scores)
    for i in ids:
        pt, cat = und_scores[i]
        pg, _ = gen_scores[i]
        for cat_key in (cat, "overall"):
            c = counts[cat_key]
            c["TI"] += pt * pg
            c["Ti"] += pt * (1 - pg)
            c["tI"] += (1 - pt) * pg
            c["ti"] += (1 - pt) * (1 - pg)
            c["n"] += 1
    return counts


def fit_mirt(counts_by_model: dict, steps: int = 2000, lr: float = 0.05, seed: int = 0):
    """counts_by_model: {model_name: {"TI":.., "Ti":.., "tI":.., "ti":.., "n":..}}
    Returns {model_name: {"theta_text":.., "theta_image":.., "beta_text":.., "beta_image":..}}"""
    torch.manual_seed(seed)
    names = list(counts_by_model)
    n_models = len(names)

    n_text_success = torch.tensor([counts_by_model[m]["TI"] + counts_by_model[m]["Ti"] for m in names])
    n_text_fail = torch.tensor([counts_by_model[m]["tI"] + counts_by_model[m]["ti"] for m in names])
    n_image_success = torch.tensor([counts_by_model[m]["TI"] + counts_by_model[m]["tI"] for m in names])
    n_image_fail = torch.tensor([counts_by_model[m]["Ti"] + counts_by_model[m]["ti"] for m in names])

    theta = torch.zeros(n_models, 2, requires_grad=True)
    beta = torch.zeros(2, requires_grad=True)
    mu = torch.zeros(2, requires_grad=True)
    use_shared_prior = n_models >= 3  # degenerate otherwise (Sigma unidentifiable from <3 points)
    if use_shared_prior:
        L_raw = torch.eye(2, requires_grad=True)
        params = [theta, beta, mu, L_raw]
    else:
        params = [theta, beta]

    opt = torch.optim.Adam(params, lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        p_text = torch.sigmoid(theta[:, 0] - beta[0])
        p_image = torch.sigmoid(theta[:, 1] - beta[1])
        eps = 1e-7
        ll = (
            n_text_success * torch.log(p_text + eps) + n_text_fail * torch.log(1 - p_text + eps)
            + n_image_success * torch.log(p_image + eps) + n_image_fail * torch.log(1 - p_image + eps)
        ).sum()

        if use_shared_prior:
            L = torch.tril(L_raw)
            L = L + torch.eye(2) * 1e-3  # keep positive definite
            Sigma = L @ L.T
            diff = theta - mu
            Sigma_inv = torch.inverse(Sigma)
            prior = -0.5 * torch.einsum("bi,ij,bj->b", diff, Sigma_inv, diff).sum()
            prior = prior - 0.5 * n_models * torch.logdet(Sigma)
            loss = -(ll + prior)
        else:
            # fixed N(0, I) prior per model (Eq. 1 without the learned-Sigma coupling)
            prior = -0.5 * (theta ** 2).sum()
            loss = -(ll + prior)

        loss.backward()
        opt.step()

    result = {}
    for idx, name in enumerate(names):
        result[name] = {
            "theta_text": theta[idx, 0].item(),
            "theta_image": theta[idx, 1].item(),
            "beta_text": beta[0].item(),
            "beta_image": beta[1].item(),
        }
    return result, use_shared_prior


def gap_from_theta(theta_text: float, theta_image: float, counts: dict,
                    lambda_fail: float = 2.0, lambda_succ: float = 2.0) -> float:
    """Eq. 12-16: normalized absolute gap with co-success/co-failure reward-penalty."""
    delta = theta_text - theta_image
    g_abs = abs(delta) / (1 + abs(delta))  # Eq. 13, in [0, 1)

    n = counts["n"]
    if n == 0:
        return g_abs * 100
    f_i = counts["ti"] / n   # co-failure rate
    s_i = counts["TI"] / n   # co-success rate

    def logit(x, eps=1e-6):
        x = min(max(x, eps), 1 - eps)
        return torch.log(torch.tensor(x / (1 - x))).item()

    def sigmoid(x):
        return 1 / (1 + pow(2.718281828, -x))

    adjusted_logit = logit(g_abs) + (lambda_fail * f_i - lambda_succ * s_i)
    g_adjusted = sigmoid(adjusted_logit)
    return g_adjusted * 100  # normalized to [0, 100]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True,
                        help="name=output_dir, repeatable. output_dir must contain und_judged.jsonl and gen_judged.jsonl")
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    model_dirs = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        model_dirs[name] = path

    und_by_model = {m: load_scores(os.path.join(d, "und_judged.jsonl")) for m, d in model_dirs.items()}
    gen_by_model = {m: load_scores(os.path.join(d, "gen_judged.jsonl")) for m, d in model_dirs.items()}

    counts_by_model_cat = {m: compute_counts(und_by_model[m], gen_by_model[m]) for m in model_dirs}

    for cat in CATEGORIES + ["overall"]:
        counts_by_model = {m: counts_by_model_cat[m].get(cat, {"TI": 0, "Ti": 0, "tI": 0, "ti": 0, "n": 0})
                           for m in model_dirs}
        if all(c["n"] == 0 for c in counts_by_model.values()):
            continue
        fit, used_shared_prior = fit_mirt(counts_by_model, steps=args.steps)
        print(f"\n=== {cat} ===  (shared-prior Sigma {'learned' if used_shared_prior else 'fixed N(0,I) -- fewer than 3 models'})")
        for m in model_dirs:
            theta = fit[m]
            n = counts_by_model[m]["n"]
            succ = counts_by_model[m]["TI"] / n * 100 if n else 0.0
            und_acc = (counts_by_model[m]["TI"] + counts_by_model[m]["Ti"]) / n * 100 if n else 0.0
            gen_acc = (counts_by_model[m]["TI"] + counts_by_model[m]["tI"]) / n * 100 if n else 0.0
            gap = gap_from_theta(theta["theta_text"], theta["theta_image"], counts_by_model[m])
            print(f"  {m}: n={n} Succ={succ:.2f} Und={und_acc:.2f} Gen={gen_acc:.2f} Gap={gap:.2f} "
                 f"(theta_text={theta['theta_text']:.3f}, theta_image={theta['theta_image']:.3f})")


if __name__ == "__main__":
    main()
