"""Vision-LLM judge for Show-o2's GapEval outputs (paper Sec. 2.3, Appendix E.1).

Supports two interchangeable judge providers -- both vision-capable, both used/validated by the
paper itself (Appendix B cross-validates GPT-5-mini against Gemini3-Flash, Pearson r=0.9656):
  --provider gemini (default)  model: gemini-2.5-flash   needs GEMINI_API_KEY
  --provider openai            model: gpt-5-mini          needs OPENAI_API_KEY

Reads:
  - <output_dir>/und_outputs.jsonl  (id, und_prompt, answer)      from showo2_runner.py
  - <output_dir>/gen_outputs.jsonl  (id, gen_prompt, image_path)  from showo2_runner.py
and writes:
  - <output_dir>/und_judged.jsonl  (id, category, score, reason)
  - <output_dir>/gen_judged.jsonl  (id, category, score, reason)

The paper runs 10 independent judge samples per item and averages; this script exposes
--samples for the same behavior (default 1 for a cheap smoke test).
"""
import argparse
import base64
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from data import GapEvalItem, load_gapeval  # noqa: E402
from judge_prompts import get_template  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
load_dotenv()  # picks up show-o2/.env (searches this dir and parents)

JUDGE_INSTRUCTIONS = (
    "You are a strict evaluation judge. Follow the rules exactly and respond with ONLY a JSON "
    'object of the form {"score": 0 or 1, "reason": "<one or two sentences>"}. No other text.'
)

DEFAULT_MODEL = {"gemini": "gemini-2.5-flash", "openai": "gpt-5-mini"}


def _b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _mime_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "jpeg"
    return f"image/{'jpeg' if ext == 'jpg' else ext}"


def build_prompt(item: GapEvalItem, direction: str, model_answer: str, generated_image_path: str = None):
    """Returns (prompt_text, [image_path, ...]) -- provider-agnostic."""
    template = get_template(item.category, direction)
    if direction == "und":
        question = item.und_prompt + ("\n" + item.question_context if item.question_context else "")
        answer = model_answer
        primary_image = item.input_image_path
    else:
        question = item.gen_prompt + ("\n" + item.question_context if item.question_context else "")
        answer = item.reference_text or ""
        primary_image = generated_image_path

    text = JUDGE_INSTRUCTIONS + "\n\n" + template.format(question=question, answer=answer)

    images = []
    if primary_image and os.path.isfile(primary_image):
        images.append(primary_image)
    if item.reference_image_path and os.path.isfile(item.reference_image_path):
        images.append(item.reference_image_path)

    return text, images


# ------------------------------------------------------------------ providers --
class OpenAIJudge:
    def __init__(self, model: str):
        from openai import OpenAI
        self.client = OpenAI()  # reads OPENAI_API_KEY from env
        self.model = model

    def call(self, prompt_text: str, image_paths: list) -> dict:
        content = [{"type": "text", "text": prompt_text}]
        for path in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{_mime_type(path)};base64,{_b64_image(path)}"},
            })
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)


class GeminiJudge:
    def __init__(self, model: str):
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) for --provider gemini")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self._types = __import__("google.genai.types", fromlist=["types"])

    def call(self, prompt_text: str, image_paths: list) -> dict:
        types = self._types
        parts = [types.Part.from_text(text=prompt_text)]
        for path in image_paths:
            with open(path, "rb") as f:
                parts.append(types.Part.from_bytes(data=f.read(), mime_type=_mime_type(path)))
        resp = self.client.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(resp.text)


def make_judge(provider: str, model: str):
    if provider == "openai":
        return OpenAIJudge(model)
    if provider == "gemini":
        return GeminiJudge(model)
    raise ValueError(f"Unknown provider {provider!r}")


# ------------------------------------------------------------------------------ --
def judge_one(judge, item: GapEvalItem, direction: str, model_answer: str,
              generated_image_path: str = None, retries: int = 3):
    prompt_text, image_paths = build_prompt(item, direction, model_answer, generated_image_path)
    last_err = None
    for attempt in range(retries):
        try:
            parsed = judge.call(prompt_text, image_paths)
            score = int(parsed.get("score", 0))
            reason = parsed.get("reason", "")
            return score, reason
        except Exception as e:  # noqa: BLE001 - retry on any transient API/parse error
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Judge failed for item {item.id} ({direction}) after {retries} tries: {last_err}")


def _run_direction(judge, items_by_id, output_dir, direction, samples):
    in_path = os.path.join(output_dir, f"{direction}_outputs.jsonl")
    out_path = os.path.join(output_dir, f"{direction}_judged.jsonl")
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            item = items_by_id[row["id"]]
            answer = row["answer"] if direction == "und" else None
            image_path = row.get("image_path")
            scores, reasons = [], []
            for _ in range(samples):
                s, r = judge_one(judge, item, direction, answer, image_path)
                scores.append(s)
                reasons.append(r)
            avg = sum(scores) / len(scores)
            fout.write(json.dumps({
                "id": item.id, "category": item.category, "subcategory": item.subcategory,
                "score": avg, "raw_scores": scores, "reasons": reasons,
            }, ensure_ascii=False) + "\n")
            print(f"[{direction}] id={item.id} cat={item.category} score={avg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(SCRIPT_DIR, "data"))
    parser.add_argument("--output_dir", required=True, help="Directory containing *_outputs.jsonl from showo2_runner.py")
    parser.add_argument("--provider", choices=["gemini", "openai"], default="gemini")
    parser.add_argument("--model", default=None, help="Defaults to gemini-2.5-flash / gpt-5-mini per provider")
    parser.add_argument("--direction", choices=["und", "gen", "both"], default="both")
    parser.add_argument("--samples", type=int, default=1, help="Judge repeats per item, averaged (paper uses 10)")
    args = parser.parse_args()

    model = args.model or DEFAULT_MODEL[args.provider]
    judge = make_judge(args.provider, model)
    items_by_id = {item.id: item for item in load_gapeval(args.data_dir)}

    if args.direction in ("und", "both"):
        _run_direction(judge, items_by_id, args.output_dir, "und", args.samples)
    if args.direction in ("gen", "both"):
        _run_direction(judge, items_by_id, args.output_dir, "gen", args.samples)


if __name__ == "__main__":
    main()
