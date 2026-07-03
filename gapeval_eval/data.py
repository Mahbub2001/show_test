"""Loads and normalizes the GapEval dataset (FrancisChen1/GapEval on HF).

The HF repo is not a `datasets`-loadable table -- it's a raw file tree:

    data_<id>/prompt.json         # one dict, or a one-element list containing a dict
    data_<id>/image/<name>.jpg    # question image(s); referenced by img/image/input_image
    data_<id>/ref/<name>.png      # ground-truth generation image (only for some items)

`prompt.json` fields are inconsistent across categories (the released repo
predates a documented schema), so this module normalizes every item into a
single `GapEvalItem` with the fields the eval pipeline actually needs, while
keeping `raw` around for category-specific judge prompts (e.g. the counting
task's structured object/number spec).
"""
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

# Raw category strings in the dataset -> taxonomy used in the paper (Sec 2.2)
CATEGORY_MAP = {
    "reasoning": "Reasoning",
    "physics": "Reasoning",
    "Multi Hop": "World Knowledge",
    "counting": "Numerical Perception",
    "rule_based": "Instruction Following",
}

INPUT_IMAGE_KEYS = ("image", "img", "input_image")
REFERENCE_TEXT_KEYS = ("reference", "expect_text", "answer", "gt_prompt")


@dataclass
class GapEvalItem:
    id: int
    folder: str
    category: str
    raw_category: str
    subcategory: Optional[str]
    subsubcategory: Optional[str]
    und_prompt: str
    gen_prompt: str
    reference_text: Optional[str]
    input_image_path: Optional[str]
    reference_image_path: Optional[str]
    raw: dict = field(default_factory=dict)

    @property
    def question_context(self) -> str:
        """Extra structured context (e.g. counting's object/number spec) to
        splice into judge prompts as part of [Question]."""
        extra = {}
        if "task" in self.raw:
            extra["task"] = self.raw["task"]
        if "explain" in self.raw:
            extra["explain"] = self.raw["explain"]
        return json.dumps(extra, ensure_ascii=False) if extra else ""


def _first_present(d: dict, keys) -> Optional[Any]:
    for k in keys:
        if d.get(k):
            return d[k]
    return None


def _stringify(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _resolve_reference_image(folder: str, raw: dict) -> Optional[str]:
    ref_dir = os.path.join(folder, "ref")
    if os.path.isdir(ref_dir):
        files = sorted(os.listdir(ref_dir))
        if files:
            return os.path.join(ref_dir, files[0])
    expect_image = raw.get("expect_image")
    if expect_image:
        candidate = os.path.join(folder, expect_image)
        if os.path.isfile(candidate):
            return candidate
    return None


def _load_one(folder: str) -> GapEvalItem:
    with open(os.path.join(folder, "prompt.json"), encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        raw = raw[0]

    input_image_rel = _first_present(raw, INPUT_IMAGE_KEYS)
    input_image_path = os.path.join(folder, input_image_rel) if input_image_rel else None
    if input_image_path and not os.path.isfile(input_image_path):
        input_image_path = None

    raw_category = raw.get("category", "unknown")
    return GapEvalItem(
        id=raw["id"],
        folder=folder,
        category=CATEGORY_MAP.get(raw_category, raw_category),
        raw_category=raw_category,
        subcategory=raw.get("subcategory"),
        subsubcategory=raw.get("subsubcategory"),
        und_prompt=raw["und_prompt"],
        gen_prompt=raw["gen_prompt"],
        reference_text=_stringify(_first_present(raw, REFERENCE_TEXT_KEYS)),
        input_image_path=input_image_path,
        reference_image_path=_resolve_reference_image(folder, raw),
        raw=raw,
    )


def load_gapeval(data_dir: str) -> list[GapEvalItem]:
    """Load every data_<id>/prompt.json under `data_dir` (the snapshot_download target)."""
    folders = sorted(
        glob.glob(os.path.join(data_dir, "data_*")),
        key=lambda p: int(os.path.basename(p).split("_")[1]),
    )
    items = [_load_one(f) for f in folders]
    return items


if __name__ == "__main__":
    import argparse
    import collections

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "data"))
    args = parser.parse_args()

    items = load_gapeval(args.data_dir)
    print(f"Loaded {len(items)} items")
    cats = collections.Counter(i.category for i in items)
    print("Categories:", dict(cats))
    n_with_input_img = sum(1 for i in items if i.input_image_path)
    n_with_ref_img = sum(1 for i in items if i.reference_image_path)
    n_with_ref_text = sum(1 for i in items if i.reference_text)
    print(f"With input image: {n_with_input_img}, with reference image: {n_with_ref_img}, "
          f"with reference text: {n_with_ref_text}")
    print("Sample item:", items[0])
