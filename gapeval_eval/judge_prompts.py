"""Verbatim judge-prompt templates from the GapEval paper (gapEval.pdf, Tables 4-11).

Each template is filled with [Question] / [Answer] text; [Image] and [Reference_Image] are
attached as separate image inputs to the vision-capable judge model (see judge.py), not
string-substituted.

Direction semantics (see paper Sec 2.3 + Appendix E.1):
  - understanding: [Image] = question image, [Answer] = the model's TEXT response.
  - generation:    [Image] = the model's GENERATED image, [Answer] = reference text answer.
In both directions [Question] is the direction-specific prompt (und_prompt / gen_prompt) and
[Reference_Image] is the ground-truth image when the item has one (omitted otherwise).
"""

UNDERSTANDING_WORLD_KNOWLEDGE = """[Image]
[Reference_Image]
Here is the question:{question}
Here is the answer:{answer}
Please judge the correctness of the answer. You should follow the following rules:
1. It includes the core information present in the [Answer]. If it does not contain the content of the reference text, judge as "not".
2. It reasonably describes the main subject and scene shown in the [Reference_Image].
3. It does not need to give an exhaustive or detailed account of every feature in the image.
4. Omissions or variations are acceptable, as long as the text covers the essential elements stated in the reference and matches the main content of the image.
5. Only if the generated text misses the core information of the reference or fails to describe the main subject of the image should it be judged as "not"."""

GENERATION_WORLD_KNOWLEDGE = """[Image]
[Reference_Image]
Here is the question:{question}
Here is the answer:{answer}
Please judge the correctness of the answer. You should follow the following rules:
1. Compare the generated image [Image] with the reference image [Reference_Image] and the caption [Question], and decide whether the image should be judged as pass (score 1) or fail (score 0).
2. If [Image] and [Reference_Image] are identical or extremely similar (i.e., contain visual regions that look directly copy-pasted with the same pixels, appearance, texture, and details), you must judge this as plagiarism and assign score 0; this plagiarism check has the highest priority and only original, newly generated images may pass.
3. Judge as pass if [Image] clearly presents the main subject, core scene, and key information required by [Reference_Image] and [Question]; exact reproduction of every element, attribute, arrangement, or color is not necessary, and differences in style, details, or smaller elements are allowed.
4. Judge as fail (score 0) if [Image] misses or seriously misinterprets the core content, main objects, or key semantics described in [Reference_Image] or [Question], or if it obviously contradicts the caption or omits elements that must be strictly matched.
5. Treat minor differences and reasonable variations as acceptable as long as the overall main information, semantics, and scene still match [Question], but never override the anti-plagiarism rule when making the final decision."""

UNDERSTANDING_REASONING = """[Image]
[Reference_Image]
Here is the question:{question}
Here is the answer:{answer}
Please judge the correctness of the answer. You should follow the following rules:
1. You are given a reasoning problem [Question], an authoritative reference_answer (the expected result or phenomenon), and a model-generated answer [Answer]. Your goal is to determine whether the final outcome/result expressed in [Answer] matches the reference_answer.
2. Check only whether [Answer] actually provides a final result/answer to the problem; ignore any reasoning, formulas, mechanisms, or intermediate steps when making the judgment.
3. Treat differences in wording, phrasing, or format between [Answer] and the reference_answer as acceptable, as long as they clearly describe the same final physical outcome or phenomenon.
4. If the final result in [Answer] is present and matches the reference_answer, mark it as correct (score = 1), even if the explanation, derivation, or mechanism is incomplete or physically incorrect.
5. If the final result in [Answer] contradicts, omits, or fails to provide the expected outcome described by the reference_answer, mark it as incorrect (score = 0)."""

GENERATION_REASONING = """[Image]
[Reference_Image]
Here is the question:{question}
Here is the answer:{answer}
Please judge the correctness of the answer. You should follow the following rules:
1. Compare [Image] with [Reference_Image] and [Answer] and check whether all required answer-relevant elements are present: the main result, key objects, and core information needed to visually answer the physics question posed by the problem. Major answer-relevant objects must not be missing or clearly misrepresented.
2. Examine each main object and its physical configuration: position, alignment, orientation, relative height, order, distance, contact, and any changes such as addition, removal, joining, splitting, or shape transformation. Verify that every expected answer-relevant object from [Reference_Image] is properly accounted for in [Image].
3. Check the physical relationships and processes: connections, supports, flows, force directions, movements and events. The depiction in [Image] must be logically and physically plausible and reflect the transformation or event described in [Answer]. Any critical new object that changes the expected physical outcome should cause rejection.
4. Accept minor differences in style, color, artistic rendering, and irrelevant extra objects that do not change the physical result. Focus on whether the main result and key scientific meaning match [Answer] and whether the similarity between [Image] and [Reference_Image] is correct in terms of physics outcome, not in minor visual detail.
5. Provide reasoning that clearly explains matches and differences for the above aspects. Assign "yes" (score = 1) only if the main physical result and all crucial answer elements match [Answer]; otherwise assign "no" (score = 0)."""

UNDERSTANDING_NUMERICAL_PERCEPTION = """[Image]
[Reference_Image]
Here is the question:{question}
Here is the answer:{answer}
Please judge the correctness of the answer. You should follow the following rules:
1. Use the JSON task in [Question] (its "objects" and "number" fields) as the authoritative specification of which object types and exact counts are required. The final confirmed result stated in [Answer] must include only those specified object types, with counts that exactly match the JSON, and must not introduce any extra or non-specified objects.
2. Count objects as individuals only when they are clearly and unambiguously described, with object-specific physical or functional features that match the corresponding JSON class (e.g., shape, color, material, labeling, size, context). Exclude partial, ambiguous, grouped, or hybrid/fused objects from all class counts, and explicitly note any such cases in the "reason" field instead of counting them.
3. Verify that each target object's final count in [Answer] matches the JSON exactly: no overcount, undercount, or mislabeling. Check for double counting where one object might be described multiple ways, and ensure that all required target objects are present, correctly identified, and not confused with other types.
4. Distinguish between analysis and final result: [Answer] may discuss or analyze non-target objects while reasoning, but the final confirmed result it reports must refer only to the JSON-specified object types and their counts. Any factual inconsistency or contradiction between the descriptive content and the final numbers/types should be treated as an error.
5. In the "reason" field, detail all findings, including how counts were derived, any ambiguities, hybrids, or errors. Assign score = 1 only if the final confirmed result in [Answer] exactly matches the object types and numbers in the JSON, with no extra objects in the claimed result; otherwise assign score = 0."""

GENERATION_NUMERICAL_PERCEPTION = """[Image]
[Reference_Image]
Here is the question:{question}
Here is the answer:{answer}
Please judge the correctness of the answer. You should follow the following rules:
1. Use the JSON specification in [Question] (its "objects" list and corresponding "number" fields) as the exact target: [Image] passes (score = 1) only if every specified object type appears with exactly the required quantity and class, regardless of other non-target real objects that may be present.
2. Count only distinct, real 3D object instances that fully and clearly match the visual features of their type (shape, color, label, typical context, etc.). Do not count drawings, icons, symbolic representations, or misrepresented/fused objects; each instance in overlaps, stacks, or crowds must still be individually countable and unambiguously identifiable.
3. Assign each counted instance to the correct class exactly once: do not double count the same item due to reflections, shadows, or repeated renderings, and treat any mislabeling (e.g., calling a notebook a dictionary) or hybrid objects (e.g., "dictionary-notebook" blends) as errors that must not contribute to any class's count.
4. Accept variations in appearance, design, pose, perspective, or partial occlusion as long as the object's identity remains clear; exclude partial or ambiguous cases where identity is uncertain. Ensure that each counted instance is classified to the single most suitable type and that no object is counted or classified more than once.
5. In the "reason" field, provide a concise but detailed explanation of the match/mismatch logic, documenting any missed counts, misidentifications, fused or uncountable objects, or double counting. Assign score = 1 only if all specified object types match their required quantities and classes exactly under these rules; otherwise assign score = 0."""

UNDERSTANDING_INSTRUCTION_FOLLOWING = """[Image]
[Reference_Image]
Here is the question:{question}
Here is the answer:{answer}
Please judge the correctness of the answer. You should follow the following rules:
1. From [Question], understand the rule that modifies the image scenario and the reference text that concisely describes the expected result or core feature after this rule is applied. Identify the core aspect, feature, or outcome that must appear once the rule is in effect.
2. Read [Answer] and extract all relevant features or changes it describes. Focus only on meaning: ignore extra detail, background information, unrelated content, length, and wording differences. Allow paraphrasing, scientific equivalence, and logical inference as long as the intended meaning can be reasonably matched.
3. Accept [Answer] as correct if it clearly or implicitly describes the core feature/result stated in the reference text, or if it shows a correct understanding and application of the core change introduced by the rule (either condition is sufficient). Minimal, direct answers are acceptable as long as the expected meaning is present.
4. Reject [Answer] if, after considering the rule, it omits or contradicts the intended meaning of the reference text, fails to reflect a correct understanding of the rule, or provides a different or incompatible interpretation of the rule. In ambiguous cases, accept only when the expected result can still be reasonably inferred from [Answer].
5. Scoring: assign score = 1 only if [Answer] covers the core meaning of the reference text or reasonably reflects a correct understanding and application of the rule under the above conditions; otherwise assign score = 0."""

GENERATION_INSTRUCTION_FOLLOWING = """[Image]
[Reference_Image]
Here is the question:{question}
Here is the answer:{answer}
Please judge the correctness of the answer. You should follow the following rules:
1. From [Question], fully understand the rule's intent and logic, and how it is supposed to modify the original image (objects, features, or arrangements in the original scenario). Identify the core effect or result that must appear after the rule is applied.
2. Evaluate [Image] as the result after applying the rule to the original image: check whether all main features and modifications demanded by the rule are present, and whether any required objects or features have been unintentionally omitted. Focus on whether the rule's core meaning and result are clearly implemented, regardless of color, layout, style, or minor details.
3. Use [Reference_Image] only as a sanity-check for the expected outcome: it is not the only correct solution and should not be used to enforce aesthetic, spatial, or stylistic accuracy. Ignore differences in object position, artistic style, decoration, or other aspects that do not directly relate to the rule's modification.
4. Accept any plausible depiction as correct ("yes") if [Image] clearly implements the rule's effect on the original image and represents the required meaning/result, even when style or layout differ from [Reference_Image]. Reject ("no") if [Image] fails to implement the rule, omits required modifications, contradicts the rule's meaning, or shows a critical misunderstanding of the rule.
5. In the "reason" field, clearly explain your judgment logic, focusing on how the rule was or was not correctly applied to the original image and whether the final modification in [Image] matches the intended effect of the rule."""

# Secondary empirical-study prompts (Sec. 4 / Appendix E.3, Tables 12-13) -- knowledge
# injection/editing evaluation, not the main GapEval capability/gap scoring.
UNDERSTANDING_KNOWLEDGE = """[Image]
Here is the question:{question}
Here is the answer:{answer}
Please judge the correctness of the answer. You should follow the following rules:
1. Ensure the subject described in the [Answer] matches the subject in the ground_truth (whether it's an animal, object, person, etc.).
2. If the output_text and ground_truth both describe the basic features, position, state, or other relevant characteristics of the subject consistently, it is considered correct.
3. If there are differences in non-essential details (such as posture, angle, or state), these can be ignored, and it is still considered correct.
4. Only when the subject described in the output_text is entirely wrong (e.g., "cat" is described as "dog") should it be considered incorrect."""

GENERATION_KNOWLEDGE_EDIT = """[Image]
Here is the question:{question}
Here is the answer:{answer}
Please judge the correctness of the answer. You should follow the following rules:
1. Ensure the subject depicted in the [Image] is the same as the subject in the ground_truth (whether it's an animal, object, person, etc.).
2. If the [Image] clearly depicts the same main subject as the ground_truth, even if there are variations in its state, expression, angle, or other minor details, it is considered correct.
3. If the output_image is chaotic, unclear, or does not represent the subject described in the ground_truth at all, it will be considered incorrect.
4. Minor differences in non-essential features like mood, position, or posture are acceptable, as long as the subject is still clearly the same."""


TEMPLATES = {
    ("World Knowledge", "und"): UNDERSTANDING_WORLD_KNOWLEDGE,
    ("World Knowledge", "gen"): GENERATION_WORLD_KNOWLEDGE,
    ("Reasoning", "und"): UNDERSTANDING_REASONING,
    ("Reasoning", "gen"): GENERATION_REASONING,
    ("Numerical Perception", "und"): UNDERSTANDING_NUMERICAL_PERCEPTION,
    ("Numerical Perception", "gen"): GENERATION_NUMERICAL_PERCEPTION,
    ("Instruction Following", "und"): UNDERSTANDING_INSTRUCTION_FOLLOWING,
    ("Instruction Following", "gen"): GENERATION_INSTRUCTION_FOLLOWING,
}


def get_template(category: str, direction: str) -> str:
    try:
        return TEMPLATES[(category, direction)]
    except KeyError as e:
        raise ValueError(f"No judge prompt for category={category!r} direction={direction!r}") from e
