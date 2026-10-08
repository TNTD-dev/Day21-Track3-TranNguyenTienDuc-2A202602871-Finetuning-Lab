# Hugging Face dataset selection for B2 and B3

Checked 2026-10-07 against publisher cards and Hugging Face APIs.

## Recommendation

Use **openai/gsm8k**, configuration `main`, pinned at revision `740312add88f781978c0658806c59bc2815b9866`. Human-authored worked solutions and explicit numeric final labels are easier to audit than translated Numina examples or synthetic Orca solutions. Prepare a **learner-curated public mathematics subset**, not a novel private corpus. This changes the lab domain from Vietnamese ticket classification to English mathematics but does **not** establish a new distribution relative to Qwen pretraining.

The implementation agent prepared 250 train, 50 validation and 50 evaluation records with global exact/numeric-template deduplication and the actual Qwen tokenizer: reported maximum 368 tokens, `max_length=512`. These are implementation-agent measurements, not a publisher guarantee. See `docs/HF-DATASET-QUALITY-REVIEW.md` for the limited manual review and exclusions recommended before freezing the final corpus.

## Publisher facts

- **NuminaMath-1.5:** Apache-2.0; one `train` split with 896,215 rows. Fields: `problem`, `solution`, `answer`, `problem_type`, `question_type`, `problem_is_valid`, `solution_is_valid`, `source`, `synthetic`. The card describes CoT-formatted solutions collected from exercises, contests, PDFs and forums. Its `cn_k12` source has 268,819 rows, including 149,010 word problems. Some answers are `proof`, `notfound`, missing, symbolic or malformed, so an answer field alone does not guarantee automatic grading. The publisher discusses parsing issues in its predecessor and manual corrections for selected sources. [Publisher card](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5), [API](https://huggingface.co/api/datasets/AI-MO/NuminaMath-1.5).
- **NuminaMath-CoT:** Apache-2.0; card metadata lists 859,494 training rows and 100 test rows. Columns are `source`, `problem`, `solution`, and `messages`. Solutions contain reasoning but no dedicated answer column; extracting final answers adds risk. Sources include GSM8K and MATH. [Publisher card](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT), [API](https://huggingface.co/api/datasets/AI-MO/NuminaMath-CoT).
- **GSM8K:** MIT; `main` and `socratic` configurations each have 7,473 train and 1,319 test rows. `question` and `answer` contain a word problem and a worked solution ending with `####` numeric answer. Problems require 2–8 elementary calculation steps. Its compact, explicit solutions make it the easiest fallback; it is also a long-established public benchmark, so unseen-pretraining claims would be unjustified. [Publisher card](https://huggingface.co/datasets/openai/gsm8k), [Original repository](https://github.com/openai/grade-school-math), [API](https://huggingface.co/api/datasets/openai/gsm8k).

All three API responses report `private=false` and `gated=false`; no access approval was needed. Observed revisions: NuminaMath-1.5 `1b05109f9e5c1ad06c0663519502416c30b300f8`; CoT `9d8d210c9f6a36c8f3cd84045668c9b7800ef517`; GSM8K `740312add88f781978c0658806c59bc2815b9866`.

## Proposed local curation and experiment

These are experiment-design recommendations, not publisher claims:

1. Load pinned GSM8K `main`; use upstream train for learner train/validation and upstream test for held-out evaluation. Preserve source split and row identifiers.
2. Extract the final numeric label after `####`; retain the original answer and human solution. Reject ambiguous questions, malformed labels and inconsistent reasoning. Remove calculator markup only, without inventing solution steps. Record reviewed IDs and actual review counts.
3. Tokenize the complete rendered chat, including authentic solution and final answer, with the actual Qwen tokenizer. Keep only samples fitting `max_length<=1024` without truncation. Do not shorten or invent a reasoning trace to meet the limit. Reserve output budget during generation.
4. Target at least 250 training examples, 50 validation examples and 50 held-out test examples. If insufficient examples survive, scan more rows; report measured accepted counts and token statistics, never infer them from the raw dataset size.
5. Normalize Unicode, whitespace and case, then deduplicate question hashes **before** assigning splits with a fixed seed. Also check near duplicates and duplicate worked solutions; group repeated questions together. Publish split hashes, accepted source-row IDs, rejection counts and source revision.
6. Preserve the original `solution` as the trace and the independently extracted `answer` as final response. Render both masks from the exact same examples, model, prompt and template: assistant-only supervises trace plus answer; response-only masks the trace. Inspect the actual token masks and verify nonempty trace and answer spans.
7. Freeze evaluation and decoding settings before either training run. Compare numeric final-answer accuracy, trace-format validity, latency, loss and regression under equal step budgets. A syntactically valid trace is not proof of mathematically valid reasoning; label `valid_trace_rate` accordingly and retain outputs for manual review.

## Limits that must appear in the report

Train/test decontamination here establishes separation within our experiment. It cannot establish absence from model pretraining. NuminaMath contains historical public material and multiple upstream sources, so cross-source duplicates remain possible. A new domain relative to the lab's supplied ticket data is a distribution change for this lab, not necessarily a novel distribution for the base model. Report this limitation explicitly; if the rubric requires base-model novelty, this public corpus alone does not prove it.

The remaining implementation gate is a measured scan establishing enough short, correct, numerically gradeable examples. Dataset size and length ranges alone do not establish suitability for a 4B model on T4. No GPU experiment was run as part of this research.

## Follow-up source audit

The [NuminaMath row API at offset 650000](https://datasets-server.huggingface.co/rows?dataset=AI-MO%2FNuminaMath-1.5&config=default&split=train&offset=650000&length=10) exposes `orca_math` examples with readable English worked solutions and separate numeric answers. Rows 650004 (plates, 27), 650005 (marbles, 48), 650007 (age difference, 7) and 650008 (spacing, 2.3) have straightforward derivations. However, row 650003 uses invalid proportionality for an alloy buoyancy problem; row 650006 has visibly incorrect sleep-duration arithmetic; row 650009 asks several questions but the answer field covers only the final result. All carry `Yes` validity labels. Thus, matching the last boxed answer and checking metadata cannot establish correctness. Reject ambiguous, multipart and suspect solutions; audit arithmetic and reasoning before admitting examples.

Microsoft's [original Orca-Math card](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) identifies GPT-4-Turbo-generated answers and synthetic expansion of questions, MIT licensing, and upstream Lila/DMath sources. Preserve this provenance when using the Numina-derived subset: these are genuine dataset-provided **teacher-generated worked solutions**, not human ground-truth traces or observed internal reasoning. Preserve them without fabricating additional steps. Neither the original card nor Numina establishes absence from Qwen pretraining.
