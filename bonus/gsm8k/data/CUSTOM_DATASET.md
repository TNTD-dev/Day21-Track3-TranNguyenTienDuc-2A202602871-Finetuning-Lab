# Custom dataset — GSM8K numeric reasoning subset

Source: https://huggingface.co/datasets/openai/gsm8k
Pinned revision: `740312add88f781978c0658806c59bc2815b9866`. License: MIT. Original authors: OpenAI GSM8K.
Official source documentation: https://github.com/openai/grade-school-math

Domain: English grade-school math word problems, adapted to reasoning traces followed
by a JSON numeric answer. The original human-written solution is retained; calculator
annotations `<<...>>` are removed as text, never executed. Answers after `####` are
normalized using exact rational arithmetic. Source split and row index are recorded.

Counts: 250 train, 50 validation, 50 eval. Eval comes only from the official test split.
Exact normalized questions and templates with numeric literals replaced are deduplicated
across all three sets before writing. This checks experimental leakage, not all semantic
paraphrases. SHA-256 checksums freeze the selected files before any training.

Before freezing, a manual review of 20 train and 10 eval candidates identified ambiguity,
inconsistent explanatory prose, and unit/wording issues. Excluded source train rows
3341, 7373, 5032 and test rows 389, 1313, replacing them via the same deterministic
sampling procedure. This is a sample review, not a claim that all 350 rows were manually
verified. See SOURCE_QUALITY_REVIEW.md for the actual findings.

Every rendered example is at most 512 tokens; none is clipped. Both mask modes
have a nonzero answer loss, exclude the question, and have different supervised spans.
See results/mask_proof.json and results/token_stats.json.

This is a learner-curated subset of published data, not original private data. It is a
different task/domain/response format from the core CSKH corpus. We cannot establish
that it is unseen by Qwen during pretraining; baseline results and this limitation must
be reported honestly. Inspect REVIEW_SAMPLES.md; source annotations are not a guarantee
of perfect labels or faithful model reasoning.
