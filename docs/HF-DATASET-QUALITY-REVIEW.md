# GSM8K local sample quality review

Reviewed **30 records**: first 20 in `.lavish/bonus_gsm8k/data/train.jsonl` and first 10 in `eval_target.jsonl`. This is a limited manual arithmetic and ambiguity review, **not a manual audit of all 350 records**. Files were not changed by this reviewer. Row numbers below refer to the preserved upstream source split/row at pinned revision `740312add88f781978c0658806c59bc2815b9866`.

## Recommended exclusions before freezing

- **train row 3341**, ID `9a7d95af461725130bbc7cac77fb6a8b35d002d2c1ea6175bc385d403d4fe8c6`: COVID cases. The third-week wording says “2000 more cases” without a comparison reference. Label 9500 treats 2000 as the third week's absolute count (5000+2500+2000); a natural comparison with week 2 makes week 3 equal 4500 and total 12000. The arithmetic is internally consistent with the label but the question is ambiguous. Exclude rather than rewrite the original dataset silently.
- **train row 7373**, ID `ef1a365a6d5a7d5e6e3b9eca5e5415836dfc91ec17cfe2e87b93cee502108c45`: reading time. Final answer 810 minutes is correct: weekdays 5×90=450, weekend 2×180=360. Earlier solution computes weekday morning 150, weekend morning 120 and all evenings 540. However, the final explanatory sentence incorrectly calls 120 the evenings and 540 the weekend. Exclude for a reasoning-trace comparison, or record a transparent editorial correction separately; do not portray this prose as a clean trace.

## Minor assumptions and wording issues

- **train row 5032**, ID `329ba6ef9541ba9279a148fe6b123ea05178961a91b2b49fdca07b4d811209a2`: battery charge time 66 assumes devices are charged sequentially and half-charge takes half the full-charge time. Intended elementary model is clear, but real chargers need not behave linearly. Optional exclusion if aiming for unusually strict unambiguous data.
- **train row 1914** uses watts instead of energy units for an electricity bill. Label 1350 follows the supplied unit price and quantity; physical terminology is simplified.
- **eval/test row 389**, ID `17a31b91347c9a1d770b6d8e145169979de2a5a0e2fd649f5ad70dbd0d46a608`: typo “bathing” rather than buying the bracelet; numerical reasoning and label 6 agree.
- **eval/test row 1313**, ID `5f00ebb4858593f13433757d80c2678a6bb7541805438f9b090b0bc3d43c6d3d`: writes `$100/$50` although the divisor is 50 watermelons. Numerical result 2 per melon is correct.

The remaining checked records had coherent numerical results under ordinary elementary word-problem assumptions. This statement does not establish correctness of unchecked records.

## Reproducible checked rows

Train: 1458, 3620, 6312, 4833, 5032, 1914, 1258, 5173, 3457, 3341, 1740, 1770, 6698, 4207, 7373, 5862, 3248, 5189, 4307, 4225.

Eval/upstream test: 1010, 765, 961, 389, 321, 1313, 764, 822, 811, 183.

Source provenance: [GSM8K publisher card](https://huggingface.co/datasets/openai/gsm8k), [pinned repository](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866). Public dataset exposure during base-model pretraining remains unknown. Neither local deduplication nor this sample review proves a novel distribution to Qwen.
