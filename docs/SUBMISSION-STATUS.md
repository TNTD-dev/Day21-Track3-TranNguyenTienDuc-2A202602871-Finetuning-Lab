# Submission status

Report and artifacts are submitted as Option B. Core JSON/CSV come from the student's Drive results ZIP; bonus summary and executed notebook come from Kaggle; dataset checksums match the curated corpus. No additional GPU experiments were run while writing the report.

Evidence limitations:
- Core baseline per-ticket predictions were not saved. The qualitative table compares recorded rank8/rank64 predictions; it does not pretend to prove two losses versus the optimized prompt.
- Full Kaggle prediction/adapter ZIP was not supplied. Baseline output-completion/truncation is a hypothesis based on trace rate, generation budget and logs, not a verified token-by-token diagnosis.
- B1 hot-swap completion was provided as successful subprocess execution; its original hotswap JSON was lost. Merge JSON is preserved.
- Public Hub adapter comes from the earlier Colab run, distinguished in the report.

The mechanical verifier does not certify every qualitative rubric requirement or guarantee 100+15 points.
