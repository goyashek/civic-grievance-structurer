# model selection results

These are the saved outputs from the 40-case development bake-off. The six JSON files contain the model responses and summary metrics; `mlruns/` is the complete local MLflow file store for all six runs, and `run_timestamps.json` is a quick timestamp index.

The comparison used zero-shot and fixed few-shot prompts. It was used for model selection only, not for final test-set reporting. SmolLM3-3B was kept as the base model because it gave the strongest overall structured output with the lowest memory use among the candidates.

From the repository root, open the saved experiment with `mlflow ui --backend-store-uri data/model_selection_results/mlruns`.
