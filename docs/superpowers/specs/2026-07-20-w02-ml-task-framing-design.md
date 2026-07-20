# ML-03 notebook design

## Scope

Create `work/notebooks/w02_ml_task_framing.ipynb` for the existing provisional lane: Refresh / Content Opportunity Scoring. The notebook is a framing exercise, not a model-training or causal-impact claim.

## Task framing

The task is ranking, supported by a starter-data classification proxy. The system would rank one anonymized content item per row for human review. The starter proxy is `trend_direction == "down"`; it is useful to explain the workflow but is not a future prediction target.

The primary success metric is Precision@20. It matches a team that can inspect about 20 candidates in a review cycle. A learned ranking earns use only if it improves this metric over a transparent baseline rule on a suitable holdout.

## Notebook structure

1. State the lane as a ranking task and distinguish it from its classification proxy.
2. Define `is_declining_proxy` in code and explain that neither `trend_direction` nor `trend_pct` can be a feature.
3. Explain Precision@20 and the capacity tradeoff it represents.
4. Load the starter CSV and display a small dataframe with one row per anonymized content item, an ID used only for grouping, safe candidate signals, and the proxy target.
5. Explain why learned ranking may improve on a fixed rule while retaining a rule as the baseline and retaining human review.
6. End with a self-check against the assignment requirements.

## Limits and safety

The output is a decision-support queue for a content strategist, not an instruction to edit a page. It cannot establish that a refresh causes recovery, prove a Google ranking factor, or predict a future outcome from this starter proxy. The later capstone should replace the proxy with a time-separated future target and use leakage-safe validation.

## Verification

Execute the notebook against `data/raw/content_refresh_anonymized.csv`. Confirm that its code cells have execution output, that the unit-of-analysis dataframe is visible, and that the committed notebook contains all six required sections.
