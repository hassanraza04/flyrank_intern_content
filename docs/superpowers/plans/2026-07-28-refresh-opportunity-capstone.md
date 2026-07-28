# Refresh Opportunity Scoring Capstone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a public-safe, reproducible refresh-opportunity model on the FlyRank warehouse and publish its research paper through GitHub Pages.

**Architecture:** DuckDB aggregates remote warehouse partitions into one local feature frame. A transparent momentum baseline and sklearn classifiers score the same time-aware cohorts. Public-safe JSON summaries and SVG charts feed a static HTML paper in `docs/`; no warehouse rows are committed.

**Tech Stack:** Python 3.11, DuckDB, pandas, scikit-learn, matplotlib, Jupyter notebooks, static HTML/CSS, GitHub Pages.

## Global Constraints

- Keep the reference `scripts/` pipeline unchanged. All capstone code belongs under `work/`.
- Use a Colab Secret or environment variable named `HF_TOKEN`. Never place a token in code, output, or git history.
- Query warehouse Parquet via `hf://`, aggregate in SQL, and cache only local ephemeral data under `work/outputs/`.
- Use `client_hash_id` and `content_hash_id` only for joins, grouping, and time-frame construction. Never include either as a feature or public output.
- Use `fact_content_daily_performance`, not the final-month `_sample` table, for feature development.
- Features must use only the two 28-day windows before a decision date. The label uses the later 28-day window.
- June 2026 is the sealed final outcome period. Do not tune the feature definition or model after reading its metrics.
- The paper must contain all nine required sections and credit “Built on the FlyRank ML Internship dataset” with a link to `https://flyrank.ai`.
- Do not publish client names, domains, URLs, raw queries, raw warehouse rows, pseudonymous item IDs, or credentials.

---

## File Structure

- `skills/README.md` and selected `skills/**/SKILL.md`: current official task guidance retained in the user repo.
- `work/scripts/capstone_utils.py`: pure feature-column, scoring, evaluation, and output-safety helpers.
- `work/tests/test_capstone_utils.py`: standard-library tests for the pure helper functions.
- `work/notebooks/w04_baseline_score.ipynb`: reproducible baseline and reason-code assignment.
- `work/notebooks/w05_model.ipynb`: time-aware model comparison and error analysis.
- `work/notebooks/w06_validation_audit.ipynb`: explicit leakage and sealed-test audit.
- `work/notebooks/w07_action_playbook.ipynb`: ranked actions and public-safe summary creation.
- `work/notebooks/capstone.ipynb`: final narrative notebook that mirrors the paper.
- `work/outputs/capstone_metrics.json`: committed metric receipt with no raw IDs.
- `work/outputs/capstone_summary.json`: committed public-safe counts and recommendation summaries.
- `work/figures/*.svg`: committed aggregate charts for the paper.
- `work/scripts/render_paper.py`: turns safe summaries and charts into `docs/index.html`.
- `docs/index.html`, `docs/assets/paper.css`, `docs/assets/*.svg`: GitHub Pages research paper.
- `submission/paper_url.txt`: created only after GitHub Pages provides the exact final URL; it contains one line.

## Cohort Interface

`work/scripts/capstone_utils.py` exposes these interfaces:

```python
FEATURE_COLUMNS = [
    "log_current_impressions", "impression_change_pct", "click_change_pct",
    "current_ctr", "current_avg_position", "position_change",
    "current_active_days",
]

FORBIDDEN_COLUMNS = {
    "client_hash_id", "content_hash_id", "cohort_id", "is_declining_proxy",
    "next_impressions", "future_impression_ratio",
}

def validate_feature_columns(columns: list[str]) -> None: ...
def add_baseline_score(frame: pd.DataFrame) -> pd.DataFrame: ...
def precision_at_k(labels: pd.Series, scores: pd.Series, k: int) -> float: ...
def evaluate_ranking(labels: pd.Series, scores: pd.Series, k: int = 100) -> dict[str, float]: ...
def reason_codes(row: pd.Series) -> list[str]: ...
def public_summary(frame: pd.DataFrame) -> dict: ...
```

The warehouse feature SQL produces this minimum schema before pandas receives it:

```text
cohort_id, client_hash_id, content_hash_id,
previous_impressions, current_impressions, next_impressions,
previous_clicks, current_clicks,
current_avg_position, previous_avg_position,
current_active_days, current_ctr,
log_current_impressions, impression_change_pct, click_change_pct,
position_change, is_declining_proxy
```

## Task 1: Import Current Official Guidance and Skeletons

**Files:**
- Create: `skills/README.md`
- Create: `skills/querying-big-datasets/SKILL.md`
- Create: `skills/building-baselines/SKILL.md`
- Create: `skills/training-honest-models/SKILL.md`
- Create: `skills/hunting-leakage-and-validating/SKILL.md`
- Create: `skills/writing-honest-claims/SKILL.md`
- Create: `skills/writing-research-papers/SKILL.md`
- Create: `skills/deploying-static-pages/SKILL.md`
- Create: `skills/flyrank/flyrank-data/SKILL.md`
- Create: `work/notebooks/w04_baseline_score.ipynb`
- Create: `work/notebooks/w05_model.ipynb`
- Create: `work/notebooks/w06_validation_audit.ipynb`
- Create: `work/notebooks/w07_action_playbook.ipynb`
- Create: `work/notebooks/capstone.ipynb`
- Create: `submission/README.md`

**Consumes:** The public official repository at commit `31baf1d60b12307bd5d37ce79e2797712293cecc`.

**Produces:** The current task skeletons and exact official guidance, without copying data files or changing the shared reference pipeline.

- [ ] **Step 1: Fetch and compare the official file list**

Run:

```bash
git archive --remote=https://github.com/flyrank-bih/flyrank-ml-internship-starter.git main work/notebooks 2>/dev/null
```

Expected: remote archive may be disabled. If it is, use GitHub’s file API for each listed file and preserve each file’s original contents.

- [ ] **Step 2: Add only the listed instruction and skeleton files**

Preserve notebook metadata and section order. Do not import `data/`, `outputs/`, or the official `scripts/` files. For every new notebook, replace the official Colab URL with:

```text
https://colab.research.google.com/github/hassanraza04/flyrank_intern_content/blob/main/<notebook-path>
```

- [ ] **Step 3: Verify the imported notebooks and skills**

Run:

```bash
jq -e '.nbformat == 4' work/notebooks/w04_baseline_score.ipynb
jq -e '.nbformat == 4' work/notebooks/w05_model.ipynb
jq -e '.nbformat == 4' work/notebooks/w06_validation_audit.ipynb
jq -e '.nbformat == 4' work/notebooks/w07_action_playbook.ipynb
jq -e '.nbformat == 4' work/notebooks/capstone.ipynb
test -f skills/flyrank/flyrank-data/SKILL.md
```

Expected: every command exits with status 0.

- [ ] **Step 4: Commit the imported scaffolding**

```bash
git add skills work/notebooks submission/README.md
git commit -m "Add current capstone guidance and notebook skeletons"
```

## Task 2: Build and Test Pure Capstone Helpers

**Files:**
- Create: `work/scripts/__init__.py`
- Create: `work/scripts/capstone_utils.py`
- Create: `work/tests/__init__.py`
- Create: `work/tests/test_capstone_utils.py`

**Consumes:** A pandas frame matching the cohort interface above.

**Produces:** Tested, reusable feature validation, transparent baseline scores, ranking metrics, reason codes, and public-safe aggregate summaries.

- [ ] **Step 1: Write failing tests**

```python
import unittest
import pandas as pd

from work.scripts.capstone_utils import (
    add_baseline_score, precision_at_k, validate_feature_columns,
)


class CapstoneUtilsTest(unittest.TestCase):
    def test_precision_at_k_uses_top_scores(self):
        labels = pd.Series([0, 1, 1, 0])
        scores = pd.Series([0.1, 0.9, 0.8, 0.2])
        self.assertEqual(precision_at_k(labels, scores, 2), 1.0)

    def test_feature_validator_rejects_future_column(self):
        with self.assertRaises(ValueError):
            validate_feature_columns(["current_ctr", "next_impressions"])

    def test_baseline_prioritises_visible_recent_declines(self):
        frame = pd.DataFrame({
            "current_impressions": [1000.0, 1000.0],
            "impression_change_pct": [-0.50, 0.10],
            "position_change": [1.0, -1.0],
            "current_ctr": [1.0, 1.0],
        })
        scored = add_baseline_score(frame)
        self.assertGreater(scored.loc[0, "baseline_score"], scored.loc[1, "baseline_score"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to confirm they fail**

Run:

```bash
python -m unittest work.tests.test_capstone_utils -v
```

Expected: import failure because `capstone_utils.py` does not exist.

- [ ] **Step 3: Implement the helpers**

```python
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FEATURE_COLUMNS = [
    "log_current_impressions", "impression_change_pct", "click_change_pct",
    "current_ctr", "current_avg_position", "position_change",
    "current_active_days",
]
FORBIDDEN_COLUMNS = {
    "client_hash_id", "content_hash_id", "cohort_id", "is_declining_proxy",
    "next_impressions", "future_impression_ratio",
}


def validate_feature_columns(columns: list[str]) -> None:
    blocked = sorted(set(columns) & FORBIDDEN_COLUMNS)
    if blocked:
        raise ValueError(f"Forbidden feature columns: {', '.join(blocked)}")


def add_baseline_score(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    decline = (-scored["impression_change_pct"]).clip(lower=0)
    demand = np.log1p(scored["current_impressions"]).clip(lower=0)
    position_worsening = scored["position_change"].clip(lower=0)
    scored["baseline_score"] = decline * demand + 0.10 * position_worsening
    return scored


def precision_at_k(labels: pd.Series, scores: pd.Series, k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    order = np.argsort(-np.asarray(scores))[: min(k, len(scores))]
    return float(np.asarray(labels)[order].mean())


def evaluate_ranking(labels: pd.Series, scores: pd.Series, k: int = 100) -> dict[str, float]:
    return {
        "base_rate": float(labels.mean()),
        "precision_at_k": precision_at_k(labels, scores, k),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def reason_codes(row: pd.Series) -> list[str]:
    codes: list[str] = []
    if row["impression_change_pct"] <= -0.20:
        codes.append("recent_search_momentum_down")
    if row["position_change"] >= 1.0:
        codes.append("average_position_worsened")
    if row["current_impressions"] >= 500:
        codes.append("meaningful_search_visibility")
    if row["current_ctr"] < 1.0:
        codes.append("low_ctr_review_candidate")
    return codes or ["monitor_for_more_evidence"]


def public_summary(frame: pd.DataFrame) -> dict[str, Any]:
    actions = frame["action"].value_counts().to_dict()
    reasons = frame.explode("reason_codes")["reason_codes"].value_counts().to_dict()
    return {
        "recommendations": int(len(frame)),
        "action_counts": {str(key): int(value) for key, value in actions.items()},
        "reason_code_counts": {str(key): int(value) for key, value in reasons.items()},
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

Run:

```bash
python -m unittest work.tests.test_capstone_utils -v
```

Expected: three passing tests.

- [ ] **Step 5: Commit the helpers and tests**

```bash
git add work/scripts work/tests
git commit -m "Add tested capstone scoring helpers"
```

## Task 3: Create the Warehouse Feature Frame and Baseline Notebook

**Files:**
- Modify: `work/notebooks/w04_baseline_score.ipynb`
- Create: `work/outputs/.gitkeep`

**Consumes:** `HF_TOKEN` from Colab Secrets, DuckDB, and monthly partitions from September 2025 through June 2026.

**Produces:** A cached local `work/outputs/capstone_features.parquet` during notebook execution, plus an executed baseline notebook. The parquet file remains untracked.

- [ ] **Step 1: Add a safe connection cell and explicit cohort calendar**

```python
%pip -q install duckdb pandas pyarrow scikit-learn matplotlib

import os
from pathlib import Path

import duckdb
import pandas as pd

try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except ImportError:
    HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN in Colab Secrets or the local environment.")

con = duckdb.connect()
safe_token = HF_TOKEN.replace("'", "''")
con.execute(f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{safe_token}')")

COHORTS = [
    ("2025-12", "2025-09-08", "2025-10-05", "2025-10-06", "2025-11-02", "2025-11-03", "2025-11-30"),
    ("2026-01", "2025-10-06", "2025-11-02", "2025-11-03", "2025-11-30", "2025-12-01", "2025-12-28"),
    ("2026-02", "2025-11-03", "2025-11-30", "2025-12-01", "2025-12-28", "2025-12-29", "2026-01-25"),
    ("2026-03", "2025-12-01", "2025-12-28", "2025-12-29", "2026-01-25", "2026-01-26", "2026-02-22"),
    ("2026-04", "2025-12-29", "2026-01-25", "2026-01-26", "2026-02-22", "2026-02-23", "2026-03-22"),
    ("2026-05", "2026-01-26", "2026-02-22", "2026-02-23", "2026-03-22", "2026-03-23", "2026-04-19"),
    ("2026-06-sealed", "2026-03-23", "2026-04-19", "2026-04-20", "2026-05-17", "2026-05-18", "2026-06-14"),
]
```

- [ ] **Step 2: Build the aggregate-only feature SQL**

```python
REL = "hf://datasets/FlyRank/internship-warehouse"
MONTHS = [
    "2025-09", "2025-10", "2025-11", "2025-12", "2026-01",
    "2026-02", "2026-03", "2026-04", "2026-05", "2026-06",
]
DAILY = "read_parquet([" + ", ".join(
    f"'{REL}/fact_content_daily_performance/month={month}/*.parquet'" for month in MONTHS
) + "])"
VALUES = ", ".join(
    "(" + ", ".join(repr(value) for value in row) + ")" for row in COHORTS
)

feature_sql = f"""
WITH cohorts(cohort_id, previous_start, previous_end, current_start, current_end, outcome_start, outcome_end) AS (
    VALUES {VALUES}
), eligible_clients AS (
    SELECT client_hash_id
    FROM read_parquet('{REL}/dim_clients.parquet')
    WHERE gsc_data_start <= DATE '2025-09-08'
), windows AS (
    SELECT
        c.cohort_id, f.client_hash_id, f.content_hash_id,
        SUM(CASE WHEN f.report_date BETWEEN CAST(c.previous_start AS DATE) AND CAST(c.previous_end AS DATE) THEN f.gsc_impressions ELSE 0 END) AS previous_impressions,
        SUM(CASE WHEN f.report_date BETWEEN CAST(c.current_start AS DATE) AND CAST(c.current_end AS DATE) THEN f.gsc_impressions ELSE 0 END) AS current_impressions,
        SUM(CASE WHEN f.report_date BETWEEN CAST(c.outcome_start AS DATE) AND CAST(c.outcome_end AS DATE) THEN f.gsc_impressions ELSE 0 END) AS next_impressions,
        SUM(CASE WHEN f.report_date BETWEEN CAST(c.previous_start AS DATE) AND CAST(c.previous_end AS DATE) THEN f.gsc_clicks ELSE 0 END) AS previous_clicks,
        SUM(CASE WHEN f.report_date BETWEEN CAST(c.current_start AS DATE) AND CAST(c.current_end AS DATE) THEN f.gsc_clicks ELSE 0 END) AS current_clicks,
        AVG(CASE WHEN f.report_date BETWEEN CAST(c.previous_start AS DATE) AND CAST(c.previous_end AS DATE) THEN f.gsc_avg_position END) AS previous_avg_position,
        AVG(CASE WHEN f.report_date BETWEEN CAST(c.current_start AS DATE) AND CAST(c.current_end AS DATE) THEN f.gsc_avg_position END) AS current_avg_position,
        COUNT(*) FILTER (WHERE f.report_date BETWEEN CAST(c.current_start AS DATE) AND CAST(c.current_end AS DATE) AND f.gsc_impressions > 0) AS current_active_days
    FROM {DAILY} AS f
    JOIN cohorts AS c ON f.report_date BETWEEN CAST(c.previous_start AS DATE) AND CAST(c.outcome_end AS DATE)
    JOIN eligible_clients AS e USING (client_hash_id)
    GROUP BY 1, 2, 3
)
SELECT
    *,
    LN(1 + current_impressions) AS log_current_impressions,
    1.0 * (current_impressions - previous_impressions) / NULLIF(previous_impressions, 0) AS impression_change_pct,
    1.0 * (current_clicks - previous_clicks) / NULLIF(previous_clicks, 0) AS click_change_pct,
    100.0 * current_clicks / NULLIF(current_impressions, 0) AS current_ctr,
    current_avg_position - previous_avg_position AS position_change,
    CASE WHEN next_impressions < 0.80 * current_impressions THEN 1 ELSE 0 END AS is_declining_proxy
FROM windows
WHERE current_impressions >= 100
"""
features = con.sql(feature_sql).df()
Path("work/outputs").mkdir(parents=True, exist_ok=True)
features.to_parquet("work/outputs/capstone_features.parquet", index=False)
```

- [ ] **Step 3: Add development-only grain and timeline checks**

```python
assert features["cohort_id"].eq("2026-06-sealed").any()
assert not {"next_impressions", "is_declining_proxy"}.intersection(FEATURE_COLUMNS)
assert (features["current_impressions"] >= 100).all()
features.groupby("cohort_id")["is_declining_proxy"].agg(["count", "mean"])
```

Expected: every cohort has a visible row count and base rate. Do not inspect the sealed cohort’s model metrics until Task 5.

- [ ] **Step 4: Implement and evaluate the frozen baseline on development cohorts**

```python
from work.scripts.capstone_utils import add_baseline_score, evaluate_ranking, reason_codes

development = features.query("cohort_id != '2026-06-sealed'").copy()
development = add_baseline_score(development)
validation = development.query("cohort_id == '2026-05'").copy()
baseline_validation = evaluate_ranking(
    validation["is_declining_proxy"], validation["baseline_score"], k=100
)
validation["reason_codes"] = validation.apply(reason_codes, axis=1)
print(baseline_validation)
```

Expected: a printed base rate, ROC-AUC, and Precision@100. The baseline definition is frozen after this point.

- [ ] **Step 5: Run the notebook from a fresh Colab kernel and commit only code**

Run: Colab **Runtime → Disconnect and delete runtime**, then **Run all**.

Expected: no raw daily dataframe is sent to pandas, and no `.parquet` file appears in `git status`.

```bash
git add work/notebooks/w04_baseline_score.ipynb work/outputs/.gitkeep
git commit -m "Add time-aware refresh baseline notebook"
```

## Task 4: Train the Honest Model and Record the Sealed Test Receipt

**Files:**
- Modify: `work/notebooks/w05_model.ipynb`
- Create: `work/outputs/capstone_metrics.json`
- Create: `work/figures/model_comparison.svg`
- Create: `work/figures/feature_importance.svg`

**Consumes:** `work/outputs/capstone_features.parquet` generated by Task 3 and the pure helpers from Task 2.

**Produces:** Frozen model comparison metrics, aggregate feature importance, and an executed modelling notebook. `capstone_metrics.json` contains no raw IDs.

- [ ] **Step 1: Write the model split and feature guard cell**

```python
from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from work.scripts.capstone_utils import FEATURE_COLUMNS, add_baseline_score, evaluate_ranking, validate_feature_columns

features = pd.read_parquet("work/outputs/capstone_features.parquet")
validate_feature_columns(FEATURE_COLUMNS)

train = features[features["cohort_id"].isin(["2025-12", "2026-01", "2026-02", "2026-03", "2026-04"])].copy()
validation = features[features["cohort_id"] == "2026-05"].copy()
sealed = features[features["cohort_id"] == "2026-06-sealed"].copy()
assert sealed["next_impressions"].notna().all()
```

- [ ] **Step 2: Train two bounded models and compare on the development validation cohort**

```python
models = {
    "logistic_regression": make_pipeline(
        SimpleImputer(strategy="median"),
        LogisticRegression(max_iter=1000, C=0.5, random_state=42),
    ),
    "hist_gradient_boosting": make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=150, random_state=42),
    ),
}

validation_rows = []
for name, model in models.items():
    model.fit(train[FEATURE_COLUMNS], train["is_declining_proxy"])
    probability = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    metrics = evaluate_ranking(validation["is_declining_proxy"], probability, k=100)
    validation_rows.append({"method": name, **metrics})

baseline_validation = add_baseline_score(validation)
validation_rows.append({
    "method": "momentum_baseline",
    **evaluate_ranking(baseline_validation["is_declining_proxy"], baseline_validation["baseline_score"], k=100),
})
comparison = pd.DataFrame(validation_rows).sort_values("precision_at_k", ascending=False)
comparison
```

- [ ] **Step 3: Select one model before opening the sealed test**

```python
selected_name = comparison.iloc[0]["method"]
if selected_name == "momentum_baseline":
    selected_model = None
else:
    selected_model = models[selected_name].fit(
        pd.concat([train, validation])[FEATURE_COLUMNS],
        pd.concat([train, validation])["is_declining_proxy"],
    )
print(f"Frozen choice before sealed test: {selected_name}")
```

Expected: the model choice is printed once and then kept unchanged.

- [ ] **Step 4: Evaluate the chosen method and baseline on the sealed June outcome cohort**

```python
sealed_baseline = add_baseline_score(sealed)
baseline_metrics = evaluate_ranking(
    sealed_baseline["is_declining_proxy"], sealed_baseline["baseline_score"], k=100
)

if selected_model is None:
    selected_scores = sealed_baseline["baseline_score"]
else:
    selected_scores = selected_model.predict_proba(sealed[FEATURE_COLUMNS])[:, 1]

model_metrics = evaluate_ranking(sealed["is_declining_proxy"], selected_scores, k=100)
metrics_receipt = {
    "cohort": "2026-06-sealed",
    "target": "next_28d_impressions < 0.80 * current_28d_impressions",
    "metric_k": 100,
    "feature_columns": FEATURE_COLUMNS,
    "selected_method": selected_name,
    "baseline": baseline_metrics,
    "selected_method_metrics": model_metrics,
    "random_seed": 42,
}
Path("work/outputs").mkdir(parents=True, exist_ok=True)
Path("work/outputs/capstone_metrics.json").write_text(json.dumps(metrics_receipt, indent=2))
metrics_receipt
```

- [ ] **Step 5: Add error analysis and aggregate importance charts**

For logistic regression, plot absolute standardized coefficients. For histogram gradient boosting, calculate permutation importance with `sklearn.inspection.permutation_importance` on the validation cohort. Make two SVGs: a same-split model comparison bar chart and a top-feature importance chart. Use title text that identifies the time-aware validation or sealed test and labels every axis.

- [ ] **Step 6: Verify receipt safety and commit**

Run:

```bash
jq -e '.cohort == "2026-06-sealed" and (.feature_columns | index("next_impressions") | not)' work/outputs/capstone_metrics.json
rg -n 'client_[A-Za-z0-9]|content_[A-Za-z0-9]|hf_[A-Za-z0-9]' work/outputs work/figures || true
git add work/notebooks/w05_model.ipynb work/outputs/capstone_metrics.json work/figures
git commit -m "Add validated refresh opportunity model"
```

Expected: `jq` exits 0 and the `rg` check emits no matches.

## Task 5: Validate Leakage and Produce the Action Playbook

**Files:**
- Modify: `work/notebooks/w06_validation_audit.ipynb`
- Modify: `work/notebooks/w07_action_playbook.ipynb`
- Create: `work/outputs/capstone_summary.json`
- Create: `work/figures/recommendation_actions.svg`
- Create: `work/figures/recommendation_reasons.svg`

**Consumes:** Cached feature frame, selected model receipt, and `capstone_utils` helpers.

**Produces:** A documented leakage audit, public-safe recommendation aggregates, and action/reason charts.

- [ ] **Step 1: Add an explicit future-column attack in the validation notebook**

```python
honest_columns = FEATURE_COLUMNS
leaky_columns = FEATURE_COLUMNS + ["next_impressions"]
validate_feature_columns(honest_columns)

try:
    validate_feature_columns(leaky_columns)
except ValueError as error:
    print(f"Leakage guard worked: {error}")
else:
    raise AssertionError("Future outcome was accepted as a feature")
```

Then fit a development-only model using `future_impression_ratio = next_impressions / current_impressions` once, print its near-perfect score beside the honest score, delete the column, and assert it is absent before any sealed test code executes.

- [ ] **Step 2: Add the validation checklist and a population-selection limitation**

Write markdown that confirms the timeline, forbidden-column guard, time-aware split, base rate, frozen baseline, and final June period. State that requiring 100 current-window impressions selects pages with established visibility and may not generalize to new or very low-traffic pages.

- [ ] **Step 3: Build the public-safe action frame in the playbook notebook**

```python
from work.scripts.capstone_utils import public_summary, reason_codes

queue = sealed.copy()
queue["model_score"] = selected_scores
queue["reason_codes"] = queue.apply(reason_codes, axis=1)
queue["action"] = "monitor"
queue.loc[(queue["model_score"] >= queue["model_score"].quantile(0.90)) & (queue["current_impressions"] >= 500), "action"] = "refresh_review"
queue.loc[(queue["action"] == "refresh_review") & (queue["position_change"] >= 1.0), "action"] = "refresh_and_search_intent_review"
queue.loc[(queue["action"] == "refresh_review") & (queue["current_ctr"] < 1.0), "action"] = "refresh_and_snippet_review"

summary = public_summary(queue)
summary["action_definitions"] = {
    "refresh_review": "Inspect content quality, freshness, and search-demand alignment.",
    "refresh_and_search_intent_review": "Review content coverage and search-intent alignment before changing the page.",
    "refresh_and_snippet_review": "Review title and snippet presentation alongside the content.",
    "monitor": "Collect more evidence before recommending an edit.",
}
Path("work/outputs/capstone_summary.json").write_text(json.dumps(summary, indent=2))
```

- [ ] **Step 4: Create two aggregate charts and verify no raw rows are exported**

Make `recommendation_actions.svg` from `summary["action_counts"]` and `recommendation_reasons.svg` from the ten largest `reason_code_counts`. Do not write `queue` to CSV, JSON, HTML, figures, or git.

- [ ] **Step 5: Commit the audit, action playbook, and safe receipts**

```bash
git add work/notebooks/w06_validation_audit.ipynb work/notebooks/w07_action_playbook.ipynb work/outputs/capstone_summary.json work/figures
git commit -m "Add capstone validation audit and action playbook"
```

## Task 6: Write the Capstone Notebook and Static Research Paper

**Files:**
- Modify: `work/notebooks/capstone.ipynb`
- Create: `work/capstone_report.md`
- Create: `work/scripts/render_paper.py`
- Create: `docs/assets/paper.css`
- Create: `docs/index.html`

**Consumes:** `capstone_metrics.json`, `capstone_summary.json`, and the four aggregate SVG files.

**Produces:** A notebook and a static paper with matching public-safe numbers and all required sections.

- [ ] **Step 1: Fill the capstone notebook in the required order**

Use these exact section headings:

```text
Question
Data
Methodology
Results (vs baseline)
Limitations
Ranked recommendations
Artifacts the paper embeds
Self-check
ML-12: 5-minute demo outline, social cut, and employer summary
```

All numbers must come from the committed JSON receipts. The notebook may display aggregate tables and charts, but not the raw ranked queue.

- [ ] **Step 2: Create a single-purpose paper renderer**

Implement `render_paper.py` with this entry point:

```python
def render_paper(metrics_path: Path, summary_path: Path, figures_dir: Path, output_path: Path) -> None:
    """Render a public-safe static capstone paper from aggregate receipts."""
```

The implementation reads the two JSON files, copies SVG figures to `docs/assets/`, and writes an HTML document containing these headings in order:

```text
Title and Abstract
Introduction: the decision this supports
Data
Methodology
Results
Limitations and honest framing
Ranked recommendations
Reproducibility
Acknowledgments and data credit
```

The acknowledgement must include:

```html
Built on the <a href="https://flyrank.ai">FlyRank ML Internship dataset</a>.
```

- [ ] **Step 3: Write the public-safe paper copy after metrics exist**

Use only these claim forms:

```text
We observed … in this release and period.
The selected model ranked … at Precision@100 of … on the sealed time-aware cohort.
These recommendations are decision support for editorial review, not automated publishing decisions.
This analysis does not establish that refreshing a page causes an improvement in search visibility.
```

Write the abstract last using exactly five sentences: question, data, method, result, intended use and limitation.

- [ ] **Step 4: Render and locally verify the page**

Run:

```bash
python work/scripts/render_paper.py
rg -n 'client_[A-Za-z0-9]|content_[A-Za-z0-9]|hf_[A-Za-z0-9]|https?://[^" ]*(?:example-client|query)' docs work/capstone_report.md
test -f docs/index.html
```

Expected: the safety scan emits no matches and `docs/index.html` exists.

- [ ] **Step 5: Commit the paper source and rendered site**

```bash
git add work/notebooks/capstone.ipynb work/capstone_report.md work/scripts/render_paper.py docs/index.html docs/assets
git commit -m "Add refresh opportunity research paper"
```

## Task 7: Publish and Verify GitHub Pages

**Files:**
- Create: `submission/paper_url.txt`

**Consumes:** `docs/index.html` on the `main` branch.

**Produces:** A live public paper URL and the mandatory one-line submission file.

- [ ] **Step 1: Push the final paper commit**

```bash
git push origin main
```

- [ ] **Step 2: Enable Pages in GitHub**

In GitHub: repository **Settings → Pages → Build and deployment → Deploy from a branch → `main` → `/docs` → Save**.

Expected: GitHub displays the final site URL after deployment.

- [ ] **Step 3: Verify the public paper like a new visitor**

Open the URL in an incognito browser window and on a phone. Confirm every chart loads, the FlyRank credit link works, all nine headings appear, and no private data is visible.

- [ ] **Step 4: Record exactly one final URL and verify it**

```bash
printf '%s\n' 'https://hassanraza04.github.io/flyrank_intern_content/' > submission/paper_url.txt
wc -l submission/paper_url.txt
git add submission/paper_url.txt
git commit -m "Record deployed capstone paper URL"
git push origin main
```

Expected: `wc -l` reports `1`. Replace the example URL only if GitHub Pages reports a different exact address.

## Plan Self-Review

- Spec coverage: Tasks 1 through 7 cover the current official guidance, full warehouse feature frame, baseline, model, time-aware sealed test, leakage attack, aggregate recommendations, nine-section paper, data credit, and Pages submission URL.
- Placeholder scan: complete. Every helper, file, command, and safety check is named. Runtime metrics are intentionally generated only by the executed warehouse notebook.
- Interface consistency: Task 2 defines the feature helpers used by Tasks 3 through 5. Task 3 writes the feature frame consumed by Task 4. Task 4 writes metrics used by Tasks 5 and 6. Task 5 writes the summary used by Task 6.
