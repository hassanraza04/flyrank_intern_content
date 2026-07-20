# ML-03 Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and commit an executed Week 2 notebook that frames the Refresh / Content Opportunity Scoring lane as a ranking task supported by a current-window classification proxy.

**Architecture:** The notebook is self-contained and reads the existing anonymized starter CSV. Markdown cells make the task, proxy, metric, action, and limits explicit. Two executable pandas cells load the data, derive the proxy, and display the page-level decision grain with safe candidate signals.

**Tech Stack:** Jupyter Notebook 4, Python 3, pandas, starter CSV, git.

## Global Constraints

- Create only `work/notebooks/w02_ml_task_framing.ipynb` for the assignment deliverable.
- Keep one row equal to one pseudonymized content item; never use `content_id` or `client_id` as model features.
- Treat `trend_direction == "down"` as a current-window starter proxy, not a future outcome.
- Exclude `trend_direction` and `trend_pct` from candidate features because they encode the proxy outcome.
- Use public-safe, decision-support language and make no causal or Google-ranking-factor claim.
- Commit with `hassanraza04 <136234799+hassanraza04@users.noreply.github.com>`.

---

### Task 1: Create and execute the ML-03 notebook

**Files:**
- Create: `work/notebooks/w02_ml_task_framing.ipynb`
- Read: `data/raw/content_refresh_anonymized.csv`
- Read: `docs/ml-intern-dataset-and-lane-guide.md`
- Read: `docs/data-dictionary.md`

**Interfaces:**
- Consumes: `data/raw/content_refresh_anonymized.csv`, with one row per content item.
- Produces: an executed notebook with six assignment sections and a derived `is_declining_proxy` column.

- [ ] **Step 1: Write the failing structural validation**

Run this before creating the notebook. It must fail because the notebook does not yet exist.

```bash
python - <<'PY'
from pathlib import Path

notebook = Path("work/notebooks/w02_ml_task_framing.ipynb")
assert notebook.exists(), "Missing ML-03 notebook"
PY
```

Expected: `AssertionError: Missing ML-03 notebook`.

- [ ] **Step 2: Create the notebook with the required framing**

Create Markdown sections with these exact headings:

```text
## 1. My lane as an ML task (type)
## 2. Target or proxy
## 3. Success metric
## 4. The unit of analysis, as a real dataframe
## 5. Why ML beats a fixed rule here
## 6. Self-check
```

State that the lane is a **ranking task using a classification proxy**. State that the proxy is `is_declining_proxy = (trend_direction == "down")`, that it is only a teaching proxy, and that a future capstone must use a time-separated target. Define Precision@20 as the primary metric and connect it to a 20-page review capacity.

Use this load cell so it works from either the repository root or `work/notebooks/`:

```python
from pathlib import Path
import pandas as pd

candidate_paths = [root / "data/raw/content_refresh_anonymized.csv" for root in [Path.cwd(), *Path.cwd().parents]]
data_path = next((path for path in candidate_paths if path.exists()), None)
if data_path is None:
    raise FileNotFoundError("Could not find data/raw/content_refresh_anonymized.csv.")

df = pd.read_csv(data_path)
df["is_declining_proxy"] = df["trend_direction"].eq("down")
print(f"Loaded {len(df):,} rows. One row represents one anonymized content item.")
```

Use this unit-of-analysis cell:

```python
unit_columns = [
    "content_id", "client_id", "impressions_90d", "days_since_last_update",
    "avg_position", "ctr", "engagement_rate", "is_declining_proxy"
]
unit_of_analysis = df.loc[:, unit_columns].head(8).copy()
display(unit_of_analysis)
print(f"Rows: {len(df):,}; unique content IDs: {df['content_id'].nunique():,}; unique clients: {df['client_id'].nunique():,}.")
print(f"Current decline proxy rate: {df['is_declining_proxy'].mean():.1%}.")
```

Explain that the displayed identifiers are for grouping and traceability only. Explain that `trend_direction` and `trend_pct` are intentionally absent from the candidate feature list because they leak the current proxy.

- [ ] **Step 3: Execute the notebook**

Run:

```bash
jupyter nbconvert --to notebook --execute --inplace work/notebooks/w02_ml_task_framing.ipynb --ExecutePreprocessor.timeout=120
```

Expected: exit code 0 and output stored in the notebook for the data-load and unit-of-analysis cells.

### Task 2: Verify and publish the deliverable

**Files:**
- Verify: `work/notebooks/w02_ml_task_framing.ipynb`
- Commit: `work/notebooks/w02_ml_task_framing.ipynb`

**Interfaces:**
- Consumes: executed notebook from Task 1.
- Produces: a clean committed notebook on `main`.

- [ ] **Step 1: Verify the executed notebook structurally**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("work/notebooks/w02_ml_task_framing.ipynb")
notebook = json.loads(path.read_text())
markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
for heading in [
    "## 1. My lane as an ML task (type)",
    "## 2. Target or proxy",
    "## 3. Success metric",
    "## 4. The unit of analysis, as a real dataframe",
    "## 5. Why ML beats a fixed rule here",
    "## 6. Self-check",
]:
    assert heading in markdown, heading
code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
assert len(code_cells) >= 2
assert all(cell["execution_count"] is not None for cell in code_cells)
assert all(not any(output.get("output_type") == "error" for output in cell["outputs"]) for cell in code_cells)
outputs = "\n".join("".join(output.get("text", [])) for cell in code_cells for output in cell["outputs"])
assert "One row represents one anonymized content item" in outputs
assert "Current decline proxy rate: 54.2%" in outputs
print("ML-03 notebook structure and execution output verified.")
PY
```

Expected: `ML-03 notebook structure and execution output verified.`

- [ ] **Step 2: Verify the change and commit it under the user identity**

Run:

```bash
git diff --check
git add work/notebooks/w02_ml_task_framing.ipynb
git -c user.name=hassanraza04 -c user.email=136234799+hassanraza04@users.noreply.github.com commit -m "Add Week 2 ML task framing notebook"
git push origin main
```

Expected: no whitespace errors, one new notebook in the commit, and `main -> main` from the push.

## Self-review

- Spec coverage: Task 1 covers the lane type, proxy, metric, unit of analysis, fixed-rule comparison, action, and self-check. Task 2 verifies execution and publishes the required notebook.
- Placeholder scan: this plan contains no unfinished implementation markers.
- Type consistency: the notebook creates `is_declining_proxy` from `trend_direction`; the validation and prose use the same name.
