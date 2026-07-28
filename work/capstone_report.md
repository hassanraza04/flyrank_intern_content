# Refresh Opportunity Scoring for Search Content

- **Author:** Hassan Raza
- **Lane:** Refresh / Content Opportunity Scoring
- **Repository:** https://github.com/hassanraza04/flyrank_intern_content
- **Evaluation date:** July 2026

## 1. Problem framing

This work supports a content editor's decision about which established content items deserve a human review first. One row represents one anonymised client-content item at the end of a completed 28-day search-performance window. The output is a ranked review queue, rather than an automatic recommendation to edit or publish. A wrong call spends limited editorial time on an item that does not decline, while a missed call may leave an important declining item unreviewed. ML can help because this decision depends on several signals together, including recent impressions, clicks, visibility, average position, and activity, instead of one universally reliable threshold.

## 2. Data safety

The feature frame comes from the FlyRank internship warehouse table `fact_content_daily_performance`, queried through DuckDB. It covers the partitioned period from September 2025 through June 2026 and uses stable clients with Search Console coverage beginning on or before 8 September 2025. The analysis deliberately excludes client and content identifiers from model inputs, raw domains, URLs, titles, queries, GA4 engagement fields, and future-window values. GA4 fields were excluded because their availability is uneven. The outcome label and all next-window columns are forbidden as model features. No raw warehouse records, identifiers, credentials, or private exports are committed to this repository.

## 3. Baseline

The transparent baseline gives priority to items with declining recent impressions, weighted by current search visibility, with a small additional weight when average position worsens. It is a useful fair comparison because an editor could apply it without fitting a model. On the sealed June cohort, the baseline reached precision@100 of **0.38** and ROC-AUC of **0.504** against a 0.519 decline-proxy rate.

## 4. Model / analysis

I compared logistic regression and shallow histogram gradient boosting against the baseline. The target is a future decline proxy: the following 28-day impression total is below 80% of the current 28-day total. The seven model features are log current impressions, impression change percentage, click change percentage, current CTR, current average position, position change, and number of active days in the current window. Every feature is available at the decision moment because it is derived only from completed previous and current windows.

The selected method was histogram gradient boosting. It can represent combinations such as meaningful visibility plus worsening position plus weak recent momentum without assuming that every signal has one linear effect. It is still a ranking aid, not an explanation of Google's ranking systems.

## 5. Evaluation

The evaluation is time-aware. Cohorts from December 2025 through April 2026 train the candidate models, May 2026 selects the method, and June 2026 remains sealed until the final comparison. The unit count was 162,679 training rows, 50,735 validation rows, and 46,898 sealed rows.

| Cohort | Method | Base rate | Precision@100 | ROC-AUC |
| --- | --- | ---: | ---: | ---: |
| May 2026 validation | Histogram gradient boosting | 0.447 | 0.79 | 0.593 |
| May 2026 validation | Momentum baseline | 0.447 | 0.69 | 0.522 |
| June 2026 sealed | Histogram gradient boosting | 0.519 | 0.65 | 0.560 |
| June 2026 sealed | Momentum baseline | 0.519 | 0.38 | 0.504 |

The final model's precision@100 fell from validation to the sealed cohort. That is expected uncertainty rather than a result to hide. It still exceeded the baseline by 0.27 on the sealed decision metric. The base-rate change also shows why a single percentage should not be treated as a universal promise.

## 6. Interpretation

The model produced a stronger top-of-queue ranking than a rule that uses only falling impressions and worsening position. The result suggests that combining the seven past-window signals helps distinguish pages worth reviewing first. The ROC-AUC values are modest, especially on the sealed cohort, so the model is more useful for prioritising a limited review queue than for making confident row-by-row predictions. The logistic-regression candidate did not improve on the baseline in validation, so it was not selected.

## 7. Recommendation

An editor should start with the model's top 100 items and inspect them in the private working environment. In the sealed test, 65% of that ranked set met the future decline proxy, compared with 38% from the transparent baseline. A reviewer can then use the available signals to choose a human action:

- **Refresh review:** meaningful visibility with negative recent impression momentum.
- **Search-intent review:** worsening average position, after checking query and page context privately.
- **Snippet review:** low current CTR only after confirming the item is visible and relevant.
- **Monitor:** weak or conflicting signals, rather than spending immediate editorial effort.

These are measured, directional decision-support recommendations. They do not show that a refresh causes visibility to improve, and they must not be used for automatic publishing.

The reproducible handoff is `action_queue` in the capstone notebook. It ranks the top 100 model scores and exposes only priority rank, opportunity score, recommended action, and reason codes. It deliberately omits client and content identifiers from the displayed recommendation frame.

## 8. Reproducibility

Open `work/notebooks/capstone.ipynb` in a fresh Colab runtime, request access to `FlyRank/internship-warehouse`, and add a read token as the `HF_TOKEN` Colab secret. The notebook installs its dependencies, reads the gated warehouse through DuckDB, creates a local ignored parquet cache, fits all candidates with random seed 42, selects using May, and evaluates June once as the sealed cohort. Supporting feature and scoring code is in `work/scripts/capstone_data.py` and `work/scripts/capstone_utils.py`; unit checks run with `python -m unittest work.tests.test_capstone_utils -v`.

Built on the [FlyRank ML Internship dataset](https://flyrank.ai).
