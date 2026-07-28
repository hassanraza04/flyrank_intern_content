# Refresh Opportunity Scoring Capstone Design

## Goal

Build a public-safe research paper and repeatable analysis for the Refresh / Content Opportunity Scoring lane. The work will rank content items for review based on their measured risk of a near-term fall in search impressions. It supports an editor's decision about which pages to inspect first. It does not make publishing decisions automatically or claim to predict Google's algorithm.

## Scope

The capstone has four connected outputs:

1. A DuckDB feature-building query over the gated FlyRank warehouse.
2. A transparent baseline and a learned model evaluated on the same time-aware split.
3. A ranked, public-safe action playbook with aggregate reason-code summaries only.
4. A static GitHub Pages research paper, with its direct URL recorded in `submission/paper_url.txt` after deployment.

The repo will also receive the current official notebook skeletons and task-specific skills that are missing from this older copy.

## Data contract

The unit of analysis is one client-content item at a monthly decision date. Client and content hash IDs are used only to join and group warehouse records. They are never model features and are not shown in the public paper.

Each training row uses:

- a previous 28-day window;
- a current 28-day feature window;
- a later 28-day outcome window.

The candidate set requires at least 100 impressions in the current feature window. The binary proxy `is_declining_proxy` is 1 when later-window impressions are less than 80% of current-window impressions. The definition is directional and operational, not a causal statement about content quality.

The March 2026 work already completed in ML-04 remains the development data-contract check. The final capstone feature builder uses the full `fact_content_daily_performance` table through DuckDB over `hf://`. It does not use `fact_content_daily_performance_sample` for development. June 2026 is reserved as the sealed final outcome period.

## Features and exclusions

The initial feature set is limited to signals knowable at the decision date:

- current and previous-window impressions;
- current and previous-window clicks;
- current CTR;
- current average position and its change from the previous window;
- current active search days;
- percentage change in impressions and clicks between the two past windows.

The work excludes future-window data, `trend_direction`, `trend_pct`, raw query text, client names, domains, URLs, titles, and pseudonymous IDs as model inputs. GA4 engagement features are excluded from the first capstone model because availability is sparse and uneven, as ML-04 measured.

## Validation and comparison

Development cohorts will end before the final test cohort. The model is selected using an earlier time-aware validation period. The sealed test uses a May 2026 decision date with June 2026 as its future outcome period. This prevents outcome information from leaking backwards and tests the workflow on the final available period.

The transparent baseline ranks candidates using observed momentum: a larger fall from the previous 28-day window to the current 28-day window, weighted by current search demand, receives higher priority. It is evaluated against the same test rows as the model.

The learned comparison begins with regularized logistic regression and a bounded tree-based model. The selected model is the simpler one that provides a meaningful lift over the baseline without leakage. Primary reporting uses Precision@100 because the action is a finite review queue. The paper also reports ROC-AUC, the proxy base rate, and the baseline's Precision@100 on the same split.

## Recommendation output

The final queue remains an internal analytical artifact and is not published as raw rows. The public paper shows only aggregate counts and anonymized example reason-code patterns. Its playbook maps measured patterns to human review actions:

- declining with meaningful visibility: refresh review;
- declining with worsening position: investigate search intent, coverage, and technical indexing context;
- declining with low CTR but stable visibility: title and snippet review;
- weak or uncertain signal: monitor before acting.

Every recommendation is decision support. An editor must inspect context before changing content.

## Public paper

The static site will live in `docs/` and use relative asset paths. It will include Title and Abstract, Introduction, Data, Methodology, Results, Limitations, Ranked Recommendations, Reproducibility, and Acknowledgments with the required FlyRank data credit link.

Charts and summary tables will be generated from public-safe aggregates. The paper will link to the relevant notebooks and the repository. Once GitHub Pages is enabled from `main` and `/docs`, the exact deployed URL will be recorded as the sole line in `submission/paper_url.txt`.

## Verification

Before publication, the work must satisfy these checks:

- the feature query runs from a fresh Colab kernel using a secret-based `HF_TOKEN`;
- all model inputs are available before the decision date;
- the baseline and model use the same candidates and sealed test rows;
- leakage checks demonstrate that future data is absent from honest features;
- reported metrics, charts, and paper text agree;
- the public folders contain no token, client name, domain, URL, raw query, raw export, or pseudonymous item ID;
- the GitHub Pages URL loads with all relative assets and is written exactly once to `submission/paper_url.txt`.
