#  Data Detective

Automated data quality & root-cause intelligence. Upload any CSV and get a health
score, a list of quality issues, and — the standout feature — an automatic
explanation of **where problems are concentrated and why**, not just that they exist.

Instead of stopping at "8% of values are missing," Data Detective answers the
follow-up question a human analyst would ask next: *is that random, or is it
coming from one specific segment of the data?*

## What it detects

- **Missing values** — per-column count and percentage
- **Duplicates** — exact duplicate rows
- **Incorrect data types** — e.g. a numeric column accidentally stored as text
- **Inconsistent categories** — near-duplicate labels like `"South"` / `"south"` / `"SOUTH "`
- **Outliers & anomalies** — both single-column (IQR) and multi-column (Isolation Forest)
- **Root-cause segmentation** — which categorical values anomalies are disproportionately
  concentrated in, using lift + chi-square significance testing
- **Distribution drift** — whether a numeric column's distribution shifted meaningfully
  between two time periods, using KS-test and PSI (Population Stability Index)
- **Correlations** — relationships between numeric columns
- **Composite health score (0–100)** — a weighted, explainable combination of
  completeness, uniqueness, validity, and consistency sub-scores

## How it works (short version)

1. `profiler.py` runs the basic checks (missing values, duplicates, dtypes, inconsistent categories)
2. `anomaly.py` flags anomalous rows using IQR (per-column) and Isolation Forest (multi-column)
3. `segmentation.py` — the core feature — checks whether those anomalies are disproportionately
   concentrated in specific categorical segments, using a lift calculation validated by a
   chi-square test, and turns the result into a plain-English finding
4. `drift.py` compares distributions across two time periods using KS-test and PSI
5. `health_score.py` combines everything into one explainable score
6. `app.py` (Streamlit) ties it all together into an interactive report

## Setup

```bash
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`), and
either upload your own CSV or click "Load sample dataset" to see it in action
on `sample_data.csv` — a synthetic transactions dataset with deliberately
injected quality issues (missing income data, mislabeled regions, a genuine
anomaly cluster in South region electronics/toy transactions, and an age
distribution shift partway through the year) so you can see every feature fire.

## Running tests

```bash
pip install pytest
pytest test_data_detective.py -v
```

## Prior art — how this compares to existing tools

This project is a simplified version of what commercial "data observability"
platforms (Monte Carlo, Soda) and open-source profilers (ydata-profiling,
Great Expectations) do. Being upfront about that:

- **ydata-profiling** generates similar descriptive stats (missing values,
  correlations, distributions) but stops there — it doesn't do automatic
  root-cause segmentation.
- **Great Expectations** validates data against rules you define yourself; it's
  not exploratory/automatic like this tool.
- **Monte Carlo / Soda** are paid, enterprise-scale platforms this project is
  a lightweight, from-scratch version of — focused specifically on the
  automatic root-cause narrative, which is the piece most existing tools skip.

## Project structure

```
data_detective/
├── app.py                     # Streamlit interface
├── profiler.py                # missing values, duplicates, dtypes, inconsistent categories
├── anomaly.py                 # IQR + Isolation Forest outlier detection
├── segmentation.py            # root-cause / lift-based segmentation engine
├── drift.py                   # KS-test + PSI drift detection
├── health_score.py            # weighted composite health score
├── test_data_detective.py     # pytest suite
├── sample_data.csv            # synthetic demo dataset with injected issues
└── requirements.txt
```

## Possible next steps

- Support Excel/Parquet uploads, not just CSV
- Let users upload two files directly for drift comparison (vs. requiring a
  date column in one file)
- Add a "download as PDF" report export
- Add fuzzy duplicate row detection (not just exact duplicates)
