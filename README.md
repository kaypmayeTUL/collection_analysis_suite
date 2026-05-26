# Library Collection Dashboard (slim edition)

A unified Streamlit application bundling three collection decision-support tools for Howard-Tilton Memorial Library, Tulane University. Built for the rhythms of a research library's fiscal year — cancellation review, big-ticket prep, weeding, renewal, and spend-down work.

## The three tools

### 🗺️ Collection Profiler

**What does our collection look like, and what's used?**

Coverage-led analysis across three views you move between in tabs:

- **LC Analysis** — sunburst, treemap, LC × subject heatmap, gap analysis, coverage-vs-use, and sub-class range distribution (drilling below the two-letter subclass into specific LC numeric ranges like "HQ 1101–2030.7 Women, feminism, women's studies"). Every coverage-vs-use table and the range view offer a **"show the records behind this"** drill-down (see below).
- **Subject Term Analysis** — top subjects, word cloud, title-keyword n-grams. Powered by controlled-vocabulary subject headings when present. The subject bars and subject coverage-vs-use both support the records drill-down.
- **Title Analysis** — top titles by usage, weeding review, author summary, date-range filtering, and a yearly trends sub-tab that surfaces year-over-year shifts in usage. Activates only when a usage column is present.

Accepts CSV or Excel files. Auto-detects Title, Subjects, LC Classification, usage (Loans/Checkouts/Total Accesses/Views/Downloads), Author, Location, and Date columns. Works on Alma title exports, Alma circulation exports, digital platform views, and vendor admin reports like the EBSCO Detailed Report.

### 📊 COUNTER Analyzer

**Which e-resources are pulling their weight?**

Formal COUNTER 5 reports only — TR / TR_J3 / TR_B1 / DR / PR / IR with the standard 12–13 row metadata header and monthly columns. Top titles, cancellation review, publisher rollups, and monthly trends.

If your file is a vendor *admin* export with subjects and LC (e.g., EBSCO Detailed Report), that belongs in the Profiler instead — the Profiler's three views give you more analytical leverage than the COUNTER Analyzer would.

### 🔍 Zero-Use Identifier

**What do we own that isn't being used?**

Two-file comparison. Upload a holdings file (the "universe") and a usage file (what you have evidence of use for). The tool runs a multi-identifier matching cascade — ISBN, ISSN, DOI, OCLC, then title+author fallback — and surfaces holdings that don't appear in the usage data. Includes a Match Preview tab for spot-checking joins, configurable pub-year cutoff, and an optional "treat unmatched as zero-use" toggle.

## Drilling into the records behind a conclusion

Every analytical view in the Profiler ends at an aggregate — "HQ Women/Feminism is 738 records, 1,337 loans," or "this range is overperforming," or "usage jumped in 2025." The natural next question is always *which actual titles drive that?* The drill-downs answer it without leaving the tool.

Wherever a conclusion is surfaced, a **🔎 "show the records behind this"** control appears. Pick the range, subclass, LC class, or subject you want to inspect, and the underlying titles open in an expander right there — already scoped to exactly what you clicked. Then refine within that scope:

- **Usage filter** — All, zero-usage only, at-or-below a threshold, or at-or-above a threshold. (Zero-usage-only is the fast path to weeding candidates within a flagged range.)
- **Year range** — narrow to specific years when the file carries usage dates.
- **Sort** — by usage (either direction), title, author, call number, or year.
- **Columns** — toggle which fields show in the table.
- **Export** — download the refined record list as CSV (with your analysis notes baked in), automatically added to the page's download tray.

The drill-down appears in five places: range-level coverage-vs-use, subclass-level coverage-vs-use, LC main-class coverage-vs-use, subject coverage-vs-use, and the top-subjects bar chart. In each coverage-vs-use view, **flagged areas (over- and underperforming) are listed first** in the picker, so auditing a signal is one click away.

A worked example: the LC Analysis tab flags HQ 1101–2030.7 as a range worth examining. Open its drill-down, sort by loans descending, and you see *Bodies That Matter*, *Gender Trouble*, and the other titles actually driving the number — the difference between telling a liaison "the data says HQ feminism is strong" and "these specific Butler titles are why."

**One caveat on subject drill-downs:** subject matching is a case-insensitive substring match against the raw subjects column, so it can occasionally over-match (e.g., a search for a short term catching it inside a longer word). Counts in subject drill-downs may run slightly generous; the LC-based drill-downs (range, subclass, class) are exact.

## Run locally

```bash
pip install -r requirements.txt
streamlit run library_dashboard_slim.py
```

## requirements.txt

```
streamlit>=1.28
pandas>=2.0
numpy>=1.24
plotly>=5.15
wordcloud>=1.9      # optional — enables word-cloud view in Subject Term Analysis
matplotlib>=3.7     # optional — required by wordcloud
openpyxl>=3.1       # optional — enables .xlsx upload
xlrd>=2.0           # optional — enables .xls upload
```

The wordcloud, matplotlib, openpyxl, and xlrd packages are soft dependencies — the dashboard runs without them, but specific features (word cloud rendering, Excel file uploads) will be disabled with a friendly message.

## Deploy to Streamlit Community Cloud

1. Push `library_dashboard_slim.py` and `requirements.txt` to a GitHub repo.
2. At [share.streamlit.io](https://share.streamlit.io), create a new app pointing to the repo.
3. Set the main file to `library_dashboard_slim.py`.
4. Deploy. First load takes ~10 seconds.

## What the dashboard accepts

| Tool | File type | Required columns | Optional columns |
|---|---|---|---|
| Collection Profiler | CSV or Excel (.csv, .xls, .xlsx) | Title; Subjects and/or LC Classification | Usage (Loans / Checkouts / Total Accesses / Views / Downloads), Author, Location, Date |
| COUNTER Analyzer | CSV/TSV (formal COUNTER 5) | Standard COUNTER 5 format with 12–13 row metadata header and monthly columns | — |
| Zero-Use Identifier | Two files (holdings + usage) | At least one shared identifier (ISBN, ISSN, DOI, OCLC) OR title in both files | LC Classification, Pub Year, Author |

## What it intentionally does *not* do

- **Acquisition recommendation scoring** has been extracted to its own standalone app (`recommender_app.py`). If you need to score candidate book lists against checkout history, use that app instead.
- **No live database connections** — every tool consumes file exports you control. Reproducible, auditable, no surprise API costs.
- **No AI/LLM scoring** — every metric is a deterministic calculation over your data. The "Coverage vs. Use" signals, range distributions, year-over-year comparisons, and identifier-cascade matches all come from explicit arithmetic on your file contents.

## Shared design conventions

- **Tulane palette**: green `#285C4D`, blue `#71C5E8`. Hardcoded in the `<style>` block near the top of the file.
- **Analysis-notes field**: every tool offers a free-text notes area near its results. Notes travel with CSV downloads as header comments — useful for documenting why a decision was made when you revisit the file months later.
- **Download tray**: each tool bundles every artifact it produced (CSVs, images, tables) into a single ZIP at the bottom of the page. No hunting for individual download buttons.
- **Memory-efficient loading**: large files (500K–1M+ rows) are read with `usecols` filtering to keep only relevant columns in memory.
- **Range catalog**: the Profiler's LC Analysis tab uses a curated catalog of 212 LC subclasses with 592 ranges drawn from the LC Classification Outline. Subclasses without curated ranges fall back to hundreds-bucketing (e.g., "F 1400s").
- **Records drill-downs**: every Profiler conclusion (coverage-vs-use signals, range distributions, subject frequencies) can be expanded into the underlying titles, filtered and sorted in place, and exported. No new upload required — drill-downs run on the file already loaded.

## Related files

- **`library_dashboard.py`** — the full 4-tool dashboard with the Acquisition Recommendation Scorer still integrated. Use this if you want everything in one app.
- **`recommender_app.py`** — standalone Acquisition Recommendation Scorer for scoring vendor slip lists, GOBI picks, approval-plan exceptions, and faculty requests against checkout history.
- **`recommender_app_README.md`** — deployment notes for the standalone recommender.
- **`collection_workflow_plan.docx`** — the fiscal-calendar implementation plan that maps each tool to specific decision points across the year.

## Tulane styling

The dashboard ships with Tulane green (`#285C4D`) and blue (`#71C5E8`) hardcoded, plus Source Serif 4 and DM Sans font references in the CSS. Edit the `<style>` block near the top of `library_dashboard_slim.py` to change colors or fonts.

## Version

**v2.5 (slim)** — Added inline "show the records behind this" drill-downs across the Profiler's coverage-vs-use views, range distribution, and subject bars, with in-place usage/year filtering, sorting, and export.

**v2.4 (slim)** — Acquisition Recommendation Scorer extracted to standalone app. NLTK no longer a runtime dependency. Profiler retains the full range catalog, yearly trends, and range-level Coverage-vs-Use added in v2.3.

## Contact

Kay P Maye (kmaye@tulane.edu) — Howard-Tilton Memorial Library, Tulane University
