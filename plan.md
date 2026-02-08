# Quant Club Project Plan — Transient Correlation Detection

**Project Manager:** Jack
**Team Size:** 4 (Jack + 3 members)
**Deadline:** End of February 2026
**Timeline:** Week 1 (Feb 9–14) + Reading Week (Feb 16–21) + Buffer (Feb 22–28)

---

## Project Status: What's Done

The OPTICS clustering pipeline is fully functional end-to-end. Specifically, the following are complete and working:

- **Data pipeline** — Yahoo Finance fetch for S&P 500 + semiconductor tickers (NVDA, AMD, TSM, INTC, AVGO, QCOM, ASML, MU, ARM, etc.)
- **Feature engineering** — 15+ features across momentum (RSI, MACD, ROC), volatility (historical vol, ATR), correlation (market beta, pairwise), trend (moving averages), and technical indicators (Bollinger, Stochastic, Williams %R)
- **OPTICS clustering** — Full pipeline with formation/dissolution event detection, sensitivity analysis, and random baseline correctness checks
- **Pair validation module** (`validation/pair_validation.py`) — Cluster persistence, spread computation, AR(1) mean reversion, bounce rate metrics
- **Prediction notebook** (`optics-prediction.ipynb`) — Factor attribution, Random Forest + Gradient Boosting classifiers with time-series cross-validation, ROC/precision-recall evaluation
- **Technical documentation** — `explanation.md` (26 KB, cell-by-cell breakdown) and `CLAUDE.md` (architecture guide)

## What Still Needs to Be Done

1. **K-Means clustering implementation** — New notebook adapting the pipeline to use K-Means instead of OPTICS
2. **DBScan clustering implementation** — New notebook adapting the pipeline to use DBScan instead of OPTICS
3. **Clustering method comparison** — Side-by-side analysis of OPTICS vs K-Means vs DBScan (metrics, strengths, weaknesses on this dataset)
4. **Trading module** (`trading/trading.py`) — Currently empty; needs at least a basic backtesting framework or signal generation logic
5. **PowerPoint presentation** — Slide deck for the Quant Club presentation
6. **LaTeX/Overleaf document** — Formal academic-style writeup of methodology and results
7. **README.md** — Currently empty; needs a proper project overview
8. **Code cleanup** — Ensure all notebooks run cleanly, consistent formatting, proper comments

---

## Task Assignments

### Member A — K-Means Implementation

**Focus:** Implement the full clustering pipeline using K-Means and write up findings.

**Week 1 (Feb 9–14):**
- Create `research/kmeans-clustering.ipynb` by adapting `optics-clustering-run.ipynb`
- Key changes from OPTICS: K-Means requires specifying `k` upfront, so implement an elbow method or silhouette analysis to choose optimal `k` for each time window
- Reuse the existing feature engineering code (cells 2–5 of the OPTICS notebook) — no need to rewrite
- Run clustering across the full time series and generate `cluster_history` equivalent
- Implement the same formation/dissolution detection logic
- Note differences: K-Means assigns every point to a cluster (no noise label like OPTICS's `-1`), so document how this affects formation detection

**Reading Week (Feb 16–21):**
- Run the prediction notebook against K-Means results (swap pickle files) and record model performance
- Write a short comparison section: how does K-Means perform vs OPTICS on cluster stability, number of formations detected, and prediction AUC?
- Clean up notebook with markdown cells explaining each step

**Deliverable:** Working `kmeans-clustering.ipynb` + comparison notes (markdown or shared doc)

---

### Member B — DBScan Implementation

**Focus:** Implement the full clustering pipeline using DBScan and write up findings.

**Week 1 (Feb 9–14):**
- Create `research/dbscan-clustering.ipynb` by adapting `optics-clustering-run.ipynb`
- Key changes from OPTICS: DBScan uses `eps` (neighbourhood radius) and `min_samples` — implement a k-distance graph to tune `eps`
- Reuse the existing feature engineering code as-is
- Run clustering across the full time series; DBScan does produce noise labels (`-1`) like OPTICS, so formation detection logic should transfer more directly
- Generate equivalent pickle outputs for compatibility with the prediction notebook

**Reading Week (Feb 16–21):**
- Run the prediction notebook against DBScan results and record model performance
- Write a short comparison section: DBScan vs OPTICS on cluster shape detection, sensitivity to `eps`, and noise handling
- Clean up notebook with markdown cells explaining each step

**Deliverable:** Working `dbscan-clustering.ipynb` + comparison notes

---

### Member C — Presentation & LaTeX Document

**Focus:** Build the PowerPoint deck and LaTeX/Overleaf document.

**Week 1 (Feb 9–14):**
- Read `explanation.md` thoroughly — it has everything needed to understand the methodology
- Draft the PowerPoint structure (suggested slides below) and create a first pass of slides 1–8
- Draft the LaTeX document outline and write the Introduction, Background/Literature, and Methodology sections
- Coordinate with Members A and B to get early results screenshots for the deck

**Reading Week (Feb 16–21):**
- Finalize PowerPoint with results from all three clustering methods (OPTICS + K-Means + DBScan comparison)
- Complete the LaTeX document: Results, Discussion, and Conclusion sections
- Add figures, tables, and charts from the notebooks into both documents
- Proofread and polish both deliverables

**Suggested PowerPoint Structure:**
1. Title slide (project name, team, Quant Club @ Queen's)
2. Problem statement — why transient correlations matter
3. Data & universe — semiconductor sector, tickers, timeframe
4. Feature engineering overview — categories of features used
5. Clustering methods — OPTICS, K-Means, DBScan (one slide each with key parameters)
6. Formation/dissolution detection — how we identify transient pairs
7. Validation metrics — cluster persistence, mean reversion, bounce rate
8. Prediction models — Random Forest & Gradient Boosting approach
9. Results comparison — side-by-side of three methods (table + charts)
10. Key findings & takeaways
11. Limitations & future work
12. Q&A

**Suggested LaTeX Structure:**
1. Abstract
2. Introduction (motivation, transient vs traditional pairs trading)
3. Literature Review (clustering in finance, pairs trading research)
4. Data & Feature Engineering
5. Methodology (OPTICS, K-Means, DBScan, validation metrics, prediction models)
6. Results (per-method results + comparison)
7. Discussion
8. Conclusion & Future Work
9. References

**Deliverables:** PowerPoint (.pptx) + LaTeX document (Overleaf-ready .tex)

---

### Jack (Project Manager) — Integration, Trading Module & Oversight

**Focus:** Code review, integration, comparison analysis, trading module, and keeping everyone on track.

**Week 1 (Feb 9–14):**
- Run the existing OPTICS notebooks fresh to confirm everything works cleanly and outputs are reproducible
- Review Members A and B's code as they work — make sure pickle file formats are consistent so the prediction notebook works with all three methods
- Start a `research/comparison.ipynb` notebook that loads results from all three clustering methods and produces side-by-side comparison charts
- Begin implementing `trading/trading.py` — even a basic version: take validated pairs, compute entry/exit signals from spread z-scores, calculate hypothetical PnL
- Write the README.md (project overview, setup instructions, how to run)

**Reading Week (Feb 16–21):**
- Finalize the comparison notebook with all three methods' results
- Finish the trading module (at minimum: signal generation + simple backtest with returns)
- Review Member C's presentation and LaTeX document for technical accuracy
- Final code cleanup: ensure all notebooks have consistent structure, proper comments, and run without errors
- Final integration testing: run the full pipeline end-to-end for each method

**Deliverables:** `comparison.ipynb`, `trading/trading.py`, `README.md`, final code review sign-off

---

## Weekly Meeting Agenda

### Meeting 1 — This Week (Feb 9–14)
- Walk the team through the existing codebase (use `explanation.md` as a guide)
- Assign tasks as above
- Agree on pickle file format so all clustering notebooks produce compatible outputs
- Set a mid-week check-in (Wednesday/Thursday) for progress updates
- Member C should get access to the Overleaf project set up

### Meeting 2 — Reading Week (Feb 16–21)
- Members A and B present their clustering results
- Review comparison across methods
- Member C presents draft deck and LaTeX document for feedback
- Jack demos trading module
- Identify any gaps and assign final polish tasks

### Final Push — Feb 22–28
- Buffer week for polishing, fixing issues, and rehearsing the presentation
- Everyone reviews the final PowerPoint and LaTeX document
- Dry run of the presentation

---

## Key Technical Notes for the Team

**Reusing existing code:** The feature engineering pipeline (cells 2–5 in `optics-clustering-run.ipynb`) is method-agnostic. Members A and B should copy it directly — the only thing that changes is the clustering algorithm call itself.

**Pickle file compatibility:** All clustering notebooks should save their outputs to `research/data/` with a method prefix, e.g.:
- `kmeans_cluster_history.pkl`, `kmeans_df_formations.pkl`
- `dbscan_cluster_history.pkl`, `dbscan_df_formations.pkl`

This way the comparison notebook and prediction notebook can load any method's results.

**Key difference between methods:**
- **OPTICS** — Density-based, handles variable-density clusters, produces noise labels, no need to specify number of clusters
- **K-Means** — Centroid-based, requires specifying `k`, assigns every point to a cluster (no noise), assumes roughly spherical clusters
- **DBScan** — Density-based like OPTICS but with fixed `eps` radius, produces noise labels, doesn't handle variable-density as well as OPTICS

**Validation module:** `validation/pair_validation.py` works with any clustering output as long as the format matches. No changes needed for K-Means or DBScan results.

---

## Risk & Contingency

| Risk | Mitigation |
|------|------------|
| K-Means struggles with this data (spherical assumption) | Document the limitation — it's a valid finding for the comparison |
| DBScan sensitive to `eps` choice | Use k-distance graph, try multiple values, document sensitivity |
| Team member falls behind | Jack picks up slack; prioritize code over polish |
| Trading module too ambitious | Scale back to signal generation only (skip full backtest) |
| LaTeX formatting issues | Use a standard academic template on Overleaf to save time |

---

*Last updated: February 8, 2026*
