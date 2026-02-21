# Report & Presentation Outline

## Report Structure

### 1. Introduction
- Problem statement: traditional pairs trading assumes permanent cointegration, but many real relationships are transient and event-driven
- Research question: can density-based clustering on technical features detect these transient relationships in real-time, and do they have tradeable properties?
- Universe: 40 semiconductor stocks, ~11 months hourly data

### 2. Methodology
- Feature engineering: multi-timeframe approach (short-term regime indicators + medium-term context + regime shift detection)
- Clustering pipeline: StandardScaler -> PCA (90% variance) -> clustering at each hourly timestamp
- Three algorithms tested: OPTICS, DBSCAN, KMeans (with algorithm-specific adaptations)
- Validation framework: calibration/exploitation window separation, execution lag, random baseline, permutation test

### 3. Transient Relationship Detection Results
- Clustering quality across algorithms (noise rates, cluster counts, parameter sensitivity)
- Top co-clustering pairs and subsector structure they reveal
- Formation/dissolution event statistics and duration distributions
- Permutation test: which pairs are statistically significant
- OOS stability: do patterns hold out-of-sample
- Algorithm comparison: which captures transient relationships best and why

### 4. Long-Term Relationship Investigation
- Hypothesis: pairs that repeatedly co-cluster on hourly data may have tradeable long-term properties on daily data
- Cointegration testing (Engle-Granger + Johansen) on 5-year daily data
- Rolling cointegration: persistent vs regime-dependent relationships
- Half-life, Hurst exponent, hedge ratio stability
- Cross-frequency validation: does hourly co-clustering align with daily cointegration regimes
- Which pairs (if any) are tradeable at longer timeframes

### 5. Trading Strategy & Backtest
- Transient strategy: z-score mean-reversion on validated formation events
- Long-term strategy: Bollinger/z-score on cointegrated pairs (if found)
- P&L results, win rates, Sharpe ratios, drawdowns
- Clustered pairs vs random baseline performance

### 6. Conclusions & Limitations
- What works: real-time scanning, subsector discovery, validated transient relationships
- What doesn't: duration prediction from price features alone
- Limitations: single sector, limited timeframe, no external data, no transaction costs modeled
- Future work: external data integration, universe expansion, live implementation

---

## Presentation Structure (~15-20 slides)

### Opening (2 slides)
- **Slide 1**: Title + one-sentence summary ("Using density-based clustering to detect and trade transient correlations in semiconductor stocks")
- **Slide 2**: The core idea — visual showing two stocks that temporarily move together then diverge, with the question "can we detect this in real-time?"

### Methodology (3-4 slides)
- **Slide 3**: Pipeline diagram — data fetch -> features -> PCA -> clustering -> events -> validation -> signals (one clean flow diagram)
- **Slide 4**: Feature engineering summary — table of 13 features grouped by category, emphasize multi-timeframe + regime shift indicators
- **Slide 5**: Three clustering algorithms — one bullet each on OPTICS/DBSCAN/KMeans, why we're comparing them
- **Slide 6** (optional): Validation framework — the three-window diagram (execution lag | calibration | exploitation)

### Results (5-6 slides)
- **Slide 7**: Clustering quality — key stats table (valid timestamps, avg clusters, noise rate), algorithm comparison
- **Slide 8**: Top pairs discovered — table with subsector labels, highlight that clustering finds real industry structure without being told about it
- **Slide 9**: Statistical validation — permutation test Z-scores and OOS correlation, the "this is real" slide
- **Slide 10**: Algorithm comparison — side-by-side metrics, which wins and why
- **Slide 11**: Long-term investigation results — cointegration findings on daily data, the bridge from transient to long-term
- **Slide 12**: Trading performance — P&L charts, clustered vs random lift, key backtest metrics

### Key Takeaways (2-3 slides)
- **Slide 13**: What we found — 3-4 bullet summary of main conclusions
- **Slide 14**: Limitations and what we'd do with more time
- **Slide 15**: Questions

### Appendix (as needed)
- Detailed parameter sensitivity tables
- Full pair lists
- Duration distribution plots
- Additional backtest details

---

## Visual Assets to Prepare

These are the charts/figures worth making look polished for the presentation:

1. **Pipeline flow diagram** — the single most important visual for explaining the methodology
2. **Top pairs table with subsector coloring** — shows clustering finds real structure
3. **Clustered vs random pass rate bar chart** — the "value-add" proof
4. **Duration distribution histogram** — shows transient nature visually
5. **Co-clustering frequency scatter** (OPTICS vs DBSCAN vs KMeans) — algorithm comparison
6. **Rolling cointegration heatmap** — if long-term investigation finds regime-dependent cointegration, this is a compelling visual
7. **Example trade plot** — one good pair showing prices, spread, z-score signals, and P&L for a single formation event (already exists in optics-signals.ipynb visualization cells)

---

## Notes

- The report should be thorough and technical (include methodology details, exact metrics, statistical tests)
- The presentation should be visual and narrative-driven (lead with the question, show the answer, minimize text on slides)
- Both should emphasize the algorithm comparison since that's a concrete deliverable
- If long-term cointegration is found for any pairs, that becomes the headline finding; if not, the headline is "transient detection works, prediction doesn't, here's why"
