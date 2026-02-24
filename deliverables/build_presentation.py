"""
Build presentation.pptx for Queen's Quant Club judges.
Run: python deliverables/build_presentation.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# Colours
QUEENS_BLUE = RGBColor(0, 36, 82)
QUEENS_RED = RGBColor(185, 17, 55)
DARK_GRAY = RGBColor(60, 60, 60)
LIGHT_GRAY = RGBColor(240, 240, 240)
WHITE = RGBColor(255, 255, 255)
GREEN = RGBColor(39, 174, 96)
RED = RGBColor(231, 76, 60)
BLACK = RGBColor(0, 0, 0)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_bg(slide, color=WHITE):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def text_box(slide, left, top, width, height, text, font_size=18,
             bold=False, color=DARK_GRAY, alignment=PP_ALIGN.LEFT, font_name='Calibri'):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return tf


def bullet_slide(slide, left, top, width, height, items, font_size=16, color=DARK_GRAY):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = 'Calibri'
        p.space_after = Pt(6)
        p.level = 0
    return tf


def add_table(slide, left, top, width, height, headers, rows, col_widths=None):
    n_rows = len(rows) + 1
    n_cols = len(headers)
    table_shape = slide.shapes.add_table(n_rows, n_cols, Inches(left), Inches(top),
                                          Inches(width), Inches(height))
    table = table_shape.table

    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = Inches(w)

    # Header row
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        for paragraph in cell.text_frame.paragraphs:
            paragraph.font.size = Pt(12)
            paragraph.font.bold = True
            paragraph.font.color.rgb = WHITE
            paragraph.font.name = 'Calibri'
            paragraph.alignment = PP_ALIGN.CENTER
        cell.fill.solid()
        cell.fill.fore_color.rgb = QUEENS_BLUE

    # Data rows
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = str(val)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(11)
                paragraph.font.color.rgb = DARK_GRAY
                paragraph.font.name = 'Calibri'
                paragraph.alignment = PP_ALIGN.CENTER
            if i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_GRAY

    return table


def section_header(slide, title, subtitle=""):
    add_bg(slide, QUEENS_BLUE)
    text_box(slide, 1.5, 2.5, 10, 1.5, title, font_size=40, bold=True, color=WHITE,
             alignment=PP_ALIGN.LEFT)
    if subtitle:
        text_box(slide, 1.5, 4.2, 10, 1, subtitle, font_size=20, color=RGBColor(180, 200, 220),
                 alignment=PP_ALIGN.LEFT)


# ============================================================================
# SLIDE 1: Title
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank
add_bg(slide, QUEENS_BLUE)

text_box(slide, 1.5, 1.5, 10, 1.5,
         "Clustering-Driven Pair Discovery\nFrom Semiconductors to Cross-Sector Markets",
         font_size=36, bold=True, color=WHITE, alignment=PP_ALIGN.LEFT)

text_box(slide, 1.5, 3.8, 10, 0.8,
         "Unsupervised Learning for Transient Correlation Detection Across 142 Tickers",
         font_size=20, color=RGBColor(180, 200, 220), alignment=PP_ALIGN.LEFT)

text_box(slide, 1.5, 5.5, 10, 0.6,
         "Queen's Quant Club",
         font_size=18, color=RGBColor(250, 189, 0), alignment=PP_ALIGN.LEFT)


# ============================================================================
# SLIDE 2: Agenda
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 5, 0.8, "Agenda", font_size=32, bold=True, color=QUEENS_BLUE)

items = [
    "1.  Motivation: Why clustering for pair discovery?",
    "2.  Methodology: Features, algorithms, pipeline",
    "3.  Phase 1: Semiconductor pair discovery & validation",
    "4.  Phase 2: Cross-sector expansion (142 tickers, 5 sectors)",
    "5.  5-Test Validation: Replacing cointegration",
    "6.  Cross-Sector Results: 3,712 pairs scored",
    "7.  Enhanced Backtesting: Kalman hedge, walk-forward",
    "8.  Statistical Significance: Permutation testing",
    "9.  Key Insights & Future Work",
]
bullet_slide(slide, 1.2, 1.5, 10, 5, items, font_size=20, color=DARK_GRAY)


# ============================================================================
# SLIDE 3: Section - Motivation
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
section_header(slide, "Motivation", "Why clustering for pair discovery?")


# ============================================================================
# SLIDE 4: The Problem
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 8, 0.8, "The Problem: Finding Pairs to Trade", font_size=28, bold=True, color=QUEENS_BLUE)

# Left column - Traditional
text_box(slide, 0.8, 1.5, 5.5, 0.6, "Traditional Approach", font_size=20, bold=True, color=QUEENS_RED)
bullet_slide(slide, 0.8, 2.2, 5.5, 3, [
    "Screen all C(N,2) combinations for cointegration",
    "For 142 stocks: 10,011 pairs to test",
    "High false discovery rate from multiple testing",
    "No insight into WHY pairs are related",
], font_size=16)

# Right column - Our approach
text_box(slide, 7, 1.5, 5.5, 0.6, "Our Approach", font_size=20, bold=True, color=GREEN)
bullet_slide(slide, 7, 2.2, 5.5, 3, [
    "Let clustering discover similar-behaving stocks",
    "Validate with a 5-test scored framework",
    "Compare across 3 different algorithms",
    "142 tickers, 5 sectors, 11 months hourly data",
], font_size=16)

# Key idea box
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(2.5), Inches(5.3),
                                Inches(8), Inches(1.5))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(230, 240, 250)
shape.line.color.rgb = QUEENS_BLUE
tf = shape.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Key Idea: Stocks that consistently cluster together based on real-time feature similarity are candidates for pairs trading."
p.font.size = Pt(16)
p.font.color.rgb = QUEENS_BLUE
p.font.name = 'Calibri'
p.font.bold = True
p.alignment = PP_ALIGN.CENTER


# ============================================================================
# SLIDE 5: Section - Methodology
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
section_header(slide, "Methodology", "Features, algorithms, and pipeline design")


# ============================================================================
# SLIDE 6: Pipeline
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 10, 0.8, "Pipeline Overview", font_size=28, bold=True, color=QUEENS_BLUE)

# Pipeline flow
text_box(slide, 1.5, 1.4, 10, 0.7,
         "Data  \u2192  Features (9/ticker)  \u2192  Standardize + PCA  \u2192  Cluster  \u2192  Discover Pairs  \u2192  Validate  \u2192  Backtest",
         font_size=18, bold=True, color=QUEENS_BLUE, alignment=PP_ALIGN.CENTER)

# Features table
headers = ["Category", "Feature", "Window"]
rows = [
    ["Volatility", "Short-term rolling \u03c3", "50h (~1 week)"],
    ["Market exposure", "Rolling \u03b2 to S&P 500", "50h"],
    ["Sector exposure", "Rolling \u03b2 to leave-one-out sector avg", "50h"],
    ["Momentum", "RSI", "70h (~2 weeks)"],
    ["Momentum", "5-hour return", "Instantaneous"],
    ["Regime shift", "\u0394\u03c3 (short \u2212 medium volatility)", "50h vs 147h"],
    ["Regime shift", "\u0394\u03b2_SPX (short \u2212 medium)", "50h vs 147h"],
    ["Regime shift", "\u0394\u03b2_sector (short \u2212 medium)", "50h vs 147h"],
    ["Returns", "Hourly return", "Instantaneous"],
]
add_table(slide, 1.5, 2.3, 10, 4.5, headers, rows, col_widths=[2.5, 4.5, 3.0])


# ============================================================================
# SLIDE 7: Three Algorithms
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 10, 0.8, "Three Algorithms, One Question",
         font_size=28, bold=True, color=QUEENS_BLUE)

text_box(slide, 0.8, 1.2, 11, 0.6,
         "If different algorithms with different inductive biases find the same pairs, those relationships are robust.",
         font_size=16, color=DARK_GRAY, alignment=PP_ALIGN.LEFT)

headers = ["", "OPTICS", "DBSCAN", "KMeans"]
rows = [
    ["Type", "Density-based", "Density-based", "Centroid-based"],
    ["k required?", "No (auto)", "No (auto)", "Yes (silhouette)"],
    ["Noise handling", "Native (-1 label)", "Native", "Post-hoc filter"],
    ["Avg noise rate", "58%", "28%", "~10%"],
    ["Avg clusters/ts", "3.3", "1.5", "3.1"],
    ["OOS stability (r)", "0.671", "0.832", "0.627"],
]
add_table(slide, 1.5, 2.0, 10, 3.5, headers, rows, col_widths=[2.8, 2.4, 2.4, 2.4])

text_box(slide, 0.8, 5.7, 11, 1.0,
         "OPTICS selected as primary algorithm for Phase 2: best at detecting variable-density clusters\n"
         "and transient relationship formations. Only 2 consensus pairs across all 3 algorithms' top-20\n"
         "(COP-DVN, CVX-XOM) \u2014 but permutation testing confirms 943 statistically significant pairs.",
         font_size=13, color=RGBColor(100, 100, 100))


# ============================================================================
# SLIDE 8: Section - Phase 1 Semiconductors
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
section_header(slide, "Phase 1: Semiconductors", "40 tickers, proof of concept")


# ============================================================================
# SLIDE 9: Consensus Pairs
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 10, 0.8, "8 Consensus Pairs Across All Algorithms",
         font_size=28, bold=True, color=QUEENS_BLUE)

headers = ["Pair", "OPTICS", "KMeans", "DBSCAN", "Subsector"]
rows = [
    ["KLAC\u2013LRCX", "0.378", "0.785", "0.851", "Equipment"],
    ["QRVO\u2013SWKS", "0.372", "0.888", "0.828", "RF"],
    ["AMAT\u2013LRCX", "0.363", "0.824", "0.845", "Equipment"],
    ["AMAT\u2013KLAC", "0.340", "0.807", "0.814", "Equipment"],
    ["ADI\u2013NXPI", "0.279", "0.816", "0.900", "Analog"],
    ["ADI\u2013TXN", "0.278", "0.887", "0.817", "Analog"],
    ["NXPI\u2013TXN", "0.223", "0.819", "0.806", "Analog"],
    ["ADI\u2013SWKS", "0.216", "0.830", "0.830", "Analog/RF"],
]
add_table(slide, 1.8, 1.4, 9.5, 4.5, headers, rows, col_widths=[2.0, 1.5, 1.5, 1.5, 2.0])

text_box(slide, 0.8, 6.0, 11, 0.8,
         "14/26 tradeable pairs profitable OOS (54%) with simple z-score strategy.\n"
         "Top Sharpe: NXPI\u2013ON 3.55, CDNS\u2013SNPS 2.08. Caveat: 2\u20135 trades per pair.",
         font_size=14, color=DARK_GRAY)


# ============================================================================
# SLIDE 10: Subsector Discovery
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 10, 0.8, "The Algorithm Discovers Industry Structure",
         font_size=28, bold=True, color=QUEENS_BLUE)

text_box(slide, 0.8, 1.3, 11, 0.6,
         "No fundamental data as input \u2014 only price-derived features. Yet the clustering independently discovers the semiconductor subsector map:",
         font_size=16, color=DARK_GRAY)

clusters = [
    ("Equipment", "AMAT, LRCX, KLAC", "Shared fab customers (TSMC, Samsung, Intel)"),
    ("RF / 5G", "QRVO, SWKS, QCOM", "Smartphone and 5G demand"),
    ("Analog", "ADI, NXPI, TXN, MCHP", "Industrial and automotive end markets"),
    ("EDA", "CDNS, SNPS", "Chip design software duopoly"),
]

for i, (name, tickers, desc) in enumerate(clusters):
    y = 2.3 + i * 1.15
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    Inches(1.2), Inches(y), Inches(10.8), Inches(0.95))
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT_GRAY
    shape.line.color.rgb = QUEENS_BLUE

    text_box(slide, 1.5, y + 0.05, 2.2, 0.5, name, font_size=18, bold=True, color=QUEENS_BLUE)
    text_box(slide, 3.8, y + 0.05, 3, 0.5, tickers, font_size=16, bold=True, color=DARK_GRAY)
    text_box(slide, 7, y + 0.05, 4.8, 0.5, desc, font_size=14, color=RGBColor(100, 100, 100))

text_box(slide, 0.8, 6.5, 11, 0.5,
         "Data-driven clustering converges with known industry structure \u2192 relationships are real, not noise.",
         font_size=16, bold=True, color=QUEENS_BLUE, alignment=PP_ALIGN.CENTER)


# ============================================================================
# SLIDE 11: Section - Phase 2 Cross-Sector
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
section_header(slide, "Phase 2: Cross-Sector", "142 tickers, 5 sectors, 3,712 pairs")


# ============================================================================
# SLIDE 12: Universe Construction
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 10, 0.8, "Universe Construction: 3-Layer Pipeline",
         font_size=28, bold=True, color=QUEENS_BLUE)

# Three layers
layers = [
    ("Layer 1: Liquidity", "US-listed, market cap > $2B, volume > 5M/day, price > $5"),
    ("Layer 2: Sector Grouping", "Technology, Healthcare, Energy, Financial Services, Industrials"),
    ("Layer 3: Quality", "Positive EBITDA \u2014 removes speculative pre-revenue names"),
]
for i, (name, desc) in enumerate(layers):
    y = 1.5 + i * 1.0
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    Inches(1.2), Inches(y), Inches(10.8), Inches(0.8))
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT_GRAY
    shape.line.color.rgb = QUEENS_BLUE
    text_box(slide, 1.5, y + 0.1, 3.5, 0.5, name, font_size=17, bold=True, color=QUEENS_BLUE)
    text_box(slide, 5.2, y + 0.1, 6.5, 0.5, desc, font_size=15, color=DARK_GRAY)

# Key design decision
text_box(slide, 0.8, 4.8, 11, 0.5, "Key Design Decision: Single Combined Universe",
         font_size=20, bold=True, color=QUEENS_BLUE)
bullet_slide(slide, 1.2, 5.4, 10, 2, [
    "All 142 tickers clustered together \u2014 not per-sector",
    "Cross-sector pairs discovered organically (e.g., RTX-SHEL: Industrials/Energy)",
    "If intra-sector pairs dominate, that's an empirical result, not a constraint",
    "Result: 8,065 co-clustering pairs identified; 3,712 above noise-adjusted threshold",
], font_size=15, color=DARK_GRAY)


# ============================================================================
# SLIDE 13: Section - 5-Test Validation
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
section_header(slide, "5-Test Validation", "Replacing cointegration with a scored framework")


# ============================================================================
# SLIDE 14: Why Cointegration Fails + 5-Test Framework
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 10, 0.8, "Why Cointegration Fails on Transient Correlations",
         font_size=28, bold=True, color=QUEENS_BLUE)

# Problem box
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.3),
                                Inches(5.5), Inches(2.0))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(255, 235, 235)
shape.line.color.rgb = QUEENS_RED
text_box(slide, 1.1, 1.4, 5, 0.4, "The Problem", font_size=18, bold=True, color=QUEENS_RED)
bullet_slide(slide, 1.1, 1.9, 5, 1.5, [
    "Engle-Granger cointegration: 0% pass rate",
    "Tests for long-run equilibrium \u2014 wrong model",
    "Yet Hurst passes 90.6%, Half-life 82.3%",
    "Spreads DO mean-revert, just not permanently",
], font_size=14, color=DARK_GRAY)

# Solution box
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7), Inches(1.3),
                                Inches(5.5), Inches(2.0))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(230, 250, 235)
shape.line.color.rgb = GREEN
text_box(slide, 7.3, 1.4, 5, 0.4, "The Solution: 5-Test Scored Framework", font_size=18, bold=True, color=GREEN)
bullet_slide(slide, 7.3, 1.9, 5, 1.5, [
    "Each test scores 0 or 1 (total 0\u20135)",
    "Strong (4\u20135), Moderate (3), Weak (2), Fail (<2)",
    "58.4% of pairs are tradeable (score \u2265 3)",
    "vs 0% with cointegration!",
], font_size=14, color=DARK_GRAY)

# Framework table
headers = ["Test", "Threshold", "Pass Rate", "Role"]
rows = [
    ["ADF on spread", "p < 0.10", "38.4%", "Spread stationarity (hard)"],
    ["Half-life", "5\u201360 days", "82.3%", "Mean-reversion speed"],
    ["Hurst exponent", "H < 0.5", "90.6%", "Mean-reverting (easy)"],
    ["Variance ratio", "Reject RW @10%", "21.2%", "Short-run MR (hardest)"],
    ["Rolling corr", "Stability > 0.5", "33.7%", "Return correlation persistence"],
]
add_table(slide, 1.5, 3.6, 10, 3.2, headers, rows, col_widths=[2.2, 2.0, 1.5, 4.3])


# ============================================================================
# SLIDE 15: Section - Cross-Sector Results
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
section_header(slide, "Cross-Sector Results", "3,712 pairs scored across 5 sectors")


# ============================================================================
# SLIDE 16: Score Distribution + Sector Matrix
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 10, 0.8, "Score Distribution & Sector Breakdown",
         font_size=28, bold=True, color=QUEENS_BLUE)

# Score distribution (left)
text_box(slide, 0.8, 1.3, 5, 0.5, "Score Distribution (3,712 pairs)", font_size=18, bold=True, color=QUEENS_BLUE)
headers = ["Classification", "Count", "%"]
rows = [
    ["Strong (4\u20135)", "596", "16.1%"],
    ["Moderate (3)", "1,572", "42.3%"],
    ["Weak (2)", "1,235", "33.3%"],
    ["Fail (0\u20131)", "309", "8.3%"],
]
add_table(slide, 0.8, 1.9, 5.2, 2.5, headers, rows, col_widths=[2.2, 1.5, 1.5])

# Sector matrix (right)
text_box(slide, 7, 1.3, 5.5, 0.5, "Top Sector Combinations", font_size=18, bold=True, color=QUEENS_BLUE)
headers = ["Sectors", "Pairs", "Avg Score"]
rows = [
    ["Technology\u2013Technology", "903", "2.62"],
    ["Energy\u2013Energy", "262", "3.10"],
    ["Industrials\u2013Technology", "250", "2.82"],
    ["Healthcare\u2013Technology", "221", "2.59"],
    ["Healthcare\u2013Healthcare", "183", "2.88"],
]
add_table(slide, 7, 1.9, 5.5, 2.8, headers, rows, col_widths=[2.8, 1.2, 1.5])

# Intra vs Cross comparison
text_box(slide, 0.8, 4.8, 11, 0.5, "Intra-Sector vs Cross-Sector", font_size=18, bold=True, color=QUEENS_BLUE)
headers = ["Type", "Pairs", "Avg Score", "Pass Rate", "Avg Sharpe", "Best Sharpe"]
rows = [
    ["Intra-sector", "1,464", "2.75", "19.3%", "1.95", "1.96"],
    ["Cross-sector", "2,248", "2.60", "13.9%", "1.59", "2.56"],
]
add_table(slide, 1.5, 5.4, 10, 1.5, headers, rows, col_widths=[2.0, 1.2, 1.5, 1.5, 1.8, 2.0])


# ============================================================================
# SLIDE 17: Cross-Sector Discoveries
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 10, 0.8, "Cross-Sector Discoveries",
         font_size=28, bold=True, color=QUEENS_BLUE)

# RTX-SHEL
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.3),
                                Inches(5.5), Inches(2.0))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(230, 240, 250)
shape.line.color.rgb = QUEENS_BLUE
text_box(slide, 1.1, 1.4, 5, 0.4, "RTX\u2013SHEL (Score 4, Strong)", font_size=18, bold=True, color=QUEENS_BLUE)
bullet_slide(slide, 1.1, 1.9, 5, 1.2, [
    "Industrials / Energy cross-sector link",
    "Defense contractor \u2194 Energy major",
    "Enhanced PnL: +10.07, Kalman: +13.87",
], font_size=14, color=DARK_GRAY)

# APLD Cluster
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7), Inches(1.3),
                                Inches(5.5), Inches(2.0))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(230, 240, 250)
shape.line.color.rgb = QUEENS_BLUE
text_box(slide, 7.3, 1.4, 5, 0.4, "APLD Crypto Mining Cluster", font_size=18, bold=True, color=QUEENS_BLUE)
bullet_slide(slide, 7.3, 1.9, 5, 1.2, [
    "APLD/HUT/CIFR/IREN across Tech & FinServ",
    "APLD-HUT: Score 4 (Strong)",
    "Meaningful economic link, not noise",
], font_size=14, color=DARK_GRAY)

# Top pairs table
text_box(slide, 0.8, 3.6, 10, 0.5, "Top Pairs (by score then Sharpe, min 5 trades)", font_size=18, bold=True, color=QUEENS_BLUE)
headers = ["Type", "Pair", "Score", "ADF p", "Hurst", "Trades", "Sharpe"]
rows = [
    ["Cross", "ADT\u2013NOW", "4", "0.0002", "0.11", "5", "2.56"],
    ["Intra", "APH\u2013TSM", "4", "0.529", "0.49", "5", "1.93"],
    ["Cross", "CRBG\u2013NOW", "4", "0.323", "0.33", "5", "1.64"],
    ["Intra", "CRBG\u2013WU", "3", "0.178", "0.23", "5", "1.96"],
    ["Cross", "DXCM\u2013NOK", "3", "0.312", "0.29", "5", "0.57"],
]
add_table(slide, 1.5, 4.2, 10, 2.8, headers, rows, col_widths=[1.0, 2.0, 1.0, 1.2, 1.2, 1.2, 1.2])


# ============================================================================
# SLIDE 18: Section - Enhanced Backtesting
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
section_header(slide, "Enhanced Backtesting", "Adaptive z-scores, Kalman hedge, transaction costs")


# ============================================================================
# SLIDE 19: Three-Strategy Comparison
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 10, 0.8, "Three-Strategy Comparison",
         font_size=28, bold=True, color=QUEENS_BLUE)

headers = ["Strategy", "Description", "Trades/Pair", "Profitable", "Avg PnL"]
rows = [
    ["Baseline", "Static z=2.0, OLS, no costs", "2.3", "53%", "0.356"],
    ["Enhanced", "Optimized z, OLS, 10bps", "2.9", "50%", "0.378"],
    ["Kalman", "Optimized z, Kalman \u03b2, 10bps", "2.8", "59%", "0.408"],
]
add_table(slide, 1.5, 1.3, 10, 2.2, headers, rows, col_widths=[1.5, 3.5, 1.5, 1.5, 1.5])

# Kalman fix explanation
text_box(slide, 0.8, 3.8, 5.5, 0.5, "Kalman Hedge Ratio Fix", font_size=20, bold=True, color=QUEENS_BLUE)
bullet_slide(slide, 0.8, 4.4, 5.5, 2.5, [
    "Adaptive Kalman spread (beta updated each step)",
    "  \u2192 Creates artificial mean-reversion by construction",
    "  \u2192 Prediction error ~ white noise, trivially tradeable",
    "",
    "Fix: Kalman on calibration only \u2192 terminal beta",
    "  \u2192 Fixed beta applied to OOS = honest test",
], font_size=14, color=DARK_GRAY)

# Top enhanced pairs
text_box(slide, 7, 3.8, 5.5, 0.5, "Top Enhanced Pairs", font_size=20, bold=True, color=QUEENS_BLUE)
headers = ["Pair", "Type", "Enhanced PnL", "Kalman PnL"]
rows = [
    ["AMAT-TSM", "intra", "+30.30", "+54.19"],
    ["CVX-SU", "intra", "+13.16", "+10.41"],
    ["APLD-IREN", "cross", "+10.40", "+9.47"],
    ["RTX-SHEL", "cross", "+10.07", "+13.87"],
    ["HUT-RIOT", "intra", "+9.91", "+5.65"],
]
add_table(slide, 7, 4.4, 5.5, 2.8, headers, rows, col_widths=[1.5, 0.8, 1.5, 1.5])


# ============================================================================
# SLIDE 20: Walk-Forward Validation
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 10, 0.8, "Walk-Forward Validation",
         font_size=28, bold=True, color=QUEENS_BLUE)

text_box(slide, 0.8, 1.2, 11, 0.6,
         "5-split rolling calibration/OOS windows. Tests whether performance persists across multiple temporal splits, not just one lucky window.",
         font_size=16, color=DARK_GRAY)

headers = ["Pair", "Splits", "Avg Sharpe", "Std Sharpe", "Best Split", "Total Trades"]
rows = [
    ["RUN\u2013VNET", "5", "3.47", "1.06", "4.08", "13"],
    ["NVDA\u2013ORCL", "5", "2.59", "3.10", "0.39", "9"],
]
add_table(slide, 2.5, 2.2, 8, 1.8, headers, rows, col_widths=[1.8, 1.0, 1.3, 1.3, 1.3, 1.3])

# Insight box
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(2), Inches(4.5),
                                Inches(9), Inches(2.2))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(230, 240, 250)
shape.line.color.rgb = QUEENS_BLUE
tf = shape.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Walk-forward addresses the single-split limitation"
p.font.size = Pt(20)
p.font.bold = True
p.font.color.rgb = QUEENS_BLUE
p.font.name = 'Calibri'
p.alignment = PP_ALIGN.CENTER
p2 = tf.add_paragraph()
p2.text = ("RUN-VNET: Sharpe 3.47 with low variance (std 1.06) across 5 splits \u2192 robust signal.\n"
           "NVDA-ORCL: Sharpe 2.59 but high variance (std 3.10) \u2192 less reliable.\n"
           "Walk-forward was listed as Future Work in Phase 1 \u2014 now completed.")
p2.font.size = Pt(14)
p2.font.color.rgb = DARK_GRAY
p2.font.name = 'Calibri'
p2.alignment = PP_ALIGN.CENTER


# ============================================================================
# SLIDE 21: Section - Statistical Significance
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
section_header(slide, "Statistical Significance", "Permutation testing on clustering structure")


# ============================================================================
# SLIDE 22: Permutation Test Results
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 10, 0.8, "Permutation Test: 943 Significant Pairs",
         font_size=28, bold=True, color=QUEENS_BLUE)

text_box(slide, 0.8, 1.2, 11, 0.8,
         "Feature-shuffle test: at 80 sampled timestamps across 30 permutations, shuffle feature vectors across tickers "
         "(preserving cross-feature correlation) and re-run the full pipeline. Compare observed co-clustering frequency to null distribution.",
         font_size=15, color=DARK_GRAY)

# Summary stats
text_box(slide, 0.8, 2.2, 4, 0.5, "8,086 pairs tested", font_size=20, bold=True, color=QUEENS_BLUE)
text_box(slide, 4, 2.2, 4, 0.5, "943 significant (Z > 1.96)", font_size=20, bold=True, color=GREEN)
text_box(slide, 8.5, 2.2, 4, 0.5, "575 significant AND tradeable", font_size=20, bold=True, color=QUEENS_BLUE)

# Top significant pairs
headers = ["Pair", "Z-Score", "Sector"]
rows = [
    ["PBR\u2013PBR-A", "39.92", "Energy"],
    ["AAL\u2013DAL", "38.10", "Industrials"],
    ["CVX\u2013XOM", "34.23", "Energy"],
    ["DAL\u2013UAL", "32.92", "Industrials"],
    ["AAL\u2013UAL", "29.96", "Industrials"],
    ["COP\u2013OXY", "29.44", "Energy"],
    ["BP\u2013COP", "23.89", "Energy"],
    ["HAL\u2013SLB", "23.67", "Energy"],
    ["COP\u2013DVN", "23.13", "Energy"],
    ["AMAT\u2013LRCX", "22.55", "Technology"],
]
add_table(slide, 3, 3.0, 7, 4.0, headers, rows, col_widths=[2.5, 1.5, 3.0])


# ============================================================================
# SLIDE 23: Section - Key Insights
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
section_header(slide, "Key Insights", "What works, what doesn't, and what's next")


# ============================================================================
# SLIDE 24: What Works / Limitations
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 10, 0.8, "What We Learned",
         font_size=28, bold=True, color=QUEENS_BLUE)

# What works
text_box(slide, 0.8, 1.3, 5.5, 0.5, "What Works", font_size=20, bold=True, color=GREEN)
bullet_slide(slide, 0.8, 1.9, 5.8, 3.5, [
    "Clustering discovers real economic relationships",
    "  \u2014 semiconductor subsectors, crypto mining clusters",
    "5-test framework: 58% tradeable vs 0% cointegration",
    "Kalman hedge: 59% profitable (vs 53% OLS baseline)",
    "Walk-forward validates top pairs across 5 splits",
    "943 pairs confirmed by permutation test",
], font_size=15, color=DARK_GRAY)

# Limitations
text_box(slide, 7, 1.3, 5.5, 0.5, "Limitations", font_size=20, bold=True, color=QUEENS_RED)
bullet_slide(slide, 7, 1.9, 5.8, 3.5, [
    "Cross-sector pairs rare (5 at 0.08 threshold)",
    "Short OOS (~68 days): 2\u20133 trades per pair",
    "Consensus \u2260 profitability (Phase 1 finding)",
    "10bps flat cost is simplified",
    "No portfolio-level risk management",
], font_size=15, color=DARK_GRAY)

# Bottom line
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(2), Inches(5.5),
                                Inches(9), Inches(1.3))
shape.fill.solid()
shape.fill.fore_color.rgb = QUEENS_BLUE
shape.line.color.rgb = QUEENS_BLUE
tf = shape.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "The framework generalizes beyond semiconductors. Transient correlations are real,\nstatistically significant, and partially tradeable \u2014 but trade count remains the binding constraint."
p.font.size = Pt(17)
p.font.bold = True
p.font.color.rgb = WHITE
p.font.name = 'Calibri'
p.alignment = PP_ALIGN.CENTER


# ============================================================================
# SLIDE 25: Future Work + Thank You
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 10, 0.8, "Future Work",
         font_size=28, bold=True, color=QUEENS_BLUE)

items = [
    ("Walk-Forward Validation", "5-split rolling cal/OOS", True),
    ("Adaptive Hedge Ratios", "Kalman terminal beta estimation", True),
    ("Expanded Universe", "142 tickers, 5 sectors via 3-layer screening", True),
    ("Transaction Costs", "10bps round-trip incorporated", True),
    ("Real-Time Pipeline", "Live formation \u2192 validation \u2192 signal generation", False),
    ("Portfolio Optimization", "Correlated pairs risk management", False),
    ("Longer Historical Data", "Multi-year hourly data for robust estimation", False),
]

for i, (title, desc, done) in enumerate(items):
    y = 1.3 + i * 0.75
    status = "\u2705" if done else "\u2B1C"
    color = GREEN if done else DARK_GRAY
    text_box(slide, 1.0, y, 0.5, 0.5, status, font_size=18, color=color)
    text_box(slide, 1.6, y, 4, 0.5, title, font_size=17, bold=True, color=QUEENS_BLUE)
    text_box(slide, 5.8, y, 6.5, 0.5, desc, font_size=15, color=DARK_GRAY)

# Thank you
text_box(slide, 3, 6.5, 7, 0.8, "Thank You \u2014 Questions?",
         font_size=28, bold=True, color=QUEENS_BLUE, alignment=PP_ALIGN.CENTER)


# ============================================================================
# SAVE
# ============================================================================
import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presentation.pptx')
prs.save(out_path)
print(f"Saved: {out_path}")
print(f"Slides: {len(prs.slides)}")
