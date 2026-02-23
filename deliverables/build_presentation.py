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
         "Clustering-Driven Pair Discovery\nin Semiconductor Equities",
         font_size=36, bold=True, color=WHITE, alignment=PP_ALIGN.LEFT)

text_box(slide, 1.5, 3.8, 10, 0.8,
         "Using Unsupervised Learning to Identify Tradeable Relationships",
         font_size=20, color=RGBColor(180, 200, 220), alignment=PP_ALIGN.LEFT)

text_box(slide, 1.5, 5.5, 10, 0.6,
         "Queen's Quant Club",  # TODO: Add names
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
    "3.  Pair Discovery: Consensus across 3 algorithms",
    "4.  Validation: Classical pairs trading tests",
    "5.  Out-of-Sample Results: Honest backtesting",
    "6.  Key Insights & Lessons Learned",
    "7.  Future Work",
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
    "For 40 stocks: 780 pairs to test",
    "High false discovery rate from multiple testing",
    "No insight into WHY pairs are related",
], font_size=16)

# Right column - Our approach
text_box(slide, 7, 1.5, 5.5, 0.6, "Our Approach", font_size=20, bold=True, color=GREEN)
bullet_slide(slide, 7, 2.2, 5.5, 3, [
    "Let clustering discover similar-behaving stocks",
    "Validate discovered pairs with classical tests",
    "Compare across 3 different algorithms",
    "40 semiconductors, 11 months hourly data",
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
add_table(slide, 1.5, 2.0, 10, 4.0, headers, rows, col_widths=[2.8, 2.4, 2.4, 2.4])

text_box(slide, 0.8, 6.2, 11, 0.6,
         "OOS stability = Pearson correlation of co-clustering frequencies between first 67% and last 33% of timestamps.",
         font_size=13, color=RGBColor(120, 120, 120))


# ============================================================================
# SLIDE 8: Section - Pair Discovery
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
section_header(slide, "Pair Discovery", "All three algorithms converge on the same pairs")


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
         "OPTICS frequencies are 2\u20134x lower due to 58% noise rate. Noise-adjusted rates are comparable.\n11 pairs statistically significant via feature-shuffle permutation test (Z > 1.96, p < 0.05).",
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
# SLIDE 11: Section - Validation
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
section_header(slide, "Pair Validation", "Classical pairs trading tests on discovered pairs")


# ============================================================================
# SLIDE 12: Validation Results
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 10, 0.8, "Classical Pairs Trading Validation",
         font_size=28, bold=True, color=QUEENS_BLUE)

# Left - Criteria
text_box(slide, 0.8, 1.3, 6, 0.5, "Three criteria on calibration window (first 67%):",
         font_size=16, bold=True, color=DARK_GRAY)
bullet_slide(slide, 0.8, 1.9, 6, 2.5, [
    "1. Engle-Granger cointegration (p < 0.05, bidirectional)",
    "2. Half-life of mean reversion: 5\u201360 days",
    "3. Hurst exponent < 0.5 (mean-reverting, not trending)",
    "",
    "Hedge ratio: OLS, calibration window only.",
    "33 pairs tested (noise-adjusted freq > 15%).",
], font_size=15)

# Right - Results
headers = ["Result", "Count"]
rows = [
    ["Passed (3/3)", "4"],
    ["Near-miss (2/3)", "22"],
    ["Failed (0\u20131/3)", "7"],
    ["Total tested", "33"],
]
add_table(slide, 8, 1.5, 4, 2.8, headers, rows, col_widths=[2.5, 1.5])

# Bottom insight
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.5), Inches(5.2),
                                Inches(10), Inches(1.8))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(230, 240, 250)
shape.line.color.rgb = QUEENS_BLUE
tf = shape.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "26 of 33 pairs show mean-reverting spreads (\u2265 2 criteria met)"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = QUEENS_BLUE
p.font.name = 'Calibri'
p.alignment = PP_ALIGN.CENTER
p2 = tf.add_paragraph()
p2.text = "The 22 near-misses typically pass Hurst + half-life but fail cointegration \u2014 consistent with transient dynamics: relationships that form and dissolve over time."
p2.font.size = Pt(14)
p2.font.color.rgb = DARK_GRAY
p2.font.name = 'Calibri'
p2.alignment = PP_ALIGN.CENTER


# ============================================================================
# SLIDE 13: Section - OOS
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
section_header(slide, "Out-of-Sample Results", "Honest backtesting on unseen data")


# ============================================================================
# SLIDE 14: OOS Backtest Results
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 10, 0.8, "Out-of-Sample Backtest Results",
         font_size=28, bold=True, color=QUEENS_BLUE)

text_box(slide, 0.8, 1.1, 11, 0.5,
         "Z-score mean reversion: lookback 20d, enter |z| \u2265 2.0, exit |z| \u2264 0.5. OOS window only (~3.5 months).",
         font_size=14, color=DARK_GRAY)

headers = ["Pair", "Coint p", "Half-Life", "Hurst", "Trades", "P&L", "Sharpe", "Win %"]
rows = [
    ["NXPI\u2013ON", "0.271", "8.6d", "0.413", "2", "6.56", "3.55", "100%"],
    ["CDNS\u2013SNPS", "0.620", "21.5d", "0.447", "3", "36.26", "2.08", "100%"],
    ["QCOM\u2013QRVO", "0.019", "5.3d", "0.205", "2", "47.46", "1.44", "100%"],
    ["KLAC\u2013TSM", "0.006", "3.6d", "0.181", "2", "47.55", "1.35", "100%"],
    ["LRCX\u2013NVMI", "0.422", "13.2d", "0.440", "2", "16.19", "1.04", "100%"],
    ["NXPI\u2013POWI", "0.568", "13.5d", "0.372", "5", "37.32", "0.85", "80%"],
    ["MCHP\u2013ON", "0.039", "5.2d", "0.313", "3", "3.62", "0.65", "67%"],
]
add_table(slide, 0.8, 1.7, 8.5, 4.2, headers, rows,
          col_widths=[1.6, 0.9, 1.0, 0.9, 0.8, 0.9, 0.9, 0.8])

# Side summary
text_box(slide, 9.8, 1.8, 3, 0.5, "Summary", font_size=20, bold=True, color=QUEENS_BLUE)
bullet_slide(slide, 9.8, 2.5, 3.2, 4, [
    "14 of 26 tradeable pairs profitable OOS (54%)",
    "",
    "Top Sharpe ratios: 1.4\u20133.6 on unseen data",
    "",
    "Caveat: 2\u20135 trades per pair \u2014 directionally encouraging but low sample size",
], font_size=14, color=DARK_GRAY)


# ============================================================================
# SLIDE 15: Look-Ahead Bias
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 10, 0.8, "The Look-Ahead Bias Lesson",
         font_size=28, bold=True, color=QUEENS_BLUE)

# Before box
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5),
                                Inches(5.5), Inches(2.8))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(255, 235, 235)
shape.line.color.rgb = QUEENS_RED
text_box(slide, 1.1, 1.6, 5, 0.5, "Before: Full-Period Analysis", font_size=20, bold=True, color=QUEENS_RED)
bullet_slide(slide, 1.1, 2.2, 5, 2, [
    "Hedge ratio fit on ALL data",
    "Backtest on SAME data",
    "23/23 pairs profitable",
    "80\u2013100% win rates",
], font_size=16, color=DARK_GRAY)

# After box
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7), Inches(1.5),
                                Inches(5.5), Inches(2.8))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(230, 250, 235)
shape.line.color.rgb = GREEN
text_box(slide, 7.3, 1.6, 5, 0.5, "After: Proper Cal/OOS Split", font_size=20, bold=True, color=GREEN)
bullet_slide(slide, 7.3, 2.2, 5, 2, [
    "Hedge ratio on FIRST 67%",
    "Backtest on LAST 33%",
    "14/26 pairs profitable",
    "Some strong pairs reversed",
], font_size=16, color=DARK_GRAY)

# Example
text_box(slide, 1.5, 4.8, 10, 0.8,
         "ADI\u2013NXPI:  In-sample Sharpe 1.08  \u2192  OOS Sharpe \u22121.76",
         font_size=22, bold=True, color=DARK_GRAY, alignment=PP_ALIGN.CENTER)

shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(3), Inches(5.8),
                                Inches(7), Inches(1))
shape.fill.solid()
shape.fill.fore_color.rgb = QUEENS_BLUE
shape.line.color.rgb = QUEENS_BLUE
tf = shape.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Out-of-sample validation is non-negotiable."
p.font.size = Pt(20)
p.font.bold = True
p.font.color.rgb = WHITE
p.font.name = 'Calibri'
p.alignment = PP_ALIGN.CENTER


# ============================================================================
# SLIDE 16: Consensus vs Performance
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 10, 0.8, "Does Consensus Predict Performance?",
         font_size=28, bold=True, color=QUEENS_BLUE)

headers = ["Group", "Tradeable Pairs", "Avg Sharpe (OOS)", "Avg P&L (OOS)"]
rows = [
    ["Consensus (all 3 algos)", "6", "\u22120.251", "\u221213.73"],
    ["Non-consensus", "20", "+0.056", "\u22128.76"],
]
add_table(slide, 2, 1.6, 9, 1.8, headers, rows, col_widths=[3.0, 2.0, 2.0, 2.0])

text_box(slide, 1.5, 3.8, 10, 0.5, "Consensus pairs underperformed non-consensus OOS.",
         font_size=20, bold=True, color=QUEENS_RED, alignment=PP_ALIGN.CENTER)

# Insight boxes
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.5), Inches(4.6),
                                Inches(4.8), Inches(2))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(230, 240, 250)
shape.line.color.rgb = QUEENS_BLUE
tf = shape.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Consensus = Validation of Relationship"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = QUEENS_BLUE
p.alignment = PP_ALIGN.CENTER
p2 = tf.add_paragraph()
p2.text = "Cross-algorithm agreement confirms structurally related pairs"
p2.font.size = Pt(13)
p2.font.color.rgb = DARK_GRAY
p2.alignment = PP_ALIGN.CENTER

shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7), Inches(4.6),
                                Inches(4.8), Inches(2))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(255, 235, 235)
shape.line.color.rgb = QUEENS_RED
tf = shape.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Consensus \u2260 Trading Signal"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = QUEENS_RED
p.alignment = PP_ALIGN.CENTER
p2 = tf.add_paragraph()
p2.text = "Does not predict out-of-sample profitability"
p2.font.size = Pt(13)
p2.font.color.rgb = DARK_GRAY
p2.alignment = PP_ALIGN.CENTER


# ============================================================================
# SLIDE 17: Section - Insights
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
section_header(slide, "Key Insights", "What we learned")


# ============================================================================
# SLIDE 18: What We Learned
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 10, 0.8, "What We Learned",
         font_size=28, bold=True, color=QUEENS_BLUE)

# What works
text_box(slide, 0.8, 1.3, 5.5, 0.5, "What Works", font_size=20, bold=True, color=GREEN)
bullet_slide(slide, 0.8, 1.9, 5.8, 3.5, [
    "Clustering reliably discovers real subsector relationships",
    "3 algorithms converge \u2192 robust, algorithm-agnostic signal",
    "26/33 pairs show mean-reverting spreads",
    "Top OOS performers: genuine Sharpe 1.4\u20133.6",
    "Permutation test confirms statistical significance (11 pairs)",
], font_size=15, color=DARK_GRAY)

# What doesn't
text_box(slide, 7, 1.3, 5.5, 0.5, "Limitations", font_size=20, bold=True, color=QUEENS_RED)
bullet_slide(slide, 7, 1.9, 5.8, 3.5, [
    "Only 54% profitable OOS with simple z-score strategy",
    "Consensus does not predict OOS performance",
    "Fixed OLS hedge ratios drift over time",
    "Low trade counts (2\u20135) limit statistical power",
    "Zero transaction cost assumption",
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
p.text = "Clustering is an effective pair discovery tool.\nTurning discoveries into profitable trades requires strategy refinement."
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = WHITE
p.font.name = 'Calibri'
p.alignment = PP_ALIGN.CENTER


# ============================================================================
# SLIDE 19: Future Work
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
text_box(slide, 0.8, 0.4, 10, 0.8, "Future Work",
         font_size=28, bold=True, color=QUEENS_BLUE)

items = [
    ("Walk-Forward Validation", "Rolling cal/OOS splits for more robust performance estimates"),
    ("Adaptive Hedge Ratios", "Kalman filter estimates that evolve with the data"),
    ("Real-Time Pipeline", "Live formation detection \u2192 validation \u2192 signal generation"),
    ("Expanded Universe", "Test framework on other sectors to assess generalizability"),
    ("Transaction Costs", "Incorporate realistic spread, slippage, and commission"),
]

for i, (title, desc) in enumerate(items):
    y = 1.4 + i * 1.05
    text_box(slide, 1.2, y, 3.5, 0.5, f"{i+1}.  {title}", font_size=18, bold=True, color=QUEENS_BLUE)
    text_box(slide, 5, y, 7.5, 0.5, desc, font_size=16, color=DARK_GRAY)


# ============================================================================
# SLIDE 20: Thank You
# ============================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, QUEENS_BLUE)

text_box(slide, 1.5, 2.0, 10, 1.5, "Thank You",
         font_size=48, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

text_box(slide, 1.5, 3.8, 10, 1, "Questions?",
         font_size=28, color=RGBColor(180, 200, 220), alignment=PP_ALIGN.CENTER)

text_box(slide, 1.5, 5.5, 10, 0.6, "Queen's Quant Club",  # TODO: Add names
         font_size=18, color=RGBColor(250, 189, 0), alignment=PP_ALIGN.CENTER)


# ============================================================================
# SAVE
# ============================================================================
import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presentation.pptx')
prs.save(out_path)
print(f"Saved: {out_path}")
print(f"Slides: {len(prs.slides)}")
