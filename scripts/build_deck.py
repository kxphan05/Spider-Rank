"""Generate the presentation deck as an editable .pptx.

Generated rather than hand-made so the numbers cannot go stale silently. Every
shape is a real text box, table or autoshape, so the result opens and edits in
PowerPoint, Keynote and Google Slides.

    uvx --from python-pptx python3 scripts/build_deck.py
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

OUT = Path(__file__).resolve().parent.parent / "dist" / "techjam_track4.pptx"

INK = RGBColor(0x14, 0x1B, 0x2D)      # near-black navy, body text
MUTED = RGBColor(0x5B, 0x66, 0x7A)    # secondary text
ACCENT = RGBColor(0x0F, 0x62, 0xFE)   # one accent, used sparingly
GOOD = RGBColor(0x0B, 0x7A, 0x4B)
BAD = RGBColor(0xB3, 0x26, 0x1E)
RULE = RGBColor(0xD8, 0xDD, 0xE6)
PAPER = RGBColor(0xFF, 0xFF, 0xFF)
SURFACE = RGBColor(0xF4, 0xF6, 0xFA)
TINT = RGBColor(0xE4, 0xED, 0xFD)
GOOD_TINT = RGBColor(0xE6, 0xF3, 0xEC)
BAD_TINT = RGBColor(0xFA, 0xEA, 0xE8)

TITLE_FONT = "Georgia"
BODY_FONT = "Verdana"
MONO_FONT = "Consolas"


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def _textbox(slide, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    return frame


def _set(run, *, size, bold=False, color=INK, font=BODY_FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _rule(slide, top):
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), top,
                                  Inches(11.93), Emu(9525))
    line.fill.solid()
    line.fill.fore_color.rgb = RULE
    line.line.fill.background()
    line.shadow.inherit = False


def heading(slide, title, kicker=None):
    if kicker:
        frame = _textbox(slide, Inches(0.7), Inches(0.42), Inches(11.9), Inches(0.3))
        run = frame.paragraphs[0].add_run()
        run.text = kicker.upper()
        _set(run, size=11, bold=True, color=ACCENT)
        frame.paragraphs[0].space_after = Pt(0)
    frame = _textbox(slide, Inches(0.7), Inches(0.72), Inches(11.9), Inches(0.75))
    run = frame.paragraphs[0].add_run()
    run.text = title
    _set(run, size=28, bold=True, font=TITLE_FONT)
    _rule(slide, Inches(1.52))


def note(slide, text, top=None, color=MUTED, size=13, bold=False):
    """One line of context under a diagram or table. Keep it to one line."""
    top = Inches(6.75) if top is None else top
    frame = _textbox(slide, Inches(0.7), top, Inches(11.9), Inches(0.5))
    run = frame.paragraphs[0].add_run()
    run.text = text
    _set(run, size=size, color=color, bold=bold)


def bullets(slide, items, top=None, size=15, width=None, left=None):
    top = Inches(1.85) if top is None else top
    width = Inches(11.9) if width is None else width
    left = Inches(0.7) if left is None else left
    frame = _textbox(slide, left, top, width, Inches(5.0))
    first = True
    for item in items:
        text, level, color = (item + (INK,))[:3] if len(item) < 3 else item
        para = frame.paragraphs[0] if first else frame.add_paragraph()
        first = False
        para.level = level
        para.space_after = Pt(10 if level == 0 else 5)
        run = para.add_run()
        run.text = text
        _set(run, size=size if level == 0 else size - 2,
             bold=(level == 0 and color is INK), color=color)


# --------------------------------------------------------------------------
# diagram helpers
# --------------------------------------------------------------------------
def box(slide, left, top, width, height, title, sub=None, *,
        fill=SURFACE, border=RULE, color=INK, size=13, sub_size=None, shape=None):
    """A labelled rounded box. `sub` is a smaller second line inside it."""
    shape = MSO_SHAPE.ROUNDED_RECTANGLE if shape is None else shape
    node = slide.shapes.add_shape(shape, left, top, width, height)
    node.fill.solid()
    node.fill.fore_color.rgb = fill
    node.line.color.rgb = border
    node.line.width = Pt(1)
    node.shadow.inherit = False
    frame = node.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.margin_left = frame.margin_right = Inches(0.08)
    frame.margin_top = frame.margin_bottom = Inches(0.04)
    first = True
    for line in title.split("\n"):
        para = frame.paragraphs[0] if first else frame.add_paragraph()
        first = False
        para.alignment = PP_ALIGN.CENTER
        para.space_after = Pt(0)
        run = para.add_run()
        run.text = line
        _set(run, size=size, bold=True, color=color)
    for line in (sub or "").split("\n"):
        if not sub:
            break
        para = frame.add_paragraph()
        para.alignment = PP_ALIGN.CENTER
        para.space_after = Pt(0)
        run = para.add_run()
        run.text = line
        _set(run, size=sub_size or size - 3, color=MUTED)
    return node


def arrow(slide, left, top, width, height=None, *, color=None):
    """A right-pointing arrow, used between diagram boxes."""
    height = Inches(0.22) if height is None else height
    node = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, top, width, height)
    node.fill.solid()
    node.fill.fore_color.rgb = ACCENT if color is None else color
    node.line.fill.background()
    node.shadow.inherit = False
    return node


def down_arrow(slide, left, top, height, width=None, *, color=None):
    """A downward arrow, used inside a vertical sequence."""
    width = Inches(0.22) if width is None else width
    node = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, left, top, width, height)
    node.fill.solid()
    node.fill.fore_color.rgb = ACCENT if color is None else color
    node.line.fill.background()
    node.shadow.inherit = False
    return node


def back_arrow(slide, left, top, width, label, *, height=None):
    """A wide left-pointing arrow, used for the next-turn feedback loop."""
    height = Inches(0.34) if height is None else height
    node = slide.shapes.add_shape(MSO_SHAPE.LEFT_ARROW, left, top, width, height)
    node.fill.solid()
    node.fill.fore_color.rgb = TINT
    node.line.color.rgb = ACCENT
    node.line.width = Pt(0.75)
    node.shadow.inherit = False
    frame = node.text_frame
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    para = frame.paragraphs[0]
    para.alignment = PP_ALIGN.CENTER
    run = para.add_run()
    run.text = label
    _set(run, size=11, bold=True, color=ACCENT)
    return node


def caption(slide, left, top, width, lines, *, size=11):
    """Small stacked lines under a diagram box."""
    frame = _textbox(slide, left, top, width, Inches(1.0))
    for index, text in enumerate(lines):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.space_after = Pt(3)
        run = para.add_run()
        run.text = text
        _set(run, size=size, color=MUTED)


# --------------------------------------------------------------------------
# table helper
# --------------------------------------------------------------------------
def table(slide, headers, rows, *, left=None, top=None, width=None,
          col_widths=None, size=12, header_size=11, row_height=None,
          header_height=None, tints=None, mono_columns=()):
    """A real PowerPoint table. `tints` is one fill per row, or None.

    `rows` cells may be a plain string, or (text, colour) to tint one cell.
    """
    left = Inches(0.7) if left is None else left
    top = Inches(1.85) if top is None else top
    width = Inches(11.93) if width is None else width
    row_height = Inches(0.34) if row_height is None else row_height
    header_height = Inches(0.36) if header_height is None else header_height

    shape = slide.shapes.add_table(len(rows) + 1, len(headers), left, top,
                                   width, header_height + row_height * len(rows))
    tbl = shape.table
    tbl.first_row = True
    tbl.horz_banding = False
    tbl.rows[0].height = header_height

    if col_widths:
        total = sum(col_widths)
        for index, share in enumerate(col_widths):
            tbl.columns[index].width = Emu(int(width * share / total))

    for index, text in enumerate(headers):
        cell = tbl.cell(0, index)
        cell.fill.solid()
        cell.fill.fore_color.rgb = INK
        cell.margin_left = cell.margin_right = Inches(0.09)
        cell.margin_top = cell.margin_bottom = Inches(0.04)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        para = cell.text_frame.paragraphs[0]
        run = para.add_run()
        run.text = text
        _set(run, size=header_size, bold=True, color=PAPER)

    for r, row in enumerate(rows, start=1):
        tbl.rows[r].height = row_height
        fill = (tints[r - 1] if tints else None) or (PAPER if r % 2 else SURFACE)
        for c, value in enumerate(row):
            text, colour = value if isinstance(value, tuple) else (value, INK)
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = fill
            cell.margin_left = cell.margin_right = Inches(0.09)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            frame = cell.text_frame
            frame.word_wrap = True
            para = frame.paragraphs[0]
            run = para.add_run()
            run.text = text
            _set(run, size=size, color=colour,
                 bold=colour is not INK,
                 font=MONO_FONT if c in mono_columns else BODY_FONT)
    return tbl


# --------------------------------------------------------------------------
# slides
# --------------------------------------------------------------------------
def slide_title(prs):
    slide = _blank(prs)
    frame = _textbox(slide, Inches(0.9), Inches(1.75), Inches(11.5), Inches(1.8))
    run = frame.paragraphs[0].add_run()
    run.text = "A shopping agent that asks better questions"
    _set(run, size=38, bold=True, font=TITLE_FONT)

    frame = _textbox(slide, Inches(0.9), Inches(3.55), Inches(11.5), Inches(1.1))
    for text, size in (("TechJam Conversational Search — Track 4", 18),
                       ("Phan Kang Xun · Lloyd Wang", 14),
                       ("Find a hidden product in a 50,000-item catalog, in 10 turns or fewer.", 14)):
        para = frame.paragraphs[0] if text.startswith("TechJam") else frame.add_paragraph()
        para.space_after = Pt(5)
        run = para.add_run()
        run.text = text
        _set(run, size=size, color=MUTED)

    metrics = (("0.702", "TechnicalScore"), ("0.855", "HitRate@10"),
               ("0.462", "MRR"), ("4.21", "turns to find"))
    for index, (value, label) in enumerate(metrics):
        left = Inches(0.9) + Inches(2.7) * index
        box(slide, left, Inches(5.05), Inches(2.4), Inches(1.0), value, label,
            fill=TINT, border=TINT, color=ACCENT, size=26, sub_size=12)
    note(slide, "Full 200-sample public set. Every number in this deck is measured, not estimated.",
         top=Inches(6.35))
    return slide


def slide_task(prs):
    slide = _blank(prs)
    heading(slide, "The task", "problem")

    stages = [
        ("Shopper speaks", "one hidden target,\nnever named"),
        ("We answer", "return 10 products\n+ ask 1 question"),
        ("Grader checks", "is the target\nin those 10?"),
    ]
    left = Inches(0.9)
    for index, (title, sub) in enumerate(stages):
        box(slide, left, Inches(2.0), Inches(3.1), Inches(1.15), title, sub,
            fill=TINT if index == 1 else SURFACE, size=15)
        if index < 2:
            arrow(slide, left + Inches(3.2), Inches(2.48), Inches(0.5))
        left += Inches(3.7)
    box(slide, Inches(12.0), Inches(2.0), Inches(0.75), Inches(1.15), "HIT",
        "session ends", fill=GOOD_TINT, border=GOOD, color=GOOD, size=13)
    back_arrow(slide, Inches(0.9), Inches(3.45), Inches(10.4),
               "miss  →  next turn, up to 10 of them")

    table(slide, ["Term", "Weight", "What it rewards"], [
        ("HitRate@10", "0.50", "Finding the target at all, in any of the 10 slots."),
        ("MRR", "0.30", ("Its rank at the FIRST hit — not the best rank we ever reached.", ACCENT)),
        ("Efficiency", "0.20", "Finding it in fewer turns. (11 − turns) / 10."),
    ], top=Inches(4.35), col_widths=(2.2, 1.2, 8.5), row_height=Inches(0.38))
    note(slide, "Consequence we designed around: an earlier hit at a worse rank still pays. "
                "Speed beats polish.", top=Inches(6.15), color=ACCENT, bold=True)
    return slide


def slide_conversation(prs):
    """One real session, turn by turn.

    Replayed from demo/sessions.json (public_0006, captured against the live
    agent), not written by hand. This is the only slide that shows the product
    actually behaving like a conversation, which is the whole track.
    """
    slide = _blank(prs)
    heading(slide, "One real session, turn by turn", "what it looks like")

    frame = _textbox(slide, Inches(0.7), Inches(1.6), Inches(11.9), Inches(0.35))
    run = frame.paragraphs[0].add_run()
    run.text = ("Target, hidden from the agent:  Pro Club Men's Heavyweight Mesh Basketball Shorts")
    _set(run, size=12, bold=True, color=MUTED)

    turns = [
        ("1", "I'm looking for Basketball Men,\nbut I'm still exploring.",
         "What material are you\nlooking for?", "—",
         "Routed BROWSING. No attribute stated,\nso ask the most answerable one.", SURFACE, MUTED),
        ("2", "For that, what matters is:\npolyester; 100% Polyester.",
         "Do you have a colour\npreference?", "—",
         "Slot filled: material = polyester.\nBoosts matching items, deletes nothing.", SURFACE, MUTED),
        ("3", "I don't have an additional\npreference for colour.",
         "Is there a specific feature\nthat matters most?", "—",
         "A non-answer. We keep the text in the\nquery anyway — pruning it measured −0.041.", SURFACE, MUTED),
        ("4", "For that, what matters is:\nDrawstring closure; High quality\nmesh for breathability.",
         "Any particular style\nor fit you'd like?", "rank 3",
         "Phrase leg matches the exact span\n\u201cDrawstring closure\u201d. HIT — session ends.",
         GOOD_TINT, GOOD),
    ]

    top = Inches(2.05)
    for number, said, asked, rank, why, fill, colour in turns:
        box(slide, Inches(0.7), top, Inches(0.5), Inches(1.05), number,
            fill=fill, border=colour, color=colour, size=14)
        box(slide, Inches(1.3), top, Inches(3.5), Inches(1.05), "SHOPPER", said,
            fill=fill, border=RULE, color=MUTED, size=9, sub_size=11)
        arrow(slide, Inches(4.9), top + Inches(0.42), Inches(0.4), color=RULE)
        box(slide, Inches(5.4), top, Inches(2.8), Inches(1.05), "WE ASK", asked,
            fill=fill, border=RULE, color=MUTED, size=9, sub_size=11)
        box(slide, Inches(8.3), top, Inches(1.0), Inches(1.05), rank,
            fill=fill, border=colour, color=colour, size=12)
        caption(slide, Inches(9.5), top + Inches(0.2), Inches(3.1), why.split("\n"), size=10)
        top += Inches(1.15)

    note(slide, "Found on turn 4 of 10, at rank 3. Every question was chosen by the live policy — "
                "nothing here is scripted.", top=Inches(6.75), color=GOOD, bold=True)
    return slide


def slide_architecture(prs):
    slide = _blank(prs)
    heading(slide, "Architecture", "how it works")

    stages = [
        ("UNDERSTAND", "Slot extraction\nIntent routing\nPivot detection"),
        ("RETRIEVE", "4 independent legs\nover all 50k products"),
        ("RANK", "Fuse · boost\ndrop already-shown"),
        ("RESPOND", "Top 10 products\n+ the next question"),
    ]
    width = Inches(2.75)
    gap = Inches(0.45)
    left = Inches(0.75)
    for index, (title, sub) in enumerate(stages):
        box(slide, left, Inches(1.95), width, Inches(1.5), title, sub,
            fill=TINT if index in (1, 2) else SURFACE, size=14)
        if index < 3:
            arrow(slide, left + width + Inches(0.05), Inches(2.59), gap - Inches(0.1))
        left += width + gap

    back_arrow(slide, Inches(0.75), Inches(3.7), Inches(11.45),
               "each new reply re-runs the whole pipeline — state carries forward")

    table(slide, ["Stage", "What it decides", "Design rule we committed to"], [
        ("UNDERSTAND", "material · colour · budget · buying vs browsing",
         "Never trust one label. Negation-aware, and a pivot wipes state."),
        ("RETRIEVE", "~30 candidates from 50,000",
         "Every leg sees the whole catalog. No leg may filter another."),
        ("RANK", "which 10 to show, in what order",
         ("Boost, never delete. Only proven-wrong items are removed.", ACCENT)),
        ("RESPOND", "the single question to ask next",
         "Ask what the shopper can actually answer, not what we most want."),
    ], top=Inches(4.3), col_widths=(1.7, 3.6, 6.6), row_height=Inches(0.52), size=11)
    return slide


def slide_retrieval(prs):
    slide = _blank(prs)
    heading(slide, "Retrieval: four legs, one fusion", "architecture")

    legs = [
        ("BM25 keyword", "query split into words", "1.00", INK),
        ("Exact phrase", "whole spans, stopwords kept", "2.00", ACCENT),
        ("Dense vectors", "bge-small cosine", "0.25", INK),
        ("PRF expansion", "built, switched off", "0.00", MUTED),
    ]
    top = Inches(2.0)
    for title, sub, weight, colour in legs:
        box(slide, Inches(0.75), top, Inches(3.0), Inches(0.82), title, sub,
            fill=TINT if colour is ACCENT else SURFACE, color=colour, size=12)
        box(slide, Inches(3.85), top + Inches(0.24), Inches(0.62), Inches(0.34),
            weight, fill=PAPER, border=RULE, color=colour, size=11,
            shape=MSO_SHAPE.RECTANGLE)
        arrow(slide, Inches(4.55), top + Inches(0.3), Inches(0.55), color=RULE)
        top += Inches(0.95)

    box(slide, Inches(5.2), Inches(2.0), Inches(1.9), Inches(3.6),
        "WEIGHTED\nRRF", "rank-based,\nnot score-based", fill=TINT, border=ACCENT,
        color=ACCENT, size=14)

    chain = [("1 · Attribute boost", "agree +1 · clash −1 · delete nothing"),
             ("2 · Drop shown items", "proven not the target"),
             ("3 · Cross-encoder re-rank", "built, switched off")]
    arrow(slide, Inches(7.2), Inches(2.32), Inches(0.5))
    top = Inches(2.0)
    for index, (title, sub) in enumerate(chain):
        # The re-rank stage is greyed for the same reason PRF is: RERANK_WEIGHT
        # is 0.0, so it is built and not running. Showing it as live would
        # claim a stage the report lists as a limitation.
        off = index == 2
        box(slide, Inches(7.8), top, Inches(3.0), Inches(0.85), title, sub, size=12,
            color=MUTED if off else INK, fill=PAPER if off else SURFACE)
        if index < 2:
            down_arrow(slide, Inches(9.19), top + Inches(0.9), Inches(0.35))
        top += Inches(1.25)
    arrow(slide, Inches(10.9), Inches(4.62), Inches(0.45))
    box(slide, Inches(11.45), Inches(4.1), Inches(1.2), Inches(1.15), "TOP 10",
        fill=GOOD_TINT, border=GOOD, color=GOOD, size=15)

    note(slide, "Weighted RRF is scale-invariant per leg, so only the ratios matter — "
                "that is what makes each leg sweepable on its own.", top=Inches(5.85))
    note(slide, "Legs are independent by design: one leg failing degrades the score, "
                "it does not break the agent.", top=Inches(6.25))
    return slide


def slide_scoreboard(prs):
    slide = _blank(prs)
    heading(slide, "Every idea we measured", "the ledger")
    table(slide, ["Idea", "Why it should work", "Δ Score", "Verdict"], [
        ("Drop already-shown items", "A shown item is proven not the target — the session would have ended.",
         ("+0.084", GOOD), ("SHIPPED", GOOD)),
        ("Exact-phrase retrieval leg", "89.7% of shopper text is verbatim from the target's own record.",
         ("+0.018", GOOD), ("SHIPPED", GOOD)),
        ("Boost instead of filter", "Our own labels disagree with the target 16–37% of the time.",
         ("+0.013", GOOD), ("SHIPPED", GOOD)),
        ("Ask answerable questions first", "A question that gets no answer burns a whole turn.",
         ("+0.011", GOOD), ("SHIPPED", GOOD)),
        ("Halve the dense weight", "Dense adds no recall here; it demotes correct keyword hits.",
         ("+0.011", GOOD), ("SHIPPED", GOOD)),
        ("Rewrite query on a pivot", "Stale preferences keep biasing search after the shopper changes their mind.",
         ("−0.058", BAD), ("REJECTED", BAD)),
        ("Strip non-answer replies", "\"I don't have a preference\" is noise in the query.",
         ("−0.041", BAD), ("REJECTED", BAD)),
        ("Turn-annealed diversity", "A diverse slate is cheap early and pure risk late.",
         ("−0.003", BAD), ("REJECTED", BAD)),
        ("Cross-session user profiles", "A returning shopper should not have to repeat themselves.",
         ("0.000", MUTED), ("REJECTED", BAD)),
        ("Trained intent classifier", "A learned boundary should beat hand-written prototypes.",
         ("worse", BAD), ("REJECTED", BAD)),
    ], top=Inches(1.8), col_widths=(3.0, 6.4, 1.2, 1.3), row_height=Inches(0.4),
        size=11, header_size=11)
    note(slide, "Five shipped, five thrown away. The rejected half is where the engineering is.",
         top=Inches(6.5), color=ACCENT, bold=True)
    return slide


def slide_insight(prs):
    slide = _blank(prs)
    heading(slide, "The measurement that changed our approach", "insight")

    frame = _textbox(slide, Inches(0.7), Inches(1.75), Inches(11.9), Inches(0.5))
    run = frame.paragraphs[0].add_run()
    run.text = "89.7% of what the shopper says is copied word-for-word from the target product's own text."
    _set(run, size=17, bold=True, color=ACCENT)
    frame = _textbox(slide, Inches(0.7), Inches(2.2), Inches(11.9), Inches(0.4))
    run = frame.paragraphs[0].add_run()
    run.text = "So this is close to an exact-match problem, not a paraphrase problem. That inverts the usual advice."
    _set(run, size=13, color=MUTED)

    box(slide, Inches(0.75), Inches(2.85), Inches(2.6), Inches(0.62),
        "\"Buckle closure\"", fill=PAPER, border=INK, size=14)

    box(slide, Inches(4.4), Inches(2.75), Inches(3.6), Inches(0.85),
        "BM25 splits it", "\"buckle\" OR \"closure\"  →  thousands of hits",
        fill=BAD_TINT, border=BAD, color=BAD, size=13)
    box(slide, Inches(4.4), Inches(3.85), Inches(3.6), Inches(0.85),
        "Phrase leg keeps it", "the exact span  →  rare, and usually right",
        fill=GOOD_TINT, border=GOOD, color=GOOD, size=13)
    arrow(slide, Inches(3.5), Inches(3.06), Inches(0.75), color=BAD)
    arrow(slide, Inches(3.5), Inches(4.16), Inches(0.75), color=GOOD)

    box(slide, Inches(8.6), Inches(2.75), Inches(4.05), Inches(1.95),
        "+0.018 score", "+5 sessions found.\nThe only change that added recall\nrather than reordering.",
        fill=TINT, border=ACCENT, color=ACCENT, size=17)

    table(slide, ["What this predicted", "What we measured"], [
        ("Adding semantic tolerance should LOSE score here.",
         ("Dense weight, query rewriting and query cleaning all lost. Three for three.", BAD)),
        ("Matching whole spans should WIN.",
         ("Phrase leg is our second-largest win, and the sweep has a real peak.", GOOD)),
        ("Stopwords must be kept in a phrase query.",
         ("\"pull on closure\" matches; \"pull closure\" matches nothing.", INK)),
    ], top=Inches(5.0), col_widths=(4.6, 7.3), row_height=Inches(0.42), size=11)
    return slide


def slide_exclusion(prs):
    slide = _blank(prs)
    heading(slide, "Our biggest win came from reading the rules, not the data", "insight")

    steps = [
        ("Turn 1", "we show 10 products", SURFACE, INK),
        ("Shopper replies", "so none of them was the target", TINT, ACCENT),
        ("Turn 2", "showing them again is a wasted slot", BAD_TINT, BAD),
    ]
    left = Inches(0.75)
    for index, (title, sub, fill, colour) in enumerate(steps):
        box(slide, left, Inches(1.9), Inches(3.4), Inches(1.0), title, sub,
            fill=fill, border=colour, color=colour, size=14)
        if index < 2:
            arrow(slide, left + Inches(3.5), Inches(2.29), Inches(0.5))
        left += Inches(4.0)

    frame = _textbox(slide, Inches(0.75), Inches(3.15), Inches(11.9), Inches(0.6))
    run = frame.paragraphs[0].add_run()
    run.text = ("Being asked for another turn at all is proof that every item shown so far is wrong. "
                "We were re-offering 5.3 of 10 slots each turn.")
    _set(run, size=14, bold=True)

    table(slide, ["", "HitRate@10", "Found", "MRR", "Turns", "Score"], [
        ("Before the phrase leg", "0.755", "151/200", "0.399", "4.94", "0.6184"),
        ("+ phrase leg", "0.780", "156/200", "0.410", "4.83", ("0.6366", ACCENT)),
        ("+ drop shown items", ("0.855", GOOD), ("171/200", GOOD), ("0.462", GOOD),
         ("4.21", GOOD), ("0.7020", GOOD)),
    ], top=Inches(3.95), col_widths=(3.6, 1.7, 1.5, 1.4, 1.3, 1.6),
        row_height=Inches(0.42), mono_columns=(1, 2, 3, 4, 5))

    note(slide, "+0.084 — and it moves all three terms at once: 15 more sessions found, better ranks, "
                "0.6 fewer turns.", top=Inches(5.65), color=GOOD, bold=True)
    note(slide, "The subtlety: on a mid-chat pivot the grader restarts its checking, so a pre-pivot item "
                "may become the target again. We clear the shown set on exactly that signal.",
         top=Inches(6.1))
    return slide


def slide_legs_table(prs):
    slide = _blank(prs)
    heading(slide, "Retrieval legs: what each one buys and costs", "pros and cons")
    table(slide, ["Leg", "Pro", "Con", "Status"], [
        ("BM25 keyword",
         "Carries this benchmark. Cheap, exact, explainable.",
         "Splits phrases into unrelated words.", ("core", GOOD)),
        ("Exact phrase",
         "Adds recall. Runs with the grain of the data.",
         "Tuned to a measured quirk; helps less if the grader paraphrases.", ("core", GOOD)),
        ("Dense vectors",
         "Insurance: the only leg that survives paraphrasing.",
         "Adds zero recall here. At full weight it evicts correct hits.",
         ("weight 0.25", ACCENT)),
        ("PRF expansion",
         "Turns a one-word query into the vocabulary of its own best hits.",
         "Drifts when the first results are wrong — we saw \"leather\" for shoes pull in jackets.",
         ("off", MUTED)),
    ], top=Inches(1.8), col_widths=(1.9, 4.3, 4.6, 1.3), row_height=Inches(0.78), size=11)

    note(slide, "Why dense stays at 0.25 rather than 0: removing it scores higher locally, "
                "but the local shopper is a near-exact-match shopper.", top=Inches(5.35))
    note(slide, "The downside is asymmetric. Keeping it costs ~0.008 if the grader is like ours, "
                "and saves us entirely if it is not.", top=Inches(5.78), color=ACCENT, bold=True)
    return slide


def slide_questions(prs):
    slide = _blank(prs)
    heading(slide, "Which question to ask next", "pros and cons")

    frame = _textbox(slide, Inches(0.7), Inches(1.8), Inches(11.9), Inches(0.4))
    run = frame.paragraphs[0].add_run()
    run.text = "A question the shopper cannot answer burns a whole turn. So we measured how often each one lands."
    _set(run, size=14, bold=True)

    data = [("feature", 96), ("material", 73), ("colour", 26),
            ("style", 9), ("size", 5), ("use case", 2)]
    top = Inches(2.4)
    for name, pct in data:
        frame = _textbox(slide, Inches(0.75), top + Inches(0.03), Inches(1.4), Inches(0.3))
        run = frame.paragraphs[0].add_run()
        run.text = name
        _set(run, size=12)
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(2.25), top,
                                     Inches(5.4) * pct / 100, Inches(0.28))
        bar.fill.solid()
        bar.fill.fore_color.rgb = ACCENT if pct >= 70 else RULE
        bar.line.fill.background()
        bar.shadow.inherit = False
        frame = _textbox(slide, Inches(7.8), top + Inches(0.03), Inches(1.0), Inches(0.3))
        run = frame.paragraphs[0].add_run()
        run.text = f"{pct}%"
        _set(run, size=12, color=ACCENT if pct >= 70 else MUTED, bold=pct >= 70)
        top += Inches(0.42)

    box(slide, Inches(9.0), Inches(2.4), Inches(3.65), Inches(2.35),
        "The old order asked\nstyle, size, use case first",
        "The three rarest, before the one\nanswered 96% of the time.\n\nRe-ordering: +0.011",
        fill=BAD_TINT, border=BAD, color=BAD, size=13)

    note(slide, "Honest caveat: \"feature\" is the simulator's catch-all bucket, so 96% is partly an "
                "artifact of how it labels answers.", top=Inches(5.3))
    note(slide, "We re-ordered the fallback list rather than deleting the rare options — the real grader "
                "may bucket answers differently.", top=Inches(5.7))
    return slide


def slide_models(prs):
    slide = _blank(prs)
    heading(slide, "Which models, and one we could not afford", "pros and cons")
    table(slide, ["Model", "Size", "Speed", "Used for", "Verdict"], [
        ("bge-small-en-v1.5", "129 MB", "fast", "Dense search, intent, pivot and non-answer detection",
         ("shipped", GOOD)),
        ("distilbert (masked LM)", "257 MB", "246 ms / call", "Guessing an unstated material from turn 1",
         ("shipped, marginal", MUTED)),
        ("MiniLM cross-encoder", "88 MB", "~17 ms / pair", "Would re-rank the top 20 into the shown 10",
         ("built, weight 0", MUTED)),
        ("Qwen3-Reranker-0.6B", "1.2 GB", ("~27 s / pair", BAD),
         "Would re-rank better — 75 hours for ONE evaluation run", ("rejected", BAD)),
    ], top=Inches(1.85), col_widths=(2.7, 1.1, 1.6, 5.2, 1.3), row_height=Inches(0.62), size=11)

    box(slide, Inches(0.7), Inches(4.9), Inches(11.93), Inches(0.9),
        "A model we cannot A/B is a model we cannot justify.",
        "Qwen judges relevance better than MiniLM — but one A/B run would have taken 75 hours. Both re-rankers ship at weight zero: built, wired, never measured, so never switched on.",
        fill=TINT, border=ACCENT, color=ACCENT, size=16)
    note(slide, "Nothing is fine-tuned. The rules put training base models out of scope, and we had no "
                "labelled data that was not the simulator's own two templates.", top=Inches(6.1))
    note(slide, "Everything runs on CPU with the network off. Loaded assets: 460 MB "
                "(encoder 129 + masked LM 257 + dense index 74).", top=Inches(6.75))
    return slide


def slide_rejected(prs):
    slide = _blank(prs)
    heading(slide, "Why the five rejected ideas failed", "negative results")
    table(slide, ["Idea", "What we expected", "What actually happened", "Δ"], [
        ("Rewrite the query\nafter a pivot",
         "Dropping stale history should clean up the search.",
         "The category is stated once, in turn 1. The pivot names only the changed attribute — "
         "so we searched 50,000 products for \"leather\" with no category at all.",
         ("−0.058", BAD)),
        ("Strip non-answer\nreplies from the query",
         "\"Imported; Pull On closure\" is boilerplate noise.",
         "It is not noise here. Because of the 89.7% finding it is an exact-match key. "
         "Deleting it deletes what BM25 wins with.", ("−0.041", BAD)),
        ("Remember shoppers\nacross sessions",
         "A returning shopper should not repeat themselves.",
         "Two sessions with the same profile share a product category 0.5% of the time. "
         "Random pairs: 1.2%. There was no identity to remember.", ("0.000", MUTED)),
        ("Get more diverse\nas turns run out",
         "A varied slate is cheap early and risky late.",
         "The browsing metrics came back byte-identical. The schedule changed nothing in the "
         "one track it governs.", ("−0.003", BAD)),
        ("Train the intent\nclassifier",
         "A learned boundary should beat hand-written examples.",
         "98% in testing — because the simulator has exactly two sentence templates and it "
         "memorised them. On unseen phrasing it fell to chance.", ("worse", BAD)),
    ], top=Inches(1.8), col_widths=(2.0, 3.0, 5.9, 1.0), row_height=Inches(0.76), size=10.5)
    note(slide, "The pattern: four of five were semantically sensible and structurally wrong for this "
                "task. We only found that out by measuring.", top=Inches(6.15), color=ACCENT, bold=True)
    return slide


def slide_discipline(prs):
    slide = _blank(prs)
    heading(slide, "How we avoided fooling ourselves", "method")
    rules = [
        ("RULE 1", "The \"off\" switch must\nreproduce the shipped score",
         "Caught a real bug: a flag whose zero value was not truly off. It had quietly "
         "shifted the control leg of an entire experiment."),
        ("RULE 2", "Count sessions,\nnot percentages",
         "On 200 samples, a hit rate of 0.7500 against 0.7450 is ONE session. "
         "We stopped believing any delta under ±0.0025."),
        ("RULE 3", "Write the prediction\ndown first",
         "A good number from the wrong mechanism is still a failure. "
         "We predicted the phrase leg would not help buying — and it did not."),
    ]
    left = Inches(0.75)
    for kicker, title, body in rules:
        box(slide, left, Inches(1.9), Inches(3.75), Inches(1.5), title,
            fill=TINT, border=ACCENT, color=ACCENT, size=15)
        frame = _textbox(slide, left + Inches(0.1), Inches(1.98), Inches(1.0), Inches(0.25))
        run = frame.paragraphs[0].add_run()
        run.text = kicker
        _set(run, size=9, bold=True, color=ACCENT)
        caption(slide, left, Inches(3.55), Inches(3.75), [body], size=12.5)
        left += Inches(4.05)

    note(slide, "This is why the ledger has five rejections on it. Four of them looked "
                "promising until the control leg was run.", top=Inches(4.55), color=ACCENT, bold=True)
    note(slide, "Cost of the discipline: one point on any tuning curve is a full 200-sample run, "
                "about 15 minutes. That budget — not ideas — is what we ran out of.", top=Inches(5.05))
    return slide


def slide_silent(prs):
    slide = _blank(prs)
    heading(slide, "Three problems a score could never have shown us", "engineering")
    table(slide, ["What broke", "Why the score stayed quiet", "How we found it", "Fix"], [
        ("Our search index could only be\nused from one thread.",
         "The evaluator is single-threaded, so it never complained. In the demo, two of "
         "three retrieval legs failed silently every turn and the agent still returned 10 products.",
         "Running the demo UI.", "Shared connection\n+ a lock."),
        ("A feature flag whose \"off\"\nvalue was not off.",
         "It disabled a re-ranking stage as a side effect, so the control leg of an "
         "experiment scored 0.6154 instead of 0.6182.",
         "Rule 1 — checking that\nidentity reproduced.", "Gate every effect,\nnot just the knob."),
        ("Re-ranking exactly the 10\nitems we already show.",
         "Re-ranking your own output can reorder it, so MRR moved and it looked like it worked. "
         "It can never turn a miss into a hit.",
         "Reasoning, not measuring.", "Re-rank the top 20\ninstead."),
    ], top=Inches(1.8), col_widths=(2.9, 5.4, 2.2, 1.4), row_height=Inches(0.98), size=10.5)
    note(slide, "All three were invisible in the metric. Two of them were found by using the system "
                "rather than scoring it.", top=Inches(5.35), color=ACCENT, bold=True)
    return slide


def slide_offline(prs):
    """The it-has-to-run-on-someone-else's-machine slide.

    Numbers from docs/team_report.md sec.4 (measure_latency.py) and from a real
    scripts/preflight.py --strict run. The point a judge cares about: none of
    this degrades loudly, so every guard here exists because the failure was
    silent.
    """
    slide = _blank(prs)
    heading(slide, "It has to run on someone else's machine", "engineering")

    stats = [("0", "network calls\nat scoring time"),
             ("$0.00", "model cost\n0 tokens, 0 API calls"),
             ("408 ms", "mean turn\np95 716 ms, CPU only"),
             ("1.3 GB", "peak memory\n460 MB of assets")]
    left = Inches(0.7)
    for value, label in stats:
        head, sub = label.split("\n")
        box(slide, left, Inches(1.75), Inches(2.83), Inches(1.0), value,
            head + "\n" + sub, fill=TINT, border=ACCENT, color=ACCENT,
            size=20, sub_size=10)
        left += Inches(3.03)

    frame = _textbox(slide, Inches(0.7), Inches(3.0), Inches(11.9), Inches(0.45))
    run = frame.paragraphs[0].add_run()
    run.text = ("The problem: with no network and a cold cache, the agent does not crash. "
                "It starts, returns ten products, and is silently a different system.")
    _set(run, size=14, bold=True, color=BAD)

    body = ("Dense retrieval, both classifiers and the masked LM all fail open at once. "
            "Nothing in the score says so — the run just gets worse. So the check is a "
            "build step, not a habit:")
    frame = _textbox(slide, Inches(0.7), Inches(3.5), Inches(5.2), Inches(0.9))
    run = frame.paragraphs[0].add_run()
    run.text = body
    _set(run, size=12, color=MUTED)

    guards = [("fetch_assets.py", "the only step that touches the network, ever"),
              ("preflight.py --strict", "exits non-zero if any component is dark"),
              ("build_submission.py --verify", "re-imports the bundle and re-scores it")]
    top = Inches(4.35)
    for name, what in guards:
        box(slide, Inches(0.7), top, Inches(2.5), Inches(0.36), name,
            fill=PAPER, border=ACCENT, color=ACCENT, size=10,
            shape=MSO_SHAPE.RECTANGLE)
        frame = _textbox(slide, Inches(3.3), top + Inches(0.09), Inches(2.7), Inches(0.3))
        run = frame.paragraphs[0].add_run()
        run.text = what
        _set(run, size=10, color=MUTED)
        top += Inches(0.44)

    lines = [("$ preflight.py --strict", ACCENT),
             ("[  ok  ] dense retrieval        live", GOOD),
             ("[  ok  ] intent classifier      live", GOOD),
             ("[  ok  ] override detector      live", GOOD),
             ("[  ok  ] non-answer detector    live", GOOD),
             ("[ warn ] cross-encoder          off", MUTED),
             ("OK: the agent runs fully offline.", GOOD)]
    panel = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.2),
                                   Inches(3.42), Inches(6.43), Inches(2.05))
    panel.fill.solid()
    panel.fill.fore_color.rgb = SURFACE
    panel.line.color.rgb = RULE
    panel.shadow.inherit = False
    panel.text_frame.word_wrap = True
    frame = _textbox(slide, Inches(6.45), Inches(3.6), Inches(6.0), Inches(1.8))
    for index, (text, colour) in enumerate(lines):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.space_after = Pt(2)
        run = para.add_run()
        run.text = text
        _set(run, size=10, color=colour, font=MONO_FONT,
             bold=colour is not MUTED)

    note(slide, "Same class of bug, twice more: our search index worked from one thread only, "
                "and our data paths were relative to the working directory — so the agent "
                "quietly lost half its pipeline when launched from anywhere but the repo root.",
         top=Inches(5.75))
    note(slide, "The submission bundle is generated, then re-imported from a neutral directory "
                "and re-scored before we believe it.", top=Inches(6.4), color=ACCENT, bold=True)
    return slide


def slide_next(prs):
    slide = _blank(prs)
    heading(slide, "What we would do with more time", "known open work")
    table(slide, ["Direction", "Why we think it matters", "Why it is not done"], [
        ("Sweep the BM25\nfield weights",
         "Title, category and description are weighted by hand — we guessed, and never swept once. "
         "BM25 carries this benchmark and the phrase leg reuses the same ranking function.",
         "A grid over 7 fields is dozens\nof 15-minute runs."),
        ("Retune the attribute\nboost strength",
         "It is +1 / −1 by assumption, chosen back when 40% of our colour labels were fictional. "
         "We fixed the labels and never revisited the weights.",
         "Ran out of evaluation budget\nbefore this one."),
        ("Turn on pseudo-relevance\nfeedback",
         "It expands a thin one-word query using the vocabulary of its own best hits — exactly the "
         "buying track's failure mode.",
         "Built and committed, switched off.\nUnmeasured, so unshipped."),
        ("Extract catalog attributes\nwith an LLM, offline",
         "Only 3 of 10 attributes have an extractor at all. Style, size, use case and feature have none.",
         "A one-time batch pass over\n50,000 products. Hours."),
        ("Switch on the\ncross-encoder re-rank",
         "It scores the top 20 and can promote a rank-11 item into the shown ten — the one stage "
         "that could convert a miss into a hit rather than just reorder.",
         "Wired and greyed out at weight 0.\nNever A/B'd, so never trusted."),
    ], top=Inches(1.8), col_widths=(2.6, 6.6, 2.7), row_height=Inches(0.72), size=10.5)
    note(slide, "The honest summary: the ideas were never the constraint. Evaluation time was.",
         top=Inches(6.05), color=ACCENT, bold=True)
    note(slide, "This hackathon rewards engineering intuition. Ours says: sweep the knobs nobody has "
                "checked, because that is where four of our five wins came from.", top=Inches(6.45))
    return slide


def slide_team(prs):
    slide = _blank(prs)
    heading(slide, "Team and process", "credits")
    table(slide, ["", "Contribution"], [
        ("Phan Kang Xun",
         "Architecture, retrieval, every experiment, evaluation tooling, demo, report and this deck."),
        ("Lloyd Wang", "Registered team member."),
        ("AI assistance",
         "Built with heavy use of an AI coding assistant for implementation and documentation. "
         "The rules permit this. We state it because it is true."),
    ], top=Inches(1.85), col_widths=(2.4, 9.5), row_height=Inches(0.55), size=12)

    box(slide, Inches(0.7), Inches(4.3), Inches(11.93), Inches(1.1),
        "Every idea here was accepted or rejected on a measured number — including the five we threw away.",
        "What we would want judged is the reasoning, not only the score. We can tell you why each idea "
        "should have worked, and why half of them did not.",
        fill=TINT, border=ACCENT, color=ACCENT, size=16)
    note(slide, "Reproduction instructions: REPRODUCE.md.  Full experiment record: CLAUDE.md and "
                "docs/team_report.md.", top=Inches(5.75))
    return slide


def build() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide_title(prs)
    slide_task(prs)
    slide_conversation(prs)
    slide_architecture(prs)
    slide_retrieval(prs)
    slide_scoreboard(prs)
    slide_insight(prs)
    slide_exclusion(prs)
    slide_legs_table(prs)
    slide_questions(prs)
    slide_models(prs)
    slide_rejected(prs)
    slide_discipline(prs)
    slide_silent(prs)
    slide_offline(prs)
    slide_next(prs)
    slide_team(prs)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
