"""Generate the presentation deck as an editable .pptx.

Written as a script rather than a hand-made file for the same reason the
submission bundle is generated: the numbers change, and a deck that has to be
re-typed by hand goes stale silently. Edit this file and re-run, or edit the
generated .pptx directly -- every shape is a real text box, so PowerPoint,
Keynote and Google Slides can all open and change it.

    uvx --from python-pptx python3 scripts/build_deck.py
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

OUT = Path(__file__).resolve().parent.parent / "dist" / "techjam_track4.pptx"

INK = RGBColor(0x14, 0x1B, 0x2D)      # near-black navy, body text
MUTED = RGBColor(0x5B, 0x66, 0x7A)    # secondary text
ACCENT = RGBColor(0x0F, 0x62, 0xFE)   # one accent, used sparingly
GOOD = RGBColor(0x0B, 0x7A, 0x4B)
BAD = RGBColor(0xB3, 0x26, 0x1E)
RULE = RGBColor(0xD8, 0xDD, 0xE6)
PAPER = RGBColor(0xFF, 0xFF, 0xFF)

TITLE_FONT = "Georgia"
BODY_FONT = "Verdana"


def _textbox(slide, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    return frame


def _set(run, *, size, bold=False, color=INK, font=BODY_FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _rule(slide, top):
    line = slide.shapes.add_shape(1, Inches(0.7), top, Inches(11.93), Emu(9525))
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
    _set(run, size=30, bold=True, font=TITLE_FONT)
    _rule(slide, Inches(1.58))


def bullets(slide, items, top=None, size=16, width=None, left=None):
    top = Inches(1.9) if top is None else top
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


def two_col(slide, left_title, left_items, right_title, right_items, top=None):
    top = Inches(1.95) if top is None else top
    for offset, title, items, tint in (
        (Inches(0.7), left_title, left_items, GOOD),
        (Inches(6.85), right_title, right_items, BAD),
    ):
        frame = _textbox(slide, offset, top, Inches(5.6), Inches(0.4))
        run = frame.paragraphs[0].add_run()
        run.text = title
        _set(run, size=15, bold=True, color=tint)
        body = _textbox(slide, offset, top + Inches(0.5), Inches(5.6), Inches(4.4))
        for index, text in enumerate(items):
            para = body.paragraphs[0] if index == 0 else body.add_paragraph()
            para.space_after = Pt(9)
            run = para.add_run()
            run.text = text
            _set(run, size=14)


def build() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # ---- 1. title -------------------------------------------------------
    slide = _blank(prs)
    frame = _textbox(slide, Inches(0.9), Inches(2.3), Inches(11.5), Inches(1.4))
    run = frame.paragraphs[0].add_run()
    run.text = "A shopping agent that asks better questions"
    _set(run, size=42, bold=True, font=TITLE_FONT)
    frame = _textbox(slide, Inches(0.9), Inches(3.6), Inches(11.5), Inches(0.9))
    run = frame.paragraphs[0].add_run()
    run.text = "TechJam Conversational Search — Track 4"
    _set(run, size=19, color=MUTED)
    para = frame.add_paragraph()
    run = para.add_run()
    run.text = "Phan Kang Xun · Lloyd Wang"
    _set(run, size=15, color=MUTED)
    para = frame.add_paragraph()
    run = para.add_run()
    run.text = "Find a hidden product in a 50,000-item catalog, in 10 turns or fewer."
    _set(run, size=15, color=MUTED)
    frame = _textbox(slide, Inches(0.9), Inches(5.6), Inches(11.5), Inches(0.6))
    run = frame.paragraphs[0].add_run()
    run.text = "TechnicalScore 0.6454     HitRate@10 0.790     MTTC 4.75"
    _set(run, size=17, bold=True, color=ACCENT)

    # ---- 2. the task ----------------------------------------------------
    slide = _blank(prs)
    heading(slide, "The task", "problem")
    bullets(slide, [
        ("A simulated shopper has one product in mind. They do not name it.", 0),
        ("We get 10 turns. Each turn we return 10 products and may ask one question.", 0),
        ("The session ends the moment the target appears in our 10.", 0),
        ("Score = 0.50 x HitRate@10  +  0.30 x MRR  +  0.20 x Efficiency", 0, ACCENT),
        ("Efficiency rewards finding it in fewer turns.", 1),
        ("MRR uses the rank at the FIRST hit, not the best rank we ever reached.", 1),
        ("That last detail shapes everything. Finding it earlier but lower still pays.", 0),
    ])

    # ---- 3. result ------------------------------------------------------
    slide = _blank(prs)
    heading(slide, "Where we ended up", "result")
    bullets(slide, [
        ("Starter baseline:  HitRate 0.125    MRR 0.068    MTTC 9.81", 0, MUTED),
        ("Our agent:         HitRate 0.790    MRR 0.418    MTTC 4.75", 0),
        ("6.3x the hit rate. Half the turns.", 0, ACCENT),
        ("Measured on the full 200-sample public set. No number here is an estimate.", 0, MUTED),
        ("Everything runs locally on CPU. No API is called at inference time.", 0),
    ])

    # ---- 4. architecture ------------------------------------------------
    slide = _blank(prs)
    heading(slide, "How it works", "architecture")
    steps = [
        ("1. Read the message", "Pull out material, colour and budget. Handle negation."),
        ("2. Route the intent", "Buying or browsing. This changes weights, not the pipeline."),
        ("3. Retrieve on three legs", "Keyword BM25 + exact phrase + dense vectors."),
        ("4. Fuse the rankings", "Weighted reciprocal rank fusion."),
        ("5. Boost, never filter", "Known matches float up. Nothing is deleted."),
        ("6. Re-rank and ask", "Cross-encoder on the head, then pick the next question."),
    ]
    top = Inches(1.95)
    for index, (name, detail) in enumerate(steps):
        row = top + Inches(0.78) * index
        frame = _textbox(slide, Inches(0.7), row, Inches(3.5), Inches(0.4))
        run = frame.paragraphs[0].add_run()
        run.text = name
        _set(run, size=15, bold=True)
        frame = _textbox(slide, Inches(4.3), row, Inches(8.3), Inches(0.4))
        run = frame.paragraphs[0].add_run()
        run.text = detail
        _set(run, size=14, color=MUTED)

    # ---- 5. the key insight ---------------------------------------------
    slide = _blank(prs)
    heading(slide, "The one measurement that changed our approach", "insight")
    bullets(slide, [
        ("We checked how the simulated shopper actually talks.", 0),
        ("89.7% of what they say is copied word-for-word from the target product's own text.", 0, ACCENT),
        ("So this is close to an exact-match problem, not a paraphrase problem.", 0),
        ("Standard BM25 throws that away. It splits the query into separate words.", 0),
        ("\"Buckle closure\" becomes \"buckle\" OR \"closure\" against 50,000 products.", 1),
        ("We added a leg that matches the whole phrase instead.", 0),
        ("+0.0272 score. Our biggest single win, and the only one that found NEW products.", 0, GOOD),
    ])

    # ---- 6. phrase leg pros/cons ----------------------------------------
    slide = _blank(prs)
    heading(slide, "Engineering choice: the phrase leg", "pros and cons")
    two_col(
        slide,
        "Why it works", [
            "Matches the grain of the data instead of fighting it.",
            "Adds recall. It finds 8 products nothing else found.",
            "Costs nothing extra. It reuses the index we already build.",
            "The weight curve has a real peak, so the setting is not arbitrary.",
        ],
        "What it costs", [
            "It is tuned to a measured quirk of THIS benchmark.",
            "If the real grader paraphrases, the leg helps much less.",
            "Stopwords must be kept, which is the opposite of normal practice.",
            "\"pull on closure\" breaks if you drop \"on\".",
        ],
    )

    # ---- 7. boost not filter --------------------------------------------
    slide = _blank(prs)
    heading(slide, "Engineering choice: boost, never filter", "pros and cons")
    bullets(slide, [
        ("When the shopper says \"cotton\", the obvious move is to drop everything else.", 0),
        ("We measured that. Our own catalog labels disagree with the true target 16% of the time on material, 37% on colour.", 0),
        ("So filtering deletes the right answer roughly a third of the time.", 0, BAD),
        ("Instead we re-sort: agree +1, disagree -1, unknown 0. Nothing leaves the pool.", 0, GOOD),
        ("Every metric improved when we switched.", 0),
        ("Cost: a bad label still drags a good product down, just not off the list.", 0, MUTED),
    ])

    # ---- 8. questions ---------------------------------------------------
    slide = _blank(prs)
    heading(slide, "Engineering choice: which question to ask", "pros and cons")
    bullets(slide, [
        ("A question the shopper cannot answer burns a whole turn.", 0),
        ("We measured how often each attribute gets a real answer:", 0),
        ("feature 96%   material 73%   colour 26%   style 9%   size 5%   use case 2%", 1, ACCENT),
        ("The old order asked the three rarest first.", 0, BAD),
        ("Re-ordering by answerability was worth +0.0109, mostly from fewer turns.", 0, GOOD),
        ("Honest caveat: \"feature\" is the simulator's catch-all bucket.", 0, MUTED),
        ("That 96% may not hold on the real grader. We kept the fallback rather than over-fit.", 1, MUTED),
    ])

    # ---- 9. reranking ---------------------------------------------------
    slide = _blank(prs)
    heading(slide, "Engineering choice: how deep to re-rank", "pros and cons")
    bullets(slide, [
        ("A cross-encoder reads the query and the product together. Our other legs never do.", 0),
        ("We first scored the top 10 — exactly the slate we return.", 0),
        ("That was a mistake we caught by reasoning, not by measuring.", 0, BAD),
        ("Re-ranking exactly what you already return can only reorder it.", 1),
        ("It can never turn a miss into a hit. It only moves MRR.", 1),
        ("Scoring the top 20 lets a product at rank 15 be promoted into the shown 10.", 0, GOOD),
        ("That is the only way this stage can add a hit.", 1),
    ])

    # ---- 10. model choice -----------------------------------------------
    slide = _blank(prs)
    heading(slide, "Engineering choice: which models", "pros and cons")
    bullets(slide, [
        ("bge-small (129 MB) — dense search and all three classifiers.", 0),
        ("MiniLM cross-encoder (87 MB) — re-ranking, about 17 ms per pair.", 0),
        ("We also tried Qwen3-Reranker-0.6B. It judges better.", 0),
        ("It needs 27 seconds per pair. That is ~75 hours for ONE evaluation run.", 0, BAD),
        ("So we could never A/B it. A model we cannot measure is a model we cannot justify.", 0, ACCENT),
        ("Nothing is fine-tuned. The rules put training base models out of scope.", 0, MUTED),
    ])

    # ---- 11. rejected ---------------------------------------------------
    slide = _blank(prs)
    heading(slide, "Four things we built, measured, and threw away", "negative results")
    bullets(slide, [
        ("Rewriting the query on a mid-chat pivot:  -0.0580", 0, BAD),
        ("The pivot message only names the CHANGED attribute. The category was said once, in turn 1.", 1),
        ("Dropping history searched 50,000 products for \"leather\" with no category.", 1),
        ("Cleaning \"I don't have a preference\" out of the query:  -0.0410", 0, BAD),
        ("It looks like noise. It is actually an exact-match key, because of the 89.7% finding.", 1),
        ("Cross-session user profiles:  no effect", 0, BAD),
        ("Two sessions with the same profile share a category 0.5% of the time. Chance is 1.2%.", 1),
        ("Turn-based slate diversity:  no effect", 0, BAD),
    ])

    # ---- 12. discipline -------------------------------------------------
    slide = _blank(prs)
    heading(slide, "How we avoided fooling ourselves", "method")
    bullets(slide, [
        ("Rule 1: the \"off\" setting must reproduce the shipped score before we trust any result.", 0),
        ("This caught a real bug. A flag whose zero value was not truly off.", 1),
        ("It had quietly shifted the control leg of an entire experiment.", 1, BAD),
        ("Rule 2: count sessions, not percentages.", 0),
        ("On 200 samples, 0.7500 vs 0.7450 hit rate is ONE session. That is noise.", 1),
        ("Rule 3: write the prediction down before running.", 0),
        ("A good number from the wrong mechanism is still a failure.", 1),
    ])

    # ---- 13. offline ----------------------------------------------------
    slide = _blank(prs)
    heading(slide, "It has to work with the network off", "robustness")
    bullets(slide, [
        ("The rules warn that scoring may run with no internet.", 0),
        ("We found our agent degraded silently. With no network it still started.", 0, BAD),
        ("It still returned 10 products — but dense search and all three classifiers were dead.", 1),
        ("Nothing raised an error. The score just dropped.", 1),
        ("Two guards now ship: one script fetches assets, one verifies offline startup and fails loudly.", 0, GOOD),
    ])

    # ---- 14. limitations ------------------------------------------------
    slide = _blank(prs)
    heading(slide, "What we would fix with more time", "limitations")
    bullets(slide, [
        ("Only 3 of 10 attributes have a real extractor. Style, size, use case and feature have none.", 0),
        ("Catalog labels are thin: material 71%, colour 40%, price 21%.", 0),
        ("Buying sessions score 0.688 against browsing's 0.825.", 0),
        ("The reason is measured: the typical buying constraint is ONE common word.", 1),
        ("\"cotton\" alone matches 18.8% of the catalog. That is low information, not low content.", 1),
        ("Our dense leg adds no recall on this benchmark. We kept it as insurance against paraphrasing.", 0, MUTED),
    ])

    # ---- 15. team -------------------------------------------------------
    slide = _blank(prs)
    heading(slide, "Team and process", "credits")
    bullets(slide, [
        ("Phan Kang Xun — architecture, retrieval, all experiments, evaluation tooling, report.", 0),
        ("Lloyd Wang — registered team member.", 0),
        ("Built with heavy use of an AI coding assistant for implementation and documentation.", 0),
        ("The rules permit this. We state it because it is true.", 1, MUTED),
        ("Every experiment was accepted or rejected on its measured number.", 0, ACCENT),
        ("Including the four we threw away.", 1),
    ])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
