import re
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, HRFlowable,
    KeepTogether,
)

PRINT = Path(__file__).parent / "print"
OUT = Path(__file__).parent / "neuroloc_guide.pdf"

BLACK = HexColor("#111111")
DARK = HexColor("#333333")
GREY = HexColor("#666666")
LIGHT = HexColor("#999999")
BODY = HexColor("#222222")
EQ_BG = HexColor("#f5f5f5")
EQ_BORDER = HexColor("#cccccc")
RULE = HexColor("#222222")

S = {
    "cover_title": ParagraphStyle(
        "cover_title", fontSize=32, leading=38, alignment=TA_CENTER,
        textColor=BLACK, spaceAfter=12, fontName="Helvetica-Bold",
    ),
    "cover_sub": ParagraphStyle(
        "cover_sub", fontSize=14, leading=18, alignment=TA_CENTER,
        textColor=DARK, spaceAfter=6, fontName="Helvetica",
    ),
    "cover_info": ParagraphStyle(
        "cover_info", fontSize=10, leading=14, alignment=TA_CENTER,
        textColor=BODY, spaceAfter=4, fontName="Helvetica",
    ),
    "cat_title": ParagraphStyle(
        "cat_title", fontSize=22, leading=26, textColor=DARK,
        spaceAfter=8, spaceBefore=4, fontName="Helvetica-Bold",
    ),
    "h1": ParagraphStyle(
        "h1", fontSize=16, leading=20, textColor=DARK,
        spaceAfter=6, spaceBefore=14, fontName="Helvetica-Bold",
    ),
    "h2": ParagraphStyle(
        "h2", fontSize=13, leading=16, textColor=DARK,
        spaceAfter=4, spaceBefore=10, fontName="Helvetica-Bold",
    ),
    "h3": ParagraphStyle(
        "h3", fontSize=11, leading=14, textColor=BODY,
        spaceAfter=3, spaceBefore=8, fontName="Helvetica-BoldOblique",
    ),
    "body": ParagraphStyle(
        "body", fontSize=10, leading=14, textColor=BODY,
        spaceAfter=6, alignment=TA_JUSTIFY, fontName="Helvetica",
    ),
    "eq": ParagraphStyle(
        "eq", fontSize=10, leading=14, textColor=BLACK,
        spaceAfter=8, spaceBefore=4, fontName="Courier",
        leftIndent=24, rightIndent=24,
        backColor=EQ_BG,
        borderColor=EQ_BORDER, borderWidth=0.5, borderPadding=6,
    ),
    "code": ParagraphStyle(
        "code", fontSize=9, leading=12, textColor=HexColor("#333333"),
        spaceAfter=6, fontName="Courier", leftIndent=20,
        backColor=HexColor("#f0f0f0"),
    ),
    "bullet": ParagraphStyle(
        "bullet", fontSize=10, leading=14, textColor=BODY,
        spaceAfter=3, fontName="Helvetica", leftIndent=20, bulletIndent=10,
    ),
    "toc_entry": ParagraphStyle(
        "toc_entry", fontSize=11, leading=16, textColor=DARK,
        spaceAfter=2, fontName="Helvetica", leftIndent=15,
    ),
    "toc_cat": ParagraphStyle(
        "toc_cat", fontSize=13, leading=18, textColor=DARK,
        spaceAfter=4, spaceBefore=8, fontName="Helvetica-Bold",
    ),
}

EQ_PATTERNS = [
    r'^[A-Za-z_]+\s*[\(\[{]?.*=',
    r'^[A-Za-z_]+\s*\*\s*[A-Za-z_]',
    r'^Delta_',
    r'^[A-Za-z]_\{',
    r'^softmax\(',
    r'^sign\(',
    r'^[Ee]\s*=\s*',
    r'^[Ii]\(',
    r'^[Hh]\(',
    r'^CKA\s*=',
    r'^W\s*=',
    r'^0\s*=',
    r'^pool\s*=',
    r'^denom\s*=',
    r'^r_[0-9]',
    r'^angle\s*=',
    r'^rotated\s*=',
    r'^error_',
    r'^predicted_',
    r'^spikes\s*=',
]


def is_equation(line):
    s = line.strip()
    for pat in EQ_PATTERNS:
        if re.match(pat, s):
            return True
    return False


def fmt(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__([^_]+)__', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', text)
    return text


def strip_first_h1(text):
    lines = text.strip().split("\n")
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines)


def parse(text):
    out = []
    in_code = False
    buf = []
    for line in text.strip().split("\n"):
        s = line.strip()
        if s.startswith("```"):
            if in_code:
                if buf:
                    out.append(Paragraph("<br/>".join(buf), S["code"]))
                buf = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            buf.append(s if s else "&nbsp;")
            continue
        if not s:
            out.append(Spacer(1, 3))
        elif s.startswith("## "):
            out.append(Spacer(1, 4))
            out.append(Paragraph(fmt(s[3:]), S["h2"]))
        elif s.startswith("### "):
            out.append(Paragraph(fmt(s[4:]), S["h3"]))
        elif s.startswith("# "):
            out.append(Paragraph(fmt(s[2:]), S["h1"]))
        elif s.startswith("---"):
            out.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#dddddd")))
        elif s.startswith("- "):
            out.append(Paragraph(f"<bullet>&bull;</bullet> {fmt(s[2:])}", S["bullet"]))
        elif s.startswith("    ") or is_equation(s):
            eq_text = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            out.append(Paragraph(eq_text, S["eq"]))
        else:
            out.append(Paragraph(fmt(s), S["body"]))
    return out


def read(name):
    with open(PRINT / name, encoding="utf-8") as f:
        return f.read()


def build():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=letter,
        leftMargin=0.8*inch, rightMargin=0.8*inch,
        topMargin=0.7*inch, bottomMargin=0.7*inch,
        title="Neuroloc: Biological Neural Computation Guide",
        author="Eptesicus Laboratories",
    )

    story = []

    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("NEUROLOC", S["cover_title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Biological Neural Computation for Architecture Designers",
        S["cover_sub"],
    ))
    story.append(Spacer(1, 30))
    story.append(Paragraph("Eptesicus Laboratories", S["cover_info"]))
    story.append(Paragraph(
        "A companion guide to the Neuroloc research wiki", S["cover_info"],
    ))
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "56 mechanisms  /  16 bridge documents  /  13 comparisons  /  18 simulations",
        S["cover_info"],
    ))
    story.append(Spacer(1, 40))
    story.append(HRFlowable(width="60%", thickness=2, color=RULE))
    story.append(Spacer(1, 10))
    story.append(Paragraph("2026", S["cover_info"]))

    story.append(PageBreak())
    story.append(Paragraph("CONTENTS", S["cat_title"]))
    story.append(HRFlowable(width="100%", thickness=1, color=RULE))
    story.append(Spacer(1, 12))
    toc = [
        ("I. Study Guide", [
            "The Brain in One Page",
            "Neuroscience for ML Engineers (Parts 1\u20137)",
            "Mathematical Foundations",
        ]),
        ("II. Reference", [
            "Glossary (55 terms with ML analogs)",
        ]),
        ("III. Architecture Mapping", [
            "Todorov Biology Map (every component)",
        ]),
        ("IV. Key Findings", [
            "15 Adversarial Findings",
            "Top 5 Interventions",
        ]),
    ]
    for cat, items in toc:
        story.append(Paragraph(cat, S["toc_cat"]))
        for item in items:
            story.append(Paragraph(
                f"<bullet>&bull;</bullet> {item}", S["toc_entry"],
            ))

    story.append(PageBreak())
    story.append(Paragraph("I. STUDY GUIDE", S["cat_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=RULE))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Start here if you know ML but not neuroscience. "
        "Every concept connects to something you already understand.",
        S["body"],
    ))
    story.append(Spacer(1, 6))

    brain = strip_first_h1(read("01_brain_overview.md"))
    story.append(Paragraph("The Brain in One Page", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=RULE))
    story.append(Spacer(1, 6))
    story.extend(parse(brain))

    story.append(PageBreak())
    neuro = strip_first_h1(read("02_neuroscience_primer.md"))
    story.append(Paragraph("Neuroscience for ML Engineers", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=RULE))
    story.append(Spacer(1, 6))
    story.extend(parse(neuro))

    story.append(PageBreak())
    math = strip_first_h1(read("03_mathematical_foundations.md"))
    story.append(Paragraph("Mathematical Foundations", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=RULE))
    story.append(Spacer(1, 6))
    story.extend(parse(math))

    story.append(PageBreak())
    story.append(Paragraph("II. REFERENCE", S["cat_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=RULE))
    story.append(Spacer(1, 6))
    glossary = strip_first_h1(read("04_glossary.md"))
    story.extend(parse(glossary))

    story.append(PageBreak())
    story.append(Paragraph("III. ARCHITECTURE MAPPING", S["cat_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=RULE))
    story.append(Spacer(1, 6))
    biomap = strip_first_h1(read("05_architecture_mapping.md"))
    story.extend(parse(biomap))

    story.append(PageBreak())
    story.append(Paragraph("IV. KEY FINDINGS", S["cat_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=RULE))
    story.append(Spacer(1, 6))

    findings = """## 15 Adversarial Findings

Every bridge document stress-tested its biological mapping. The dominant pattern: Todorov's components share superficial resemblance with biological mechanisms but differ in every substantive dimension.

1. ATMN is not a faithful LIF \u2014 batch reset during training makes it a per-token threshold function.
2. KDA is not STDP \u2014 it is a Hopfield-like associative memory with exponential forgetting.
3. The 41% firing rate is not sparse by cortical standards (1\u201310%).
4. The adaptive threshold alpha * mean(|x|) is not divisive normalization \u2014 it is a global scalar.
5. Next-token prediction is not predictive coding \u2014 but Millidge et al. proved they learn the same weights.
6. The 3:1 layer ratio does not map to cortical layers \u2014 it was derived from ML benchmarks.
7. Mamba3 rotation is not oscillatory dynamics \u2014 it likely serves positional encoding.
8. KDA's alpha is not neuromodulatory gain control \u2014 wrong granularity, static after training.
9. KDA+MLA is not a complementary learning system \u2014 no consolidation, no timescale separation.
10. SwiGLU gating IS dendritic coincidence detection mathematically \u2014 but one branch vs. 30\u201350.
11. The 354x energy claim is per-operation correct but system-level misleading \u2014 data movement dominates.
12. Biological and transformer attention compute different things \u2014 selection vs. retrieval.
13. Spike threshold learning is not critical period plasticity \u2014 no closing mechanism.
14. The PGA-to-grid-cell connection is weak to nonexistent \u2014 a critical control experiment is needed.
15. The residual stream is a bus, not a global workspace \u2014 it lacks ignition, selectivity, and capacity limits.

The one genuine correspondence: the outer-product associative memory in KDA (k * v^T) mirrors Hebbian learning at the mathematical level. This is not an analogy. It is the same equation.

## Top 5 Interventions (Ranked by Probability of Impact)

1. **ATMN Leak Term** (faithful LIF) \u2014 HIGH priority, Phase 5a. Add explicit exponential decay toward resting potential. Makes ATMN a real temporal integrator.

2. **Activity-Dependent Alpha** (BCM-like) \u2014 25\u201335% probability, Phase 5b+. alpha_eff = sigmoid(alpha_log + gamma * log(||S_t||)). Prevents state saturation at long contexts.

3. **k-WTA Ternary Spikes** \u2014 20\u201330% probability, Phase 5+. Replace threshold with top-k selection by absolute value. Guarantees exact sparsity.

4. **Progressive Spike Activation** \u2014 20\u201330% probability, Phase 5+. No spikes during warmup, linear ramp over 10% of training. Addresses STE gradient bias early in training.

5. **Neuromodulator Network** (130 params/layer) \u2014 15\u201325% probability, Phase 6+. Small network reads global state (loss, entropy, spike density) to modulate alpha and beta at inference time."""

    story.extend(parse(findings))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(LIGHT)
        canvas.drawCentredString(
            letter[0] / 2, 0.4 * inch,
            f"Neuroloc  |  Eptesicus Laboratories  |  {doc.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)

    import fitz
    d = fitz.open(str(OUT))
    pages = len(d)
    d.close()
    print(f"Built: {OUT}")
    print(f"Size: {OUT.stat().st_size / 1024:.0f} KB")
    print(f"Pages: {pages}")


if __name__ == "__main__":
    build()
