import re
import subprocess
from pathlib import Path

PRINT = Path(__file__).parent / "print"
OUT = Path(__file__).parent / "neuroloc_guide.tex"
PDF = Path(__file__).parent / "neuroloc_guide.pdf"
MIKTEX = Path(r"C:\Users\deyan\AppData\Local\Programs\MiKTeX\miktex\bin\x64")


def esc(text):
    text = text.replace("\\", "\\textbackslash ")
    for ch in "&%$#{}":
        text = text.replace(ch, f"\\{ch}")
    text = text.replace("_", "\\_")
    text = text.replace("~", "\\textasciitilde{}")
    text = text.replace("^", "\\textasciicircum{}")
    text = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\\textit{\1}', text)
    return text


def md_to_latex(text):
    lines = text.strip().split("\n")
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]

    out = []
    in_list = False
    in_enum = False
    in_verb = False

    for line in lines:
        s = line.strip()

        if s.startswith("```"):
            if in_verb:
                out.append("\\end{verbatim}")
                in_verb = False
            else:
                close_lists(out, in_list, in_enum)
                in_list = in_enum = False
                out.append("\\begin{verbatim}")
                in_verb = True
            continue

        if in_verb:
            out.append(line)
            continue

        if not s:
            if in_list:
                out.append("\\end{itemize}")
                in_list = False
            if in_enum:
                out.append("\\end{enumerate}")
                in_enum = False
            out.append("")
            continue

        if s.startswith("## "):
            close_lists(out, in_list, in_enum)
            in_list = in_enum = False
            out.append(f"\n\\subsection*{{{esc(s[3:])}}}\n")
        elif s.startswith("### "):
            close_lists(out, in_list, in_enum)
            in_list = in_enum = False
            out.append(f"\n\\subsubsection*{{{esc(s[4:])}}}\n")
        elif s.startswith("---"):
            close_lists(out, in_list, in_enum)
            in_list = in_enum = False
            out.append("\\vspace{4pt}\\noindent\\rule{\\textwidth}{0.3pt}\\vspace{4pt}")
        elif s.startswith("- "):
            if not in_list:
                out.append("\\begin{itemize}[leftmargin=1.5em,itemsep=2pt,parsep=0pt]")
                in_list = True
            out.append(f"  \\item {esc(s[2:])}")
        elif re.match(r'^\d+\.\s', s):
            content = re.sub(r'^\d+\.\s*', '', s)
            if not in_enum:
                out.append("\\begin{enumerate}[leftmargin=1.5em,itemsep=2pt,parsep=0pt]")
                in_enum = True
            out.append(f"  \\item {esc(content)}")
        elif s.startswith("    "):
            close_lists(out, in_list, in_enum)
            in_list = in_enum = False
            raw = s.strip()
            out.append(f"\n\\vspace{{2pt}}\\noindent\\hspace{{1.5em}}\\texttt{{{esc(raw)}}}\n")
        else:
            out.append(esc(s))

    close_lists(out, in_list, in_enum)
    if in_verb:
        out.append("\\end{verbatim}")
    return "\n".join(out)


def close_lists(out, in_list, in_enum):
    if in_list:
        out.append("\\end{itemize}")
    if in_enum:
        out.append("\\end{enumerate}")


def read(name):
    with open(PRINT / name, encoding="utf-8") as f:
        return f.read()


def build():
    brain = md_to_latex(read("01_brain_overview.md"))
    neuro = md_to_latex(read("02_neuroscience_primer.md"))
    math = md_to_latex(read("03_mathematical_foundations.md"))
    glossary = md_to_latex(read("04_glossary.md"))
    biomap = md_to_latex(read("05_architecture_mapping.md"))

    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[top=2.5cm,bottom=2.5cm,left=2.5cm,right=2.5cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{parskip}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{hyperref}
\hypersetup{hidelinks}

\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\footnotesize Neuroloc\quad|\quad Eptesicus Laboratories\quad|\quad\thepage}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

\titleformat{\section}{\Large\bfseries}{}{0pt}{}[\vspace{2pt}\hrule\vspace{6pt}]
\titleformat{\subsection}{\large\bfseries}{}{0pt}{}
\titleformat{\subsubsection}{\normalsize\bfseries\itshape}{}{0pt}{}

\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}

\begin{document}

\thispagestyle{empty}
\vspace*{4cm}
\begin{center}
{\Huge\bfseries NEUROLOC}\\[12pt]
{\large Biological Neural Computation for Architecture Designers}\\[40pt]
Eptesicus Laboratories\\[6pt]
{\small A companion guide to the Neuroloc research wiki}\\[20pt]
{\small 56 mechanisms\quad/\quad 16 bridge documents\quad/\quad 13 comparisons\quad/\quad 18 simulations}\\[40pt]
\rule{0.4\textwidth}{1pt}\\[10pt]
2026
\end{center}
\newpage

\section*{Contents}
\textbf{I. Study Guide}
\begin{itemize}[leftmargin=1.5em,itemsep=1pt]
  \item The Brain in One Page
  \item Neuroscience for ML Engineers (Parts 1--7)
  \item Mathematical Foundations
\end{itemize}
\textbf{II. Reference}
\begin{itemize}[leftmargin=1.5em,itemsep=1pt]
  \item Glossary (55 terms with ML analogs)
\end{itemize}
\textbf{III. Architecture Mapping}
\begin{itemize}[leftmargin=1.5em,itemsep=1pt]
  \item Todorov Biology Map (every component)
\end{itemize}
\textbf{IV. Key Findings}
\begin{itemize}[leftmargin=1.5em,itemsep=1pt]
  \item 15 Adversarial Findings
  \item Top 5 Interventions
\end{itemize}
\newpage

\section*{I. Study Guide}
Start here if you know ML but not neuroscience. Every concept connects to something you already understand.

\subsection*{The Brain in One Page}
""" + brain + r"""

\newpage
\subsection*{Neuroscience for ML Engineers}
""" + neuro + r"""

\newpage
\subsection*{Mathematical Foundations}
""" + math + r"""

\newpage
\section*{II. Reference}
""" + glossary + r"""

\newpage
\section*{III. Architecture Mapping}
""" + biomap + r"""

\newpage
\section*{IV. Key Findings}

\subsection*{15 Adversarial Findings}

Every bridge document stress-tested its biological mapping. The dominant pattern: Todorov's components share superficial resemblance with biological mechanisms but differ in every substantive dimension.

\begin{enumerate}[leftmargin=1.5em,itemsep=2pt]
  \item ATMN is not a faithful LIF --- batch reset during training makes it a per-token threshold function.
  \item KDA is not STDP --- it is a Hopfield-like associative memory with exponential forgetting.
  \item The 41\% firing rate is not sparse by cortical standards (1--10\%).
  \item The adaptive threshold $\alpha \cdot \mathrm{mean}(|x|)$ is not divisive normalization --- it is a global scalar.
  \item Next-token prediction is not predictive coding --- but Millidge et al.\ proved they learn the same weights.
  \item The 3:1 layer ratio does not map to cortical layers --- it was derived from ML benchmarks.
  \item Mamba3 rotation is not oscillatory dynamics --- it likely serves positional encoding.
  \item KDA's $\alpha$ is not neuromodulatory gain control --- wrong granularity, static after training.
  \item KDA+MLA is not a complementary learning system --- no consolidation, no timescale separation.
  \item SwiGLU gating IS dendritic coincidence detection mathematically --- but one branch vs.\ 30--50.
  \item The 354$\times$ energy claim is per-operation correct but system-level misleading --- data movement dominates.
  \item Biological and transformer attention compute different things --- selection vs.\ retrieval.
  \item Spike threshold learning is not critical period plasticity --- no closing mechanism.
  \item The PGA-to-grid-cell connection is weak to nonexistent --- a critical control experiment is needed.
  \item The residual stream is a bus, not a global workspace --- it lacks ignition, selectivity, and capacity limits.
\end{enumerate}

The one genuine correspondence: the outer-product associative memory in KDA ($\mathbf{k}_t \mathbf{v}_t^\top$) mirrors Hebbian learning at the mathematical level. This is not an analogy. It is the same equation.

\subsection*{Top 5 Interventions (Ranked by Probability of Impact)}

\begin{enumerate}[leftmargin=1.5em,itemsep=4pt]
  \item \textbf{ATMN Leak Term} (faithful LIF) --- HIGH priority, Phase 5a.\\
    Add explicit exponential decay toward resting potential. Makes ATMN a real temporal integrator.
  \item \textbf{Activity-Dependent Alpha} (BCM-like) --- 25--35\% probability, Phase 5b+.\\
    $\alpha_{\mathrm{eff}} = \sigma(\alpha_{\log} + \gamma \cdot \log \|\mathbf{S}_t\|)$. Prevents state saturation at long contexts.
  \item \textbf{k-WTA Ternary Spikes} --- 20--30\% probability, Phase 5+.\\
    Replace threshold with top-$k$ selection by absolute value. Guarantees exact sparsity.
  \item \textbf{Progressive Spike Activation} --- 20--30\% probability, Phase 5+.\\
    No spikes during warmup, linear ramp over 10\% of training. Addresses STE gradient bias early in training.
  \item \textbf{Neuromodulator Network} (130 params/layer) --- 15--25\% probability, Phase 6+.\\
    Small network reads global state (loss, entropy, spike density) to modulate $\alpha$ and $\beta$ at inference time.
\end{enumerate}

\end{document}
"""

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"LaTeX written: {OUT}")

    pdflatex = str(MIKTEX / "pdflatex.exe")
    for i in range(2):
        r = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-output-directory", str(OUT.parent), str(OUT)],
            capture_output=True, text=True, timeout=180,
        )

    if PDF.exists():
        errors = sum(1 for line in r.stdout.split("\n") if line.startswith("!"))
        import fitz
        d = fitz.open(str(PDF))
        print(f"Built: {PDF}")
        print(f"Size: {PDF.stat().st_size / 1024:.0f} KB")
        print(f"Pages: {len(d)}")
        print(f"LaTeX errors: {errors}")
        d.close()
    else:
        print("FAILED. Errors:")
        for line in r.stdout.split("\n"):
            if line.startswith("!"):
                print(f"  {line}")


if __name__ == "__main__":
    build()
