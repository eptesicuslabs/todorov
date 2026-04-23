# chapter 1 outline — what a number means

status: approved canonical seed (2026-04-22). exploratory outline adopted for the chapter-1 completion sprint.

## the question this chapter answers

what is a number, and why do we use symbols (letters, variables) to stand in for numbers?

## why it matters

every subsequent chapter uses numbers and symbols. without a grounded mental model of what a number IS — not just how to compute with one — everything from derivatives (ch. 2) to probability distributions (ch. 7) to softmax attention (ch. 26) reduces to pattern-matching on symbols without comprehension. this chapter builds the foundation from absolute zero: four uses of the word "number", the ladder of number systems (naturals → integers → rationals → reals), what a variable is, what a function is, the three classifications of functions that later chapters rely on (injective, surjective, bijective).

## prerequisites

none. chapter 1 is the entry point of the curriculum.

## proposed structure

~20 pages, low end of the 20-25 range per the plan's "prefer concise" rule. each section specified below with approximate page count.

### section 1 — what "number" refers to (~2pp)

four distinct uses of the word "number" in everyday language:
- counts ("three apples")
- measurements ("3.5 meters")
- labels ("phone number 555-1234")
- positions ("third in line")

only the first two are mathematical numbers in the sense this chapter cares about. the split matters for later chapters: counts / measurements become the subject of calculus; positions become the subject of indexing into tensors.

### section 2 — counting with the integers (~3pp)

the natural numbers 0, 1, 2, 3, ... as the answer to "how many?". addition and multiplication close over the naturals. subtraction does NOT (3 − 5 has no answer inside the naturals). this motivates the integers: ..., −2, −1, 0, 1, 2, ..., where every subtraction has an answer.

### section 3 — measuring with the rationals and reals (~3pp)

division does not close over the integers (1 ÷ 2 has no answer inside ℤ). this motivates the rationals — fractions p/q with integer p and nonzero integer q. the rationals are dense (between any two rationals there is another rational) but they have holes: √2 is not a rational. the reals fill the holes. the chapter does NOT rigorously construct the reals (Dedekind cuts, Cauchy sequences) — that belongs in a real-analysis course. the point here is the ladder: each number system solves a specific problem the previous one could not.

### section 4 — symbols and variables (~3pp)

why a letter can stand for any number. the distinction between a specific number (5) and a variable (x) that ranges over a specified set of numbers. common mistakes to flag:
- equality (=) vs assignment (:= or ←) — different meanings in math vs programming
- the same letter meaning different things across contexts (x in one equation is not always x in the next)
- universal vs existential use (∀x vs ∃x — brief preview, full treatment deferred to ch. 7)

minimal-pair verification trick:
- "x + 1 = 3" — an equation; solve for x
- "x + 1 = 3x" — an equation with a specific solution
- "x + 1 = 1 + x" — an identity; true for every x

### section 5 — functions (~3pp)

a function is a rule mapping inputs to outputs. notation f: X → Y. three ways to specify a function:
- formula: f(x) = x²
- table: small discrete domains
- verbal rule: "the last digit of n"

the rule IS the function; the graph is one visualization of the rule. not all functions have a closed-form formula, and not all formulas define functions (x² + y² = 1 is a relation, not a function of x).

### section 6 — domain, codomain, range (~3pp)

- domain: where inputs come from
- codomain: the set that outputs live in (declared by the definition)
- range: the set of outputs actually produced (a subset of the codomain)

common mistake: conflating codomain and range. f: ℝ → ℝ defined by f(x) = x² has codomain ℝ but range [0, ∞). the declaration of the codomain matters for later classifications.

### section 7 — injective, surjective, bijective (~2pp)

three classifications that recur through every later chapter:
- injective (one-to-one): different inputs map to different outputs
- surjective (onto): every element of the codomain is hit by some input
- bijective: both injective and surjective; has an inverse

worked examples within the section:
- f(x) = x² on ℝ → ℝ: not injective (2 and −2 both map to 4), not surjective (no input maps to −1)
- f(x) = x² on [0, ∞) → [0, ∞): injective AND surjective (every nonnegative real has exactly one nonnegative square root)
- f(x) = x³ on ℝ → ℝ: bijective

### worked example — counting arguments as a function (~1.5pp)

given 5 people, how many ways to pick 2? answer: 10. the process of getting there is a function:
- input: (n, k) where n is the set size and k is the subset size
- output: the number of k-subsets of an n-element set
- denoted C(n, k) or "n choose k"

this is both a concrete count (something you can list by hand) and an abstract function (something you can apply to arbitrary (n, k) inputs). the example ties together section 1 ("counts are numbers"), section 5 ("functions are rules"), and section 6 ("functions have a domain — (n, k) with 0 ≤ k ≤ n, and a codomain — the naturals").

### quiz (~1pp)

5-7 questions covering every section. answers in per-chapter appendix. a sample:
- which of these four are mathematical numbers: "three apples", "the third room on the left", "phone number 555-1234", "3.5 meters"?
- is subtraction closed over the naturals? justify.
- is f(x) = x² on ℝ → ℝ injective? surjective? why or why not?
- find the range of f(x) = 1/(1 + x²) on ℝ → ℝ.
- prove or disprove: every bijective function has an inverse.

## target length

~20 pages.

## source pointers

no external sources required. chapter 1 is foundational enough that it can be written from well-established mathematical knowledge without citation.

if the draft phase surfaces a specific pedagogical convention worth crediting (e.g., a particular notation from a textbook), the citation will be added there with full attribution per the naming rule.

## naming-rule compliance

chapter 1 introduces no project-specific components. the only named entities are the standard number-system symbols (ℕ, ℤ, ℚ, ℝ) — universal mathematical notation, not published-technique names. the naming rule is trivially satisfied.

## expected worked-example topics (summary)

1. converting "how many ways to pick k from n" into a function C: (n, k) → ℕ
2. minimal pair: x² on ℝ → ℝ vs x² on [0, ∞) → [0, ∞) as injectivity/surjectivity contrast
3. exercise in the quiz: show f(x) = 2x + 1 on ℝ → ℝ is bijective; find its inverse

## LaTeX notes for the drafting phase

- document class `memoir` per the plan's proposed default
- `amsmath` for set-builder and piecewise notation
- `amssymb` for ℕ, ℤ, ℚ, ℝ via `\mathbb{}`
- `amsthm` for `definition` and `example` environments
- inline equations via `$...$`, display equations via `\[...\]`
- one simple TikZ arrow-diagram visual is allowed in section 7 and already fits the draft seed
- no code blocks (chapter 1 is pure math)
- no formal `.bib` workflow required for this chapter; sources can sit in a terminal "sources and further reading" section

## approval record (2026-04-22)

this outline is no longer a pending step-1 artifact. it is the approved canonical seed for the chapter-1 completion sprint. the exploratory research notes and the existing LaTeX draft are adopted as downstream artifacts of this outline, with scope frozen to the structure above.

locked decisions:
- the current structure matches the intended chapter-1 scope
- no historical-motivation section is added
- injective / surjective / bijective stays in chapter 1
- the quiz remains at the end of the chapter
- the chapter may land below the nominal 20-page target if the frozen scope is complete and review-ready
- only lightweight visuals are allowed; the existing section-7 arrow diagram is sufficient

## chapter-1 completion note

chapter 1 now proceeds from this approved outline into audit, draft revision, compile verification, and user draft review. chapter 2 returns to the normal clean protocol and starts from outline approval before any research or drafting.
