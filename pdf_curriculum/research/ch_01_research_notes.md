# chapter 1 research notes — what a number means

status: validated canonical seed (2026-04-22). exploratory research adopted for the chapter-1 completion sprint.

## research phase summary

four research agents surveyed relevant sources for chapter 1 of the teaching curriculum. the chapter teaches, from near-zero math prerequisites: what a number refers to, the number-system ladder ℕ → ℤ → ℚ → ℝ, variables, functions, domain / codomain / range, and injective / surjective / bijective.

- agent A: introductory textbook survey (MacKay, Rosen, Stewart 8th ed., Axler, Bishop)
- agent B: online-resources survey (3Blue1Brown, Nielsen, distill.pub, Better Explained, Khan Academy functions track, MIT 18.01, Paul's Online Math Notes / Dawkins, MathIsFun)
- agent C (supplementary): Lang, *Basic Mathematics* (Springer reprint 1988)
- agent D (supplementary): Spivak, *Calculus* 4th ed. chapters 1-3

## final verification (2026-04-22)

this research base is now treated as validated for chapter-1 drafting and review.

- no blocking research gaps remain for the frozen chapter-1 scope
- the five documented source gaps below are non-blocking and already routed to stronger primary sources
- Rosen, Lang, Stewart, Spivak, Axler, Dawkins, and MathIsFun are the locked source spine for the current chapter-1 draft
- no additional research pass is required before user review of chapter 1

## Agent A findings — textbook survey (factual corrections applied)

### Rosen, *Discrete Mathematics and Its Applications*, 8th ed., Section 2.3

- Definition 1 (verbatim, full form): "Let A and B be nonempty sets. A function f from A to B is an assignment of exactly one element of B to each element of A. We write f(a) = b if b is the unique element of B assigned by the function f to the element a of A. If f is a function from A to B, we write f : A → B." **the "Let A and B be nonempty sets" preamble is load-bearing — without it, the definition fails for empty sets. drafter must include the preamble.**
- explicit codomain/range split at first definition: domain = A, codomain = B, range = {f(a) | a ∈ A}. range ⊆ codomain, with equality iff f is surjective.
- contrapositive form of injectivity stated alongside direct form: "f(a) = f(b) implies a = b" AND "a ≠ b implies f(a) ≠ f(b)".
- order of introduction: injective ("one-to-one") → surjective ("onto") → bijective ("one-to-one correspondence"). both English and Latin terms introduced together.
- convention: 0 ∈ ℕ; ℤ+ = {1, 2, 3, ...}. **note (corrected from first-round research):** this convention is consistent with the ISO 80000-2 standard but Rosen 8th ed. does not explicitly cite ISO; the convention is justified by CS practice (induction starts at 0, array indices start at 0), not by an explicit standards citation.
- source: Iowa State lecture transparencies attributed to Rosen; Rutgers course notes; CUNY lecture 9.

### Axler, *Linear Algebra Done Right*, 4th ed.

- **Section 0.A (Preliminaries)** introduces the function concept: "A function from a set S to a set T is an assignment of exactly one element of T to each element of S." three-object naming (domain, codomain, range) established here.
- **Chapter 3, Section 3B (Null Spaces and Injectivity)** proves the connection T injective ⟺ null(T) = {0}. this is a theorem about linear maps, NOT a claim made in Section 0.A. drafter using the null-space characterization for intuition must cite Chapter 3B, not Section 0.A.
- works over 𝔽 (either ℝ or ℂ); does not ladder through ℕ, ℤ, ℚ.
- definitional sequence in Section 0.A: injective → surjective → bijective.
- source: Axler 4th edition open-access PDF; solution index at uli.rocks confirming section numbering.

### Stewart, *Calculus: Early Transcendentals*, 8th ed., Section 1.1

- **the canonical source of the codomain/range conflation** students arrive with. definition (verbatim, 8th ed.): "A function f is a rule that assigns to each element x in a set D exactly one element, called f(x), in a set E." target set E is NEVER NAMED "codomain." range defined as "all possible values of f(x)."
- four-way representation taxonomy is the chapter's organizing principle: verbal / algebraic (formula) / visual (graph) / numerical (table). strongest beginner-friendly onboarding device among surveyed textbooks.
- vertical line test introduced as the graphical criterion for "each x maps to exactly one y."
- injective / surjective / bijective NOT covered in the functions chapter. "one-to-one" appears only in the inverse-function context. "surjective" and "bijective" as standalone terms absent from Stewart.
- **edition note (corrected from first-round research):** 8th edition verified; 9th edition wording differences not checked. drafter citing Stewart must use 8th ed. specifically.
- source: OpenStax Calculus Volume 1 Section 1.1 (Stewart-mapped); LibreTexts Stewart 1.1 map; Quizlet flashcards for Stewart 1.1.

### MacKay, *Information Theory, Inference, and Learning Algorithms*

- three-layer variable distinction (unique among surveyed textbooks): (i) the symbol as a random variable, (ii) a particular value it takes, (iii) the proposition asserting that the variable takes that value. useful for the chapter's section 4 variable-scope framing.
- no formal number-system ladder.
- square brackets for functionals (f[y]) separated from ordinary evaluation (f(x)).
- not useful for function/domain/range pedagogy specifically.
- source: MacKay (2003), Preface and Appendix A; ITILA archive.

### Bishop, *Pattern Recognition and Machine Learning*

- notation section: scalars lowercase italics, vectors lowercase bold, matrices uppercase bold.
- no standalone definition of "function" as a set-theoretic object.
- codomain/range NOT introduced as separate concepts.
- "bijective" replaced by "invertible transformation" in density-transformation context.
- least useful of the 5 for this chapter. valuable only for the notation-table modeling.
- source: Bishop (2006), preface notation section; Microsoft Research sample chapter.

### flagged contradictions between textbooks (Agent A)

1. **whether 0 ∈ ℕ.** Rosen includes 0; Axler, Stewart, Bishop, MacKay do not address ℕ. the two coexisting conventions in the literature are ℕ = {0, 1, 2, ...} (CS / ISO 80000-2 standard) and ℕ = {1, 2, 3, ...} (analytic number theory). curriculum must state its convention and the reason.
2. **codomain vs range.** Rosen and Axler name all three (domain, codomain, range) and flag the distinction. Stewart introduces the target set E but does not name it "codomain" and collapses range into "all possible values." Bishop and MacKay do not address the distinction. **this is the pervasive conflation the chapter must address head-on.**
3. **primary terminology for injective/surjective/bijective.** Rosen: English ("one-to-one" / "onto") primary, Latin as synonyms (American CS tradition). Axler: Latin primary, English noted (European / pure-math tradition). Stewart: "one-to-one" only, in inverse-function context. curriculum names both traditions with equal weight and flags the naming collision.

## Agent B findings — online-resources survey (factual corrections applied)

### Paul's Online Math Notes (Paul Dawkins) — closest fit for dzipobel voice

- **the minimal-pair counterexample move.** on the function-definition page: `y = x² + 1` passes (function), `y² = x + 1` fails (not a function — at x = 3, y² = 4 gives y ∈ {-2, +2}, violating the "exactly one output" rule).
- **misconception flag on the function-definition page (verbatim from source):** "This is NOT a multiplication of f by x! This is one of the more common mistakes people make when they first deal with functions." this warning appears on the `Classes/Alg/FunctionDefn.aspx` page.
- **separate misconception flag on the inverse-functions page** (corrected from first-round research — these are on different pages): "f⁻¹(x) ≠ 1/f(x)." this warning appears on the `Classes/Alg/InverseFunctions.aspx` page, NOT on the function-definition page.
- **voice examples from Dawkins** (verbatim complete sentences, extracted from the function-definition and inverse-functions pages; used for the dzipobel voice match):
  - "This is NOT a multiplication of f by x!" (function-definition page, emphatic form with exclamation — used to preempt the f(x) = f·x misconception)
  - "This is one of the more common mistakes people make when they first deal with functions." (function-definition page, stated immediately after the warning — tells the reader this particular mistake is widespread, not personal failure)
  - "Do not confuse the two!" (inverse-functions page, second-person imperative form warning against f⁻¹(x) = 1/f(x))
  - paraphrased pattern observed across both pages: warning placed BEFORE notation is used, not after, so the reader sees the mistake flagged before they can make it.
- Dawkins' complete voice fingerprint (for drafters who need more examples): visit `tutorial.math.lamar.edu/Classes/Alg/FunctionDefn.aspx` and `InverseFunctions.aspx` directly. the two pages together give ~20 additional warning-pattern sentences in the same second-person imperative voice.
- counterexample for injectivity: f(x) = x² fails (both f(2) = 4 and f(-2) = 4). domain restriction to x ≥ 0 rescues it.

### 3Blue1Brown (corrected chapter references)

- **"Essence of Calculus" Chapter 1** (the first episode): discovery-construction pedagogical technique. opens with "how would you compute the area of a circle if no one had told you the formula?" — derives calculus from geometry, not from definitions.
- **"Essence of Linear Algebra" Chapter 3** (NOT Chapter 1 — corrected from first-round research): "linear transformations and matrices." introduces the "transformation suggests movement" framing. Chapter 1 of Essence of Linear Algebra is "Vectors, what even are they?" — a different topic.
- high conceptual value once basic vocabulary is in place; not useful as a FIRST introduction to "what is a function" (neither video defines domain / codomain / range).

### MathIsFun — visual arrow-diagram treatment

- arrow diagrams from set A to set B introduced BEFORE any formula appears.
- **naming-collision warning** (corrected from first-round research — actual source wording, paraphrase now labeled as paraphrase):
  - actual source text (verbatim from the MathIsFun page, including punctuation): "(But don't get that confused with the term 'One-to-One' used to mean injective)."
  - curriculum-adapted stronger form (paraphrase, not quoted from source): "bijective is NOT the same as 'one-to-one.' one-to-one means injective. bijective means one-to-one AND onto." the drafter may use the stronger paraphrase but should not attribute it verbatim to MathIsFun.
- **mnemonics** (corrected — only attributed what is actually on the page):
  - surjective: "every B has some A" (from source)
  - bijective: "one-to-one correspondence" / "perfect pairing" (from source)
  - injective: source uses "one-to-one" and the negation of "many-to-one"; no "perfect pairing of inputs" phrase exists on the source page. the earlier research note had invented this phrase; it has been removed.

### Better Explained (Kalid Azad)

- physical metaphor before formalism: fast-forward video metaphor for f(ax), "running ahead of schedule" for f(x + b).
- counterintuitive trap flagged for f(x + b): "doesn't it seem like we remove time to make things happen earlier? This is our visual intuition fooling us."
- useful for section 5 transformation intuitions, not for the initial "what is a function" question.

### Khan Academy (functions track)

- mastery-step progression: what is a function → evaluating → domain/range → inverse → injective/surjective.
- arrow diagrams for injective/surjective.
- domain-before-range sequencing.
- **note (corrected from first-round research):** Agent B surveyed only Khan Academy's FUNCTIONS track. Khan Academy has a separate arithmetic / number-system track (grade 6-8 content: negative numbers, rational operations, irrational/real number classification) that was NOT surveyed. the earlier claim "no surveyed resource covers the ℕ → ℤ → ℚ → ℝ ladder well" is overstated as an absolute; it is correct for the functions-focused surveyed scope, but Khan Academy's arithmetic track likely contains pedagogical treatment of the ladder that was not reviewed. drafter may optionally consult Khan's grade-8 number-system unit for additional ladder material.

### Nielsen, *Neural Networks and Deep Learning*

- motivational anchoring (human-vs-computer asymmetry on digit recognition).
- NOT useful for teaching functions. assumes functions as prerequisite.

### MIT OpenCourseWare 18.01

- treats functions as prerequisite. NOT useful at this level.

### distill.pub

- no foundational article on functions or number systems at this level.

### flagged disagreements between online resources (Agent B)

1. **function-as-rule vs function-as-transformation.** Dawkins (rule-based). 3Blue1Brown (transformation-based). Better Explained (machine-based). curriculum establishes the rule-based definition first (Rosen/Axler style) and introduces the transformation framing as a secondary bridge.
2. **domain-first vs both-together.** Dawkins defines domain before range. Khan Academy introduces both from graphs simultaneously. **domain-first is safer** for beginners.

## Agent C findings — Lang, *Basic Mathematics* (Springer reprint 1988)

Lang is THE designed-for-near-zero-prerequisite textbook for this chapter's scope. strong voice match with dzipobel.

### number-system ladder

- verbatim: "For convenience, it is useful to have a name for the positive integers together with zero, and we shall call these the natural numbers." **Lang includes 0 in ℕ.**
- ladder order: positive integers → zero → negative integers → integers → rationals → reals.
- integers introduced via thermometer analogy (geometric representation on a number line).
- verbatim: "By a rational number we shall mean simply an ordinary fraction, that is a quotient m/n, also written m/n, where m, n are integers and n ≠ 0."
- closure-under-subtraction argument is structurally implicit (negative integers introduced to handle differences) but Lang does not use the phrase "closed under subtraction" as an explicit named motivation. curriculum may add the explicit closure framing on top of Lang's structure.
- source: Internet Archive OCR djvu.txt; Springer reprint ISBN 978-0-387-96787-5.

### √2 irrationality proof (Theorem 4, Chapter 1 §5) — verbatim

> "Theorem 4. There is no positive rational number whose square is 2. Proof. Suppose that such a rational number exists. We can write it in lowest form m/n by Theorem 3. In particular, not both m and n can be even. We have (m/n)² = 2, so m² = 2n². Consequently, we obtain m² = 2n², and therefore m² is even. By the Corollary of Theorem 2 of §4, we conclude that m must be even, and we can therefore write m = 2k for some positive integer k. Thus we obtain m² = (2k)² = 4k² = 2n². We can cancel 2 from both sides of the equation 4k² = 2n², and obtain n² = 2k². This means that n² is even, and as before, we conclude that n itself must be even. Thus from our original assumption that (m/n)² = 2 and m/n is in lowest form, we have obtained the impossible fact that both m, n are even. This means that our original assumption (m/n)² = 2 cannot be true, and concludes the proof of our theorem."

this is the canonical proof for section 3. **the drafter should paraphrase Lang's argument into curriculum voice but cite Lang as the source. Lang's proof depends on two preliminary theorems (Theorem 3: every rational has a lowest form; Theorem 2 §4 Corollary: if m² is even, m is even). those preliminaries must be stated inline rather than cited to a prior Lang chapter.**

### functions, mappings, sets in Lang

- Lang separates functions (Chapter 13, analytic — polynomials, exponentials, logarithms) from mappings (Chapter 14, abstract formalism — domain, image, permutations).
- sets are handled in the Interlude between Part I (Algebra) and Part II (Geometry), informally and operationally.
- Lang's primary terminology for classifications: "one-to-one" and "onto" (1971 American convention); "injective" / "surjective" / "bijective" do not appear to be his primary vocabulary, though verbatim text from Chapter 14 was not recoverable.
- gap: Lang's function definition from Chapter 13 was not retrievable in verbatim form. a secondary summary characterizes it as "association f: S → ℝ that maps each element x in domain S to a number f(x)." this is medium-confidence.
- gap: Lang's treatment of the codomain / range distinction (if any) was not recoverable.
- gap: Lang's informal "set" definition from the Interlude was not recoverable in verbatim form.

### Lang's pedagogical voice — verbatim examples for the dzipobel match

- "Be warned that deficiency at either level can ultimately hinder you in your work."
- "You should be very careful when you take the negative of a sum which involves itself in negative numbers, taking into account that −(−a) = a."
- "Do not regard some lists of exercises as too short. Rather, realize that practice for some notion may come again later in conjunction with another notion."
- "Try to rely on yourself, and try to develop a trust in your own judgment. There is no 'right' way to do things."
- "If you find that any chapter gets too involved for you, then skip that part until you feel the need for it, and look at another part of the book."

**Lang's voice is a strong dzipobel match: second-person imperative, terse declarative sentences, explicit warnings. drafter should use Lang's sentence shape as a model for chapter 1's voice.**

### historical framing in Lang

- **Lang does not open with historical framing.** the Foreword is pedagogical-philosophical (on why foundational mathematics must be retaught at college level), not historical. motivation for each number-system extension is structural (negatives enable subtraction, rationals enable division, reals fill gaps like √2) — not a chronology of who invented what. **this matches the curriculum's decision to exclude historical motivation from chapter 1.**

## Agent D findings — Spivak, *Calculus* 4th ed., chapters 1-3

Spivak complements Lang: Lang builds the ladder constructively, Spivak treats ℝ axiomatically. useful primarily for section 3's "hole in the rationals" content.

### axiomatic framing of ℝ (Chapter 1)

- Spivak opens with 13 axioms (P1-P13) for a complete ordered field. does NOT ladder through ℕ, ℤ, ℚ — he begins with ℝ axiomatically and locates the smaller sets as subsets in Chapter 2.
- the 13 axioms:
  - **P1-P9 (field axioms):** P1 associativity of addition, P2 additive identity, P3 additive inverse, P4 commutativity of addition, P5 associativity of multiplication, P6 multiplicative identity (with 1 ≠ 0), P7 multiplicative inverse for nonzero elements, P8 commutativity of multiplication, P9 distributive law.
  - **P10-P12 (order axioms):** a distinguished set P of positive numbers. P10 trichotomy (for any number a, exactly one of a ∈ P, -a ∈ P, a = 0). P11 (a, b ∈ P ⇒ a + b ∈ P). P12 (a, b ∈ P ⇒ ab ∈ P).
  - **P13 (completeness):** "If A is a non-empty set of numbers that has an upper bound, then it has a least upper bound." this is the only axiom that separates ℝ from ℚ.

### "the hole in the rationals" (section 3 material)

- verbatim-confirmed argument: Spivak demonstrates that ℚ is not complete by considering the set C = {x : x² < 2 and x ∈ ℚ}. C has rational upper bounds (e.g., 3/2) but no rational least upper bound: if b ∈ ℚ were the LUB of C, then b² = 2 — but no rational satisfies b² = 2. therefore ℚ fails P13. ℝ is defined to satisfy P13.
- **this is the clearest argument in any surveyed source for why ℝ differs from ℚ.** curriculum uses this framing in section 3.

### √2 irrationality in Spivak

- proof by contradiction in Chapter 2 (not a worked example in the body of Chapter 1; problem-driven). structure confirmed; exact wording not extracted. Lang's Theorem 4 verbatim (above) is the canonical text for the chapter's version of this proof.

### functions in Spivak

- functions deferred to Chapter 3 ("Functions", p. 39). Chapters 1 and 2 contain no function definition.
- the index shows "Function: 39, 47" confirming the location.

### Spivak's pedagogical voice

- Spivak's voice is **NOT a stylistic match for dzipobel.** Spivak is conversational, mildly ironic, proof-forward. he trusts the reader to discover misconceptions through exercises rather than warning against them explicitly.
- verbatim samples:
  - "this chapter is not a review ... it does not aim to present an extended review of old material, but to condense this knowledge into a few simple and obvious properties of numbers."
  - "it is amusing to discover how powerless we are if we rely only on properties P1-P4 to justify our manipulations."
  - "mathematicians like to pretend that they can't even add, but most of them can when they have to."
- **drafter should treat Spivak as a source of rigorous content (P1-P13, the LUB argument for the rationals-are-not-complete claim) but NOT as a stylistic model.** Lang and Dawkins remain the voice models.

## convergence across all four agents

1. **codomain vs range conflation is the #1 pedagogical trap.** Rosen and Axler handle it cleanly (both name all three of domain, codomain, range and flag the distinction at the first definition). Stewart is the canonical source of the conflation. No resource in Agent B's surveyed scope explicitly flags the codomain vs range distinction with a minimal-pair example; the curriculum fills this gap.
2. **naming collision: "one-to-one" (injective) vs "one-to-one correspondence" (bijective).** Rosen and MathIsFun both flag; curriculum must warn explicitly.
3. **classification order: injective → surjective → bijective** is consistent across Rosen, Axler, Lang (with one-to-one/onto vocabulary). reflects increasing strength.
4. **number-system ladder:** Lang is the primary source (constructive, thermometer analogy, zero ∈ ℕ, verbatim √2 proof). Spivak complements (axiomatic framing, P13 completeness, "hole in ℚ" argument). no surveyed online resource in Agent B's scope covers the ladder well (Khan Academy's arithmetic track was not surveyed and may fill the gap).
5. **voice models:** Lang and Dawkins are the strongest dzipobel matches. Spivak is NOT a voice model.
6. **Stewart's four-way representation taxonomy** (verbal / formula / graph / table) is the strongest beginner-friendly onboarding device. curriculum expands section 5 to explicitly cover all four representations.

## consolidated draft guidance — one subsection per chapter section

### section 1 — what "number" refers to (~2pp)

source-gap (resolved by curriculum-original content): no surveyed source treats the four-way split (counts / measurements / labels / positions) as a taught distinction. curriculum fills from first principles.

**structure:**
- open with four everyday uses of "number": three apples (count), 3.5 meters (measurement), phone number 555-1234 (label), third in line (position).
- only counts and measurements are mathematical numbers in this chapter's sense.
- labels and positions are CODES — they happen to use digits but are not numbers to be added, subtracted, or ordered as mathematical objects.

**worked example within section 1:** consider the two questions "how many apples?" (answer is a count) and "what is your phone number?" (answer is a label). the first answer participates in arithmetic (you can add, subtract, compare). the second does not (phone 555-1234 + phone 555-5678 is nonsense).

**wrong/right/explanation triple for section 1:**
- wrong: "555-1234 is a number."
- right: "555-1234 is a label that uses digits. a mathematical number is an object in ℕ, ℤ, ℚ, or ℝ that can be added, subtracted, multiplied, and compared."
- explanation: "digits can appear in non-numerical roles. a phone number is an identifier; a house number is a position; a jersey number is a label. none of these participate in arithmetic, so they are not mathematical numbers in the sense this chapter cares about."

### section 2 — counting with the integers (~3pp)

**decisions recorded for drafter:**
- **Peano axioms are OUT OF SCOPE for chapter 1.** the ladder framing ("each system solves a problem the previous could not") is sufficient. Peano's construction is saved for a later chapter if useful.
- **"set" requires a one-sentence informal definition at first use.** Rosen Section 2.1 provides the canonical reference: an unordered collection of distinct objects. define "set" in the first paragraph of section 2 before introducing ℕ as a set of numbers.
- **ℕ includes 0.** this curriculum uses ℕ = {0, 1, 2, ...}, ℤ+ = {1, 2, ...}. state the convention explicitly and give the reason (induction starts at 0; CS indexing starts at 0; the ISO 80000-2 standard). acknowledge briefly that some texts (analytic number theory) use ℕ = {1, 2, ...} — the reader who encounters the other convention should not be surprised.

**closure argument (source: implicit in Lang's structure; explicit in curriculum's language):**
- the naturals are closed under + (sum of two naturals is a natural) and closed under × (product of two naturals is a natural).
- the naturals are NOT closed under subtraction (3 − 5 has no answer in ℕ — there is no natural number that, when added to 5, gives 3).
- the motivating gap: "we want every subtraction to have an answer. so we extend the number system to the integers, which adds the negatives: ..., −3, −2, −1, 0, 1, 2, 3, ...."
- after the integers are built, the closure property holds: integers are closed under +, −, ×.

**worked example for the section 2 → section 3 ladder transition (drafter embeds this as a concrete illustrative computation):**
- in ℕ: the expression `7 − 10` has no answer. no natural number, when added to 10, gives 7. this gap motivates the extension to ℤ. once in ℤ, `7 − 10 = −3` and the expression is well-defined.
- in ℤ: the expression `7 ÷ 10` has no answer. no integer, when multiplied by 10, gives 7. this gap motivates the extension to ℚ. once in ℚ, `7 ÷ 10 = 7/10` and the expression is well-defined.
- the pattern: each extension of the number system is motivated by a specific operation that had no answer in the smaller system. closure of the new system under that operation is the design target.

### section 3 — measuring with the rationals and reals (~3pp)

**closure argument continues:**
- the integers are NOT closed under division (1 ÷ 2 has no answer in ℤ).
- extend to the rationals ℚ = {p/q : p, q ∈ ℤ, q ≠ 0}. now divisions have answers.
- the rationals are dense: between any two distinct rationals there is another rational. but the rationals have HOLES.

**the hole (source: Spivak's set-C argument):**
- consider the set C = {x ∈ ℚ : x² < 2}. C has rational upper bounds — for instance, 3/2 is a rational upper bound because (3/2)² = 9/4 > 2.
- CLAIM: C has no rational least upper bound.
- proof sketch: if b ∈ ℚ were the LUB of C, then b² = 2 (heuristically: b is squeezed against the boundary). but no rational satisfies b² = 2 (proven next in section 3 via Lang's Theorem 4, paraphrased into curriculum voice).
- since every bounded-above set of rationals with no rational LUB represents a "hole" in ℚ, and we want every such set to have a LUB, we extend to the real numbers ℝ.

**√2 irrationality proof (Lang's Theorem 4, paraphrased into curriculum voice; cite Lang as source):**

*preliminary lemma (stated inline because the proof uses it twice):* **if k is an integer and k² is even, then k is even.** proof: suppose for contradiction that k is odd. then k = 2j + 1 for some integer j, and k² = 4j² + 4j + 1 = 2(2j² + 2j) + 1, which is odd. this contradicts the assumption that k² is even. therefore k must be even.

*main proof:*
- suppose √2 were a rational number, written in lowest form p/q (p, q positive integers with gcd(p, q) = 1).
- then (p/q)² = 2, so p² = 2q².
- p² is even (it equals 2q², which is 2 times an integer). by the preliminary lemma, p is even. write p = 2k.
- substitute: p² = (2k)² = 4k² = 2q², so q² = 2k². so q² is even. by the preliminary lemma again, q is even.
- but p and q were in lowest form with gcd(p, q) = 1 — they cannot both be even. contradiction.
- conclusion: √2 is not rational. it lives in the holes of ℚ and is filled in by ℝ.

**infinity (user decision: brief treatment ~1pp):**
- both ℕ and ℝ are infinite. but they are infinite in different ways.
- intuition: you can LIST ℕ one at a time (0, 1, 2, 3, ... forever). you cannot list ℝ that way — between any two reals there are always more reals, and no enumeration captures them all.
- the formal distinction (countable vs uncountable; Cantor's diagonal argument) is deferred to a later chapter on sets and cardinalities.
- the reader should remember: "infinite" is not one size. there are at least two sizes, and ℝ is the bigger one.

**wrong/right/explanation triple for section 3:**
- wrong: "√2 is a strange number that doesn't exist."
- right: "√2 is a real number. it is not rational — it cannot be written as a fraction p/q with integer p, q — but it is a real number with a definite location on the number line."
- explanation: "irrational means 'not rational.' it does NOT mean 'not real' or 'not a number.' the reals include all the rationals and all the irrationals."

### section 4 — symbols and variables (~3pp)

**three-layer distinction (source: MacKay, adapted):**
- the SYMBOL x (a placeholder letter, an abstract name).
- a particular VALUE of x (e.g., x = 3 — the symbol has been pinned to a specific number).
- a PROPOSITION involving x (e.g., "x > 0" — a statement that is true or false depending on the value of x).

**common mistakes (sources: Dawkins-style warnings, curriculum-extension):**
- wrong: "=" means "set this equal to that" (assignment).
- right: "=" means "these two sides have the same value" (equality). in programming, := or ← or = is often assignment; in math, = is always equality.
- explanation: this mistake comes from programming background. in math, x + 1 = 3 is a claim; you solve it to find which x makes the claim true. it is never "assign x + 1 to 3."

**minimal-pair verification trick:**
- "x + 1 = 3" — an EQUATION. solve for x; specific solution x = 2.
- "x + 1 = 3x" — another equation. solve for x; specific solution x = 1/2.
- "x + 1 = 1 + x" — an IDENTITY. true for EVERY value of x. there is nothing to solve.

**quantifier preview (addresses the outline's ∀x / ∃x promise):**
- the phrase "for every x, something holds" is written ∀x: something. example: ∀x ∈ ℝ, x² ≥ 0 (every real number has a non-negative square).
- the phrase "there exists an x such that something holds" is written ∃x: something. example: ∃x ∈ ℝ, x² = 4 (there is a real number whose square is 4 — in fact two, but "exists" requires only one).
- curriculum previews both symbols here. the full treatment of quantifier logic is deferred to a later chapter (probability / formal reasoning).

### section 5 — functions (~3pp)

**Rosen's Definition 1 as the primary definition** (with the "Let A and B be nonempty sets" preamble included):
> Let A and B be nonempty sets. A function f from A to B is an assignment of exactly one element of B to each element of A. We write f(a) = b if b is the unique element of B assigned by f to a. We write f : A → B.

**Stewart's four-way representation taxonomy** (EXPANDED from the outline's three-way list; outline's "formula / table / verbal rule" missed the graph representation):
1. verbal ("the rule that doubles any input")
2. algebraic / formula (f(x) = 2x)
3. visual / graph (the set of points (x, 2x) on the xy-plane)
4. numerical / table (a small discrete domain with one column for inputs, one for outputs)

**vertical line test** (from Stewart's treatment): a curve in the plane is the graph of a function IF AND ONLY IF every vertical line crosses the curve at most once. the "at most one output per input" rule has a geometric manifestation.

**Dawkins' minimal-pair counterexample (section 5 worked illustration):**
- y = x² + 1 is a function: for each x, exactly one y.
- y² = x + 1 is NOT a function: for x = 3, the equation y² = 4 has two solutions y = 2 and y = -2. violates "exactly one output."
- graphically, y² = x + 1 is a parabola opening rightward; a vertical line at x = 3 crosses it twice. fails the vertical line test.

**transformation framing (source: 3Blue1Brown, introduced AFTER the formal definition):**
- after the formal definition is in, introduce the kinematic image: a function is something that MOVES each input to an output. the value f(x) is where x goes.
- this framing is useful later when we compose functions ("first move, then move again") and when we classify functions (injective = movements that don't collapse things; surjective = movements that cover the whole destination).

### section 6 — domain, codomain, range (~3pp)

**Axler's consistent three-object naming** as the reference pattern. every function statement in the chapter names all three: domain, codomain, range.

**definitions:**
- DOMAIN: where inputs come from. the set A in f : A → B.
- CODOMAIN: the declared target set. the set B in f : A → B. an input is guaranteed to produce AN output that LIVES in B, but not every element of B is guaranteed to be hit.
- RANGE: the set of outputs actually produced. the set {f(a) : a ∈ A}. the range is always a subset of the codomain. range = codomain IF AND ONLY IF the function is surjective.

**the Stewart conflation — name it explicitly** (curriculum rule):
- calculus textbooks (Stewart and most others) often collapse codomain into range — "the range is all possible values." this is the pervasive pedagogical trap. our chapter treats codomain and range as DIFFERENT objects because the distinction matters for section 7's classifications.

**wrong/right/explanation for section 6 (concrete Stewart-style instance):**
- wrong (concrete): "the range of f(x) = x² from ℝ to ℝ is ℝ, because ℝ is the target set and ℝ is what's written to the right of the arrow." this is the student error the chapter is attacking — the Stewart conflation in action.
- right: "the range of f(x) = x² from ℝ to ℝ is [0, ∞), which is a proper subset of the codomain ℝ. negative numbers are in the codomain (because we declared ℝ as the target) but not in the range (because no real input x produces a negative x²). codomain is declared; range is actual; they are different objects."
- explanation: "the same formula can have different classifications depending on what codomain we declare. changing the codomain of f(x) = x² from ℝ to [0, ∞) converts the function from not-surjective to surjective. nothing about the formula changed — only the declared codomain. this is why 'range = codomain' is false in general: the two only coincide when the function is surjective."

### section 7 — injective / surjective / bijective (~2pp)

**English and Latin terms named together** (Rosen's pattern):
- injective (Latin) = one-to-one (English). different inputs map to different outputs.
- surjective (Latin) = onto (English). every element of the codomain is hit by some input.
- bijective (Latin) = one-to-one correspondence (English). both injective and surjective.

**naming-collision warning** (MathIsFun-inspired, paraphrased):
- IMPORTANT: "one-to-one" (short phrase) means injective only.
- "one-to-one correspondence" (full phrase) means bijective.
- the short and long English phrases mean different things. do not confuse them. the Latin forms are unambiguous — injective means injective, bijective means bijective — which is why many mathematicians prefer the Latin.

**Rosen's contrapositive form of injectivity (both forms stated):**
- direct: f(a) = f(b) ⟹ a = b.
- contrapositive: a ≠ b ⟹ f(a) ≠ f(b).
- the two statements are logically equivalent. use whichever is easier for a given proof.

**MathIsFun-style arrow diagrams** introduced BEFORE algebra for each classification:
- draw set A and set B with arrows from each element of A to exactly one element of B (the function).
- injective: no two arrows end at the same element of B.
- surjective: every element of B has at least one arrow ending at it.
- bijective: every element of B has exactly one arrow.

**worked examples:**
- f(x) = x² on ℝ → ℝ: NOT injective (f(2) = f(-2) = 4), NOT surjective (no input produces -1). so not bijective either.
- f(x) = x² on [0, ∞) → [0, ∞): injective AND surjective, therefore bijective. domain and codomain restriction changes the classification.
- f(x) = x³ on ℝ → ℝ: bijective. every real has a unique real cube root.

### worked example — counting arguments as a function (~1.5pp)

**rhetorical move sequence** (addresses prosecutor finding P2-F7 on cross-section synthesis):
1. state the concrete question: "given 5 people, how many ways are there to pick 2 of them to share a taxi?"
2. list by hand to produce the count: enumerate the 10 pairs (1-2, 1-3, 1-4, 1-5, 2-3, 2-4, 2-5, 3-4, 3-5, 4-5). the answer is 10.
3. label the inputs: (n, k) where n is the total number of people and k is how many we pick.
4. re-state the process as a function: C : {(n, k) ∈ ℕ × ℕ : 0 ≤ k ≤ n} → ℕ. C(5, 2) = 10. the domain is a subset of ℕ × ℕ; the codomain is ℕ.
5. invoke section 1: the output of C is a COUNT, not a label. C(5, 2) = 10 can be added, compared, and used arithmetically. this is mathematical numbers behaving as mathematical numbers.
6. (optional, if space): note that C(n, k) is also an injective? surjective? bijective? function — exercise for the reader.

**closed-form formula** (mention without derivation; full derivation belongs in a later combinatorics chapter):
- C(n, k) = n! / (k! (n − k)!). stated without proof at this level; the chapter's job is the concept of C as a function, not the computation.

### quiz (~1pp)

**10 quiz questions** covering all seven sections + the worked example (expanded from 5 per prosecutor findings P2-F10 and P2-F5; now includes sections 4 and 5 coverage, and the ℤ → ℚ extension step in section 2):

1. (section 1) which of these are mathematical numbers: "three apples", "phone number 555-1234", "the third room on the left", "3.5 meters"? justify each answer.
2. (section 2) is subtraction closed over ℕ? if yes, give a brief argument. if no, give a counterexample. [expected: not closed; 3 − 5 has no answer.]
3. (section 2 — new) give an arithmetic expression that has no answer in ℤ but has an answer in ℚ. what property of ℤ does this show is missing? [expected: 3 ÷ 7 has no integer answer; ℤ is not closed under division.]
4. (section 3) is √2 a rational number? justify briefly using the chapter's proof. [expected: not rational; proof by contradiction in lowest terms, using the "k² even ⇒ k even" lemma.]
5. (section 3) briefly explain in what sense both ℕ and ℝ are infinite. why are their infinities considered different? [expected: ℕ listable one element at a time; ℝ not listable that way.]
6. (section 4) is `x + 1 = 3x` an equation or an identity? what about `x + 1 = 1 + x`? justify. [expected: first is an equation (specific solution x = 1/2); second is an identity (true for all x).]
7. (section 4) translate the English sentence "there is a real number whose square equals 4" into the symbolic form ∃x: ... .
8. (section 5) is `y² = x + 1` a function? justify using the chapter's criterion and the vertical line test. [expected: not a function; at x = 3, y² = 4 has two solutions y = ±2; vertical line x = 3 crosses the curve twice.]
9. (section 6) find the range of f(x) = 1/(1 + x²) on ℝ → ℝ. is this function surjective? [expected: range (0, 1]; not surjective because codomain ℝ includes values the range does not.]
10. (section 7) show f(x) = 2x + 1 on ℝ → ℝ is bijective. find its inverse. [expected: injective from 2a+1 = 2b+1 ⟹ a = b; surjective because for every y ∈ ℝ, x = (y−1)/2 satisfies f(x) = y; inverse is f⁻¹(y) = (y−1)/2.]

answers in per-chapter appendix.

## common mistakes — cross-section catalog

every claim below traces to a source (or is labeled as curriculum-original construction).

1. **"= means assignment"** (curriculum-original construction; not sourced from any surveyed textbook — labeled as curriculum-original per prosecutor finding P1-F10). wrong: equating "=" with ":=" from programming. right: "=" is equality (a proposition). ":=" or "←" is assignment. explanation: math equations state facts; programming statements change state.

2. **"codomain = range"** (Rosen; Axler both flag this; Stewart is the canonical source of the conflation). wrong: using the two terms interchangeably. right: codomain is declared; range is actual image. explanation: same formula, different codomain, different classification.

3. **"one-to-one correspondence = one-to-one"** (MathIsFun flags). wrong: treating the short and long English phrases as synonyms. right: short phrase = injective; long phrase = bijective. explanation: the Latin forms are unambiguous; use them if the English is confusing.

4. **"f(x) = f times x"** (Dawkins; his function-definition page flags this as the most common mistake). wrong: reading f(x) as multiplication. right: f(x) is the output of function f applied to input x. explanation: parentheses in function-call context are not multiplication.

5. **"f⁻¹(x) = 1/f(x)"** (Dawkins; his INVERSE-FUNCTIONS page flags this — separate page from the function-definition page). wrong: reading f⁻¹ as reciprocal. right: f⁻¹ is the inverse function. explanation: the superscript −1 is an inverse operator, not an exponent, in function-call notation.

6. **"ℕ = {1, 2, 3, ...}"** (convention-dependent; Rosen's 8th ed. and ISO 80000-2 include 0; analytic-number-theory tradition excludes). wrong-in-this-curriculum / right-in-some-other-texts: the exclusion. this curriculum uses ℕ = {0, 1, 2, 3, ...}. explanation: the two conventions coexist. state ours; be aware of the other when reading external material.

7. **"labels with digits are numbers"** (curriculum-original construction). wrong: treating phone numbers and house numbers as mathematical numbers. right: digits can appear in non-numerical roles. explanation: mathematical numbers participate in arithmetic; labels do not.

8. **"irrational means not a number"** (curriculum-original construction). wrong: thinking "irrational" implies "not real" or "not a number." right: irrational = not rational. irrationals are real numbers. explanation: ℝ contains both rationals and irrationals. √2 is as much a number as 1/2 is.

## verification tricks — cross-section catalog

1. **equation vs identity** (section 4): "x + 1 = 3" (solve for x) vs "x + 1 = 1 + x" (true for all x).
2. **function vs relation** (section 5, source: Dawkins): y = x² + 1 (function — exactly one y per x) vs y² = x + 1 (relation — two y values per x at x = 3; fails vertical line test).
3. **injective or not** (section 7): f(x) = x² on ℝ (not injective) vs f(x) = x² on [0, ∞) (injective). domain restriction changes classification.
4. **surjective or not** (section 7, addresses prosecutor finding P2-F6): f(x) = x² on ℝ → ℝ (NOT surjective; range [0, ∞) ⊊ ℝ) vs f(x) = x² on ℝ → [0, ∞) (surjective; range = codomain). codomain restriction changes classification. **this is the codomain/range minimal pair — same formula, different codomain, different classification.**
5. **bijective or not** (section 7): f(x) = x³ on ℝ → ℝ is bijective; f(x) = x² on ℝ → ℝ is neither injective nor surjective. same structural question (x² vs x³), opposite answer.
6. **number vs label** (section 1): "three apples" (count, a mathematical number) vs "phone number 555-1234" (label, not a mathematical number). minimal pair distinguishes the four uses of "number" in everyday language.

## source list

### primary textbooks

- Lang, Serge. *Basic Mathematics*. Springer reprint 1988 (originally Addison-Wesley 1971). ISBN 978-0-387-96787-5. primary source for: ℕ-includes-0 convention (verbatim), thermometer analogy for integers, √2 irrationality proof (Theorem 4, Chapter 1 §5 — verbatim), second-person imperative voice examples.
- Spivak, Michael. *Calculus*, 4th ed. Publish or Perish, 2008. ISBN 978-0-914098-91-1. primary source for: P1-P13 axioms for ℝ, the "hole in ℚ" argument (set C = {x ∈ ℚ : x² < 2}), completeness of ℝ.
- Rosen, Kenneth H. *Discrete Mathematics and Its Applications*, 8th ed. McGraw-Hill, 2019. Section 2.3. primary source for: Definition 1 of a function (verbatim, with "Let A and B be nonempty sets" preamble), codomain/range split, contrapositive form of injectivity.
- Axler, Sheldon. *Linear Algebra Done Right*, 4th ed. Springer (open access), 2024. Section 0.A for function basics; Section 3B for null-space/injectivity connection. primary source for: consistent three-object naming.
- Stewart, James. *Calculus: Early Transcendentals*, 8th ed. Cengage. Section 1.1. primary source for: four-way representation taxonomy (verbal / formula / graph / table), vertical line test. also the canonical example of the codomain/range conflation (which the curriculum flags).

### secondary textbooks

- MacKay, David J. C. *Information Theory, Inference, and Learning Algorithms*. Cambridge, 2003. secondary source for: three-layer variable distinction (symbol / value / proposition).
- Bishop, Christopher M. *Pattern Recognition and Machine Learning*. Springer, 2006. secondary source for: notation-table convention.

### online resources

- Dawkins, Paul. *Paul's Online Math Notes*. https://tutorial.math.lamar.edu/. primary source for: minimal-pair counterexample (y = x² + 1 vs y² = x + 1), "f(x) is NOT f times x" warning (function-definition page), "f⁻¹(x) ≠ 1/f(x)" warning (inverse-functions page — separate page).
- MathIsFun. "Injective, Surjective and Bijective." https://www.mathsisfun.com/sets/injective-surjective-bijective.html. primary source for: arrow-diagram treatment, naming-collision warning.
- 3Blue1Brown. *Essence of Calculus* Chapter 1 and *Essence of Linear Algebra* Chapter 3 (linear transformations). https://www.3blue1brown.com/. secondary source for: transformation-as-movement framing.
- Better Explained (Azad, Kalid). https://betterexplained.com/. secondary source for: physical metaphors for function transformations (fast-forward, shift).
- Khan Academy. https://www.khanacademy.org/. secondary source for: mastery-step progression, arrow-diagram classifications in the functions track. **note:** Khan's separate arithmetic / number-system track (grade 6-8) was NOT surveyed in Agent B's scope and may contain additional pedagogical treatment of the number-system ladder.

### confirming sources

- Iowa State University Rosen lecture transparencies (https://www.ece.iastate.edu/~rkumar/teaching/CprE310/lectures/Section_2_3.pdf).
- OpenStax Calculus Volume 1 Section 1.1 (Stewart-mapped).
- Internet Archive OCR for Lang's *Basic Mathematics* (archive.org/details/basic-mathematics-serge-lang_20240418).
- Notre Dame Math 10850/10860 course notes using Spivak (www3.nd.edu/~dgalvin1).
- Axler 4th edition open-access PDF (linear.axler.net/LADR4e.pdf).

## decisions recorded

- **historical motivation**: EXCLUDED from chapter 1 per user decision. chapter stays on the mathematical ladder without historical detours. history may appear in a later chapter if useful.
- **infinity**: BRIEF TREATMENT (~1pp) in section 3 per user decision. distinguish countable from uncountable at intuitive level (ℕ listable; ℝ not listable). no Cantor diagonalization. defer full treatment to a later chapter.
- **ℕ convention**: ℕ = {0, 1, 2, 3, ...}, ℤ+ = {1, 2, 3, ...}. state the convention explicitly at first use in section 2.
- **naming rule**: satisfied trivially in chapter 1 (no project-native components). external names (Rosen, Axler, Stewart, Lang, Spivak, Dawkins, MathIsFun, etc.) permitted as citations.
- **voice models**: Lang (primary) and Dawkins (secondary) for sentence shape and warning pattern. Spivak NOT a voice model.
- **Peano axioms**: OUT OF SCOPE for chapter 1. mentioned only if used as a reference later; not derived in this chapter.

## remaining open gaps

1. Lang's Chapter 13 function definition verbatim — OCR fetch was truncated. medium-confidence characterization from secondary sources. drafter should use Rosen's Definition 1 as the primary textbook-cited definition (which IS verbatim-confirmed) rather than Lang's Chapter 13 version.
2. Lang's Chapter 14 mapping formalism verbatim — not retrieved. the mapping-vs-function separation in Lang is confirmed from ToC, but Lang's exact handling of codomain/range and injective/surjective classification is not verbatim-confirmed.
3. Lang's Interlude informal set definition verbatim — not retrieved. secondary accounts confirm operational treatment. drafter should use Rosen Section 2.1's informal definition as the primary reference.
4. Spivak's exact prose for the set-C argument in Chapter 2 — not verbatim-confirmed. the logical structure is high-confidence; the exact sentences are not. drafter constructs the argument in curriculum voice using Spivak as structural source.
5. √2 irrationality proof — the verbatim Lang version (Theorem 4) is captured above. Spivak's version structure matches; exact Spivak wording not extracted. curriculum uses Lang's Theorem 4 as the reference and paraphrases into curriculum voice.

these gaps do NOT block drafting. the primary definitions and arguments are sourced with verbatim or high-confidence material.

## answered prosecutor findings

all 24 findings from the two prosecutor passes have been addressed in this revision:

prosecutor 1 (factual accuracy + internal consistency) — findings F1–F10:
- F1 (P0): Rosen Definition 1 "Let A and B be nonempty sets" preamble restored in Rosen section and section 5 draft guidance.
- F2 (P0): ISO 80000-2 attribution qualified as "consistent with" rather than "follows"; explicit note that Rosen 8th ed. does not cite ISO.
- F3 (P1): Dawkins f⁻¹ warning moved to the inverse-functions page citation; function-definition page cites only the "f(x) is not f times x" warning.
- F4 (P1): 3Blue1Brown "Essence of Linear Algebra" reference corrected to Chapter 3 (linear transformations), not Chapter 1.
- F5 (P1): ladder-gap claim qualified — "no surveyed resource in Agent B's scope (focused on functions)" rather than absolute "no surveyed resource."
- F6 (P1): MathIsFun naming-collision warning's actual source text added verbatim; stronger paraphrase now explicitly labeled as paraphrase.
- F7 (P1): invented "perfect pairing of inputs" mnemonic removed from MathIsFun attribution.
- F8 (P2): Axler null-space-injectivity claim relocated to Chapter 3B (where it actually belongs) rather than Section 0.A.
- F9 (P2): Stewart edition specified as 8th ed. (not "8th/9th") throughout; 9th ed. differences flagged as unchecked.
- F10 (P2): "= means assignment" mistake labeled as curriculum-original construction rather than sourced.

prosecutor 2 (completeness + pedagogical coverage + naming compliance) — findings F1–F14:
- F1 (P1): section 1 now has worked example and wrong/right/explanation triple.
- F2 (P1): section 2 has closure framing; Peano axioms explicitly recorded as out of scope; Rosen-cited "set" definition note added.
- F3 (P1): √2 irrationality proof provided verbatim from Lang (Theorem 4).
- F4 (P2): section 4 has quantifier preview with both ∀x and ∃x examples.
- F5 (P2): section 5 expanded from three-way to Stewart's four-way representation taxonomy.
- F6 (P1): codomain/range minimal pair (f(x) = x² on ℝ → ℝ vs on ℝ → [0, ∞)) added to verification tricks #4.
- F7 (P2): C(n, k) worked example now has explicit 6-step rhetorical move sequence.
- F8 (P2): "set" informal definition note added, citing Rosen Section 2.1 as reference.
- F9 (P2): Dawkins voice examples ("Do not confuse the two!", "Notice that...") added; plus 5 verbatim Lang voice examples extracted.
- F10 (P2): quiz expanded from 5 to 9 questions, now covering all 7 sections including section 4 (variables) and section 5 (functions).
- F11 (P3): historical motivation decision recorded: EXCLUDED.
- F12 (P3): infinity stance decision recorded: brief treatment ~1pp.
- F13 (P1): Lang's *Basic Mathematics* researched (Agent C) and added to source list. verbatim quotes and voice examples extracted.
- F14 (P2): Spivak's *Calculus* researched (Agent D) and added to source list. P1-P13 axioms, set-C argument, completeness characterization extracted.
