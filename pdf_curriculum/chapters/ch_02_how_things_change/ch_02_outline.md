# chapter 2 outline — how things change

status: awaiting user approval (2026-04-22). outline only. draft not yet begun.

## the question this chapter answers

what does it mean for a quantity to change, and how do we measure the rate of that change?

## why it matters

chapter 1 gave the objects. chapter 2 gives motion. every later chapter that talks about optimization, learning, dynamics, or sensitivity depends on derivatives. if you do not understand change here, gradient descent in chapter 22 and backpropagation in chapter 23 become symbol manipulation without meaning.

## prerequisites

- chapter 1 — what a number means

## proposed structure

~20 pages, concise but complete. the chapter should build from average change to instantaneous change before introducing compressed derivative rules.

### section 1 — change before notation (~2pp)

start with everyday change:
- distance over time
- temperature through the day
- money in an account

show the difference between total change and rate of change. introduce the idea of "change per unit" before any limit notation appears.

### section 2 — average change between two points (~3pp)

the secant-line idea:
- change in output over change in input
- slope as rise over run
- average velocity as a motivating example

worked examples:
- average change of `f(x) = x^2` from `x = 1` to `x = 3`
- average speed over a travel interval

### section 3 — why average change is not enough (~2pp)

show the problem with instantaneous questions:
- speed at one exact moment
- slope at one exact point

introduce the idea of shrinking the interval. this is the conceptual bridge to limits, but the chapter should keep the intuition primary and the formal limit notation secondary.

### section 4 — derivative from first principles (~4pp)

introduce the first-principles derivative:
- difference quotient
- limit as the increment goes to zero
- notation `f'(x)` and `df/dx`

core worked derivations:
- derivative of a constant
- derivative of `f(x) = x`
- derivative of `f(x) = x^2`

the reader should see at least one full expansion and cancellation line by line.

### section 5 — derivative rules as compression (~3pp)

once first principles are understood, introduce the common rules as compressed results:
- constant rule
- power rule
- sum rule
- constant-multiple rule

state clearly that the rules are not magic. they are summaries of repeated first-principles work.

### section 6 — the chain rule (~3pp)

composite functions:
- outside function and inside function
- why changing the inside changes the outside
- chain rule as rate times rate

worked examples:
- derivative of `(3x + 1)^2`
- derivative of `\sqrt{1 + x^2}` or another simple composite

### section 7 — partial derivatives (~2.5pp)

move from one-variable change to many-variable change:
- hold other variables fixed
- derivative with respect to one variable at a time
- notation `\partial f / \partial x`

worked example:
- compute all three partial derivatives of a simple three-variable function

### section 8 — what the derivative means geometrically and physically (~1.5pp)

tie the chapter together:
- tangent slope
- instantaneous velocity
- local sensitivity

this section should prepare the reader for chapter 6 (multi-dimensional change) and chapter 22 (gradient descent).

### worked example — a three-variable derivative walkthrough (~1pp)

use one function end to end, for example:
- define a simple `f(x, y, z)`
- compute one ordinary derivative after restriction
- compute partial derivatives
- interpret what each derivative means

### quiz (~1pp)

8-10 questions covering average change, first principles, derivative rules, chain rule, and partial derivatives.

## target length

~20 pages.

## source pointers

primary teaching sources to consult once the outline is approved:

- Spivak, *Calculus*, early chapters on derivative intuition and formalism
- Stewart, *Calculus: Early Transcendentals*, derivative-introduction chapters for beginner pacing
- 3Blue1Brown, *Essence of Calculus* chapter 1 for geometric intuition
- Dawkins, derivative-introduction pages for warning-pattern pedagogy

## naming-rule compliance

chapter 2 introduces no project-native architecture components. standard mathematical names such as derivative, chain rule, and partial derivative are ordinary mathematical terms, not project-native labels.

## expected worked-example topics (summary)

1. average change of `x^2` across an interval
2. first-principles derivative of `x^2`
3. chain-rule derivative of a simple composite function
4. partial derivatives of a three-variable function

## latex notes for the drafting phase

- use the shared chapter preamble
- one simple secant-to-tangent visual is allowed if it materially helps
- no code blocks
- diagrams should remain light; the chapter should lean on equations and worked algebra

## deliverable at end of step 1

this outline only. if approved, chapter 2 proceeds to research. no draft should begin before outline approval.

## awaiting user decision

- does this structure match what you want chapter 2 to cover?
- do you want the chapter to stay intuition-first, or should it become more formal earlier?
- should the geometric picture (secant to tangent) stay lightweight or become a larger visual emphasis?
