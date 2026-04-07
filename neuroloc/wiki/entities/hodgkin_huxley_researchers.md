# Alan Hodgkin and Andrew Huxley

Alan Lloyd Hodgkin (1914-1998) and Andrew Fielding Huxley (1917-2012) were British physiologists who developed the first quantitative mathematical model of the action potential in 1952, based on voltage-clamp experiments on the squid giant axon (Loligo).

their model -- four coupled ODEs describing Na+, K+, and leak currents through the nerve membrane -- remains the gold standard for biophysical neuron modeling 70+ years later. they received the Nobel Prize in Physiology or Medicine in 1963 (shared with John Eccles).

key contributions:
- voltage-clamp technique for measuring ionic currents in isolation
- the concept of voltage-gated ion channels (before channel proteins were identified)
- the gating variable formalism (m, h, n) for describing channel kinetics
- quantitative prediction of action potential shape, conduction velocity, and refractory period

relevance to todorov: the [[hodgkin_huxley]] model is the theoretical ancestor of all spike neuron models used in this project. ATMN's ternary spike threshold is a radical simplification of the HH mechanism: the explosive Na+ positive-feedback loop (the m^3*h term) is collapsed into a hard threshold, and the K+-mediated repolarization (the n^4 term) is collapsed into a reset-by-subtraction.
