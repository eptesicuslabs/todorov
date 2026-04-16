# Eugene Izhikevich

status: definitional. last fact-checked 2026-04-16.

Eugene M. Izhikevich is a computational neuroscientist known for developing the [[izhikevich_model]] (2003), a two-variable neuron model that reproduces 20+ cortical firing patterns from just 4 parameters at ~13 FLOPS per timestep.

currently CEO of Brain Corporation (San Diego), he previously led one of the largest cortical simulations ever performed (10^11 synapses, with Gerald Edelman, 2008).

key contributions:
- the Izhikevich neuron model (2003, IEEE Transactions on Neural Networks)
- "Which Model to Use for Cortical Spiking Neurons?" (2004) -- the standard cost-vs-fidelity comparison
- "Dynamical Systems in Neuroscience" (2007, MIT Press) -- textbook on bifurcation theory applied to neuron models
- large-scale thalamocortical simulation (2008, PNAS, with Edelman)

relevance to todorov: the Izhikevich model represents the best tradeoff between biological realism and computational cost among established neuron models. its 4-parameter formulation (a, b, c, d) demonstrates that diverse temporal dynamics can emerge from minimal parameterization -- a principle directly relevant to ATMN design, where 2 parameters (tau, threshold_log) may be too few to capture useful dynamics.
