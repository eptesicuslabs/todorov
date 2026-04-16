# notation

status: definitional. last fact-checked 2026-04-16.

mathematical notation conventions used across the neuroloc wiki.

## general conventions

- lowercase bold for vectors: **x**, **h**, **q**, **k**, **v**
- uppercase bold for matrices: **S**, **W**, **A**
- subscript t for time index: x_t, h_t, S_t
- superscript for layer index: x^(l)
- hat for estimates: x_hat
- tilde for modified/transformed: x_tilde

## neuroscience variables

- V_m: membrane potential (mV)
- V_th: firing threshold (mV)
- V_rest: resting potential (mV)
- V_reset: reset potential after spike (mV)
- tau_m: membrane time constant (ms)
- tau_s: synaptic time constant (ms)
- g_L: leak conductance (nS)
- C_m: membrane capacitance (pF)
- I_ext: external input current (pA)
- w: adaptation variable
- delta_t: spike slope factor in AdEx (mV)

## plasticity variables

- w_ij: synaptic weight from neuron j to neuron i
- Delta_w: weight change
- A_+, A_-: STDP amplitudes
- tau_+, tau_-: STDP time constants (ms)
- theta_BCM: BCM sliding threshold

## todorov variables

- alpha: spike threshold scaling factor OR channel-wise forgetting rate
- beta: data-dependent write gate
- S_t: KDA recurrent state matrix (R^{d_k x d_v})
- c_kv: MLA compressed latent (R^{d_c})
- h_t: mamba3 state OR ATMN membrane potential
- theta: rotation angle (RoPE or data-dependent)

## units

all biological simulations use SI-compatible units as defined by brian2:
- time: ms (milliseconds)
- voltage: mV (millivolts)
- current: pA (picoamperes)
- conductance: nS (nanosiemens)
- capacitance: pF (picofarads)
- frequency: Hz (hertz)
- energy: pJ (picojoules) for per-operation energy comparisons

## algebraic notation

- G(p,q,r): clifford algebra with p positive, q negative, r degenerate basis vectors
- G(3,0,1): projective geometric algebra (3 euclidean + 1 degenerate)
- e_i: basis vectors
- e_ij: basis bivectors (e_i wedge e_j)
- wedge: outer (exterior) product
- dot: inner (contraction) product
- tilde: reverse of a multivector (reverses blade factor order)
