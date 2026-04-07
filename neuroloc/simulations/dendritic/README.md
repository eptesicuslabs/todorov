# dendritic computation simulations

## what these demonstrate

a two-compartment neuron model demonstrating BAC (backpropagation-activated calcium spike) firing. the simulation shows that burst firing requires COINCIDENT input to both compartments, while input to either compartment alone produces a weaker response.

this is the cellular mechanism proposed by Larkum et al. (1999) and Larkum (2013) as an organizing principle for cortical associations. see [[apical_amplification]] and [[dendritic_computation]].

## scripts

### multicompartment_neuron.py
- model: two-compartment (soma + apical dendrite) with Hodgkin-Huxley-style Na+/K+ channels at the soma and Ca2+ channels at the apical dendrite, coupled by an inter-compartmental conductance g_c
- output: bac_firing.png
- shows: three conditions in a 3x2 panel layout (soma voltage left, dendritic voltage right):
  1. bottom-up only: brief basal current injection produces a single somatic action potential. the backpropagating AP is visible in the dendritic compartment but does not trigger a calcium spike (insufficient apical depolarization).
  2. top-down only: sustained apical current injection depolarizes the dendritic compartment but is insufficient to drive somatic spiking (too much attenuation through the coupling conductance).
  3. coincident (BAC firing): basal and apical inputs arrive within a narrow temporal window. the somatic AP backpropagates to the dendritic compartment where it coincides with apical depolarization, triggering a dendritic calcium spike. the calcium spike propagates back to the soma, triggering a burst of action potentials.
- key parameters:
  - g_c = 0.5 mS/cm^2 (inter-compartmental coupling)
  - g_Ca = 3.0 mS/cm^2 (dendritic Ca2+ conductance)
  - g_Na = 50.0 mS/cm^2 (somatic Na+ conductance)
  - I_basal = 2.5 nA (bottom-up input amplitude)
  - I_apical = 1.8 nA (top-down input amplitude)
- requires: brian2, matplotlib, numpy

## how to run

    pip install brian2 matplotlib numpy
    cd neuroloc/simulations/dendritic
    python multicompartment_neuron.py

## parameters to vary

- g_c (coupling conductance): controls how well the two compartments communicate. low g_c creates more independent compartments; high g_c makes them behave as a single compartment, reducing the coincidence detection property.
- I_apical timing: shifting the apical input relative to the basal input changes BAC firing probability. the effective window is ~10-100 ms (Larkum et al. 1999).
- g_Ca (Ca2+ conductance density): controls the threshold and amplitude of dendritic calcium spikes. higher g_Ca makes calcium spikes easier to trigger (lower coincidence requirement).
- I_basal amplitude: must be strong enough to trigger a somatic AP that backpropagates to the dendrite. below threshold, no BAC firing is possible regardless of apical input.

## biological correspondence

- the somatic compartment represents the soma + basal dendrites of an L5 pyramidal neuron, receiving feedforward sensory input
- the dendritic compartment represents the apical trunk + tuft, receiving top-down feedback from higher cortical areas via layer 1
- BAC firing is the cellular mechanism proposed by Larkum (2013) for binding bottom-up evidence with top-down predictions/context
- the burst output (multiple somatic APs from a single BAC event) carries a qualitatively different signal from isolated spikes, enabling multiplexed communication
