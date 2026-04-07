# consciousness simulations

## ignition_dynamics.py

simplified ignition model demonstrating the nonlinear threshold between local and global cortical activation (Dehaene & Changeux 2005).

network: 250 LIF neurons divided into 5 processor populations of 50 neurons each. within each processor, neurons are connected with probability 0.15 and synaptic weight 0.8 mV. between processors, neurons are connected with sparse long-range excitatory connections (probability 0.03, weight 0.3 mV). noise injected to all neurons (sigma = 2.0 mV). no inhibitory neurons (simplification; real ignition requires E/I balance, see the canonical microcircuit simulation for inhibitory stabilization).

experiments:
1. weak vs strong stimulus: processor 0 receives stimulus current during 200-350 ms. weak stimulus (8 mV) produces only local activation -- processor 0 fires but activity does not spread to other processors. strong stimulus (18 mV) triggers ignition -- activity spreads to all 5 processors through long-range connections, producing a global activation pattern. output: spike rasters and firing rate traces for both conditions (ignition_dynamics.png, 2x2 panels: rate traces top, rasters bottom).
2. threshold sweep: stimulus strength swept from 4 to 25 mV across 15 levels. for each level, mean firing rate of non-stimulated processors is measured during the stimulus window. shows a nonlinear transition: below ~12 mV, non-local activation is near baseline; above ~12 mV, non-local activation increases sharply. the broadcast ratio (non-local / local firing rate) shows the same nonlinear transition. output: ignition_threshold.png (2-panel figure: firing rates and broadcast ratio vs stimulus strength).

dependencies: numpy, matplotlib, Brian2

output: ignition_dynamics.png (4-panel figure), ignition_threshold.png (2-panel figure)

run: `python ignition_dynamics.py`

relevance: demonstrates the core mechanism of the Global Neuronal Workspace hypothesis -- ignition. below a stimulus threshold, activity remains local to the stimulated processor (subliminal processing). above the threshold, long-range excitatory connections trigger a self-amplifying cascade that activates all processors (global broadcast / conscious access). the transition is nonlinear and approximates the all-or-none character predicted by the GNW model (Dehaene et al. 2014). this model is a simplification: real ignition involves NMDA-mediated slow excitation for feedback, inhibitory control for competition between representations, and thalamocortical loops for additional amplification. the todorov architecture has no analog to ignition -- every layer unconditionally writes to the residual stream with no threshold for broadcast.
