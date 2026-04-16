# basal ganglia

status: definitional. last fact-checked 2026-04-16.

**why this matters**: the basal ganglia implement a biologically grounded action selection system -- a control architecture that resolves competition among candidate actions through disinhibition rather than direct excitation. this is the brain's gating mechanism: deciding what gets through and what gets suppressed. in ML, autoregressive token selection, RLHF reward signals, and memory write gates all face the same core problem (select one, suppress the rest), but no standard architecture explicitly separates the control system from the computation it controls. todorov has no control system. the basal ganglia show what one looks like.

## summary

the **basal ganglia** (a set of interconnected subcortical nuclei involved in action selection, reinforcement learning, and habit formation) are a convergence zone where cortical action candidates compete for execution through a mechanism of **selective disinhibition** (releasing a target from tonic inhibition to permit activity, rather than directly exciting it). the output nuclei (**GPi/SNr**, the internal globus pallidus and substantia nigra pars reticulata) tonically inhibit the thalamus. to execute an action, the basal ganglia do not excite the thalamus -- they remove inhibition from the specific thalamic target corresponding to the selected action while maintaining or increasing inhibition on all others.

three pathways implement this selection:

- the **direct pathway** (Go): facilitates the selected action by inhibiting GPi/SNr, releasing the thalamus
- the **indirect pathway** (NoGo): suppresses competing actions by exciting GPi/SNr via a polysynaptic route
- the **hyperdirect pathway**: provides fast global suppression before selective facilitation begins

dopamine from the **substantia nigra pars compacta** (**SNc**) modulates the balance between Go and NoGo, encoding the reward prediction error that drives reinforcement learning (see [[dopamine_system]]).

## anatomy

the basal ganglia comprise five principal nuclei:

**striatum**: the input stage, subdivided into the **caudate nucleus** (associated with goal-directed behavior and cognitive control) and **putamen** (associated with motor execution and habit formation). the striatum receives glutamatergic input from nearly all cortical areas. ~95% of striatal neurons are **medium spiny neurons** (**MSNs**, GABAergic projection neurons with dense dendritic spines that integrate thousands of cortical inputs). MSNs are divided into two populations by receptor expression: D1-MSNs (direct pathway) and D2-MSNs (indirect pathway).

**globus pallidus externus (GPe)**: a relay nucleus in the indirect pathway. receives inhibitory input from D2-MSNs and sends inhibitory projections to the subthalamic nucleus and GPi. GPe neurons fire tonically at high rates (~60-80 Hz) and are transiently silenced by striatal input.

**subthalamic nucleus (STN)**: the only glutamatergic (excitatory) nucleus in the basal ganglia. receives input from cortex (hyperdirect pathway) and GPe (indirect pathway). sends broad excitatory projections to GPi/SNr, increasing thalamic inhibition. the STN's excitatory nature makes it the amplifier of the NoGo signal.

**GPi/SNr** (globus pallidus internus / substantia nigra pars reticulata): the output stage. tonically active GABAergic neurons that inhibit the thalamus at ~80-100 Hz. action selection occurs by selectively reducing this tonic inhibition. GPi handles motor output; SNr handles oculomotor and cognitive output.

**substantia nigra pars compacta (SNc)**: the dopamine source. ~400,000 dopamine neurons in humans project to the striatum via the nigrostriatal pathway. SNc dopamine encodes the reward prediction error signal (Schultz et al. 1997) that modulates D1 and D2 receptor-mediated plasticity (see [[dopamine_system]]).

## the three pathways

### direct pathway (Go)

    cortex -> striatum (D1-MSNs) --[inhibit]--> GPi/SNr --[disinhibit]--> thalamus -> cortex

cortical input activates D1-MSNs in the striatum. these GABAergic neurons inhibit GPi/SNr, which tonically inhibits the thalamus. inhibiting the inhibitor releases the thalamus: the selected thalamic target is disinhibited and can relay activity back to cortex, completing the cortico-basal ganglia-thalamo-cortical loop. the double negative (inhibit the inhibitor) is the core computational trick -- it allows precise, focal facilitation without requiring excitatory connections to the thalamus.

ML analog: the direct pathway is functionally equivalent to a write-enable gate. in todorov's KDA layers, the beta gate controls whether new information is written to the recurrent state. D1-MSN activation = beta > threshold = "write this."

### indirect pathway (NoGo)

    cortex -> striatum (D2-MSNs) --[inhibit]--> GPe --[disinhibit]--> STN --[excite]--> GPi/SNr --[inhibit]--> thalamus

cortical input activates D2-MSNs, which inhibit GPe. GPe tonically inhibits STN, so removing GPe inhibition disinhibits STN. the now-active STN sends broad excitatory drive to GPi/SNr, increasing thalamic inhibition and suppressing competing actions. the indirect pathway has three synapses (striatum -> GPe -> STN -> GPi/SNr) compared to the direct pathway's one (striatum -> GPi/SNr), making it slower -- the NoGo signal arrives after the Go signal, providing a temporal window for the selected action to escape suppression.

ML analog: the indirect pathway is a "don't write" signal. in KDA, beta near zero suppresses the memory update. more broadly, the indirect pathway functions like attention masking: suppressing all candidates except the selected one.

### hyperdirect pathway

    cortex -> STN -> GPi/SNr -> thalamus

the fastest pathway. cortex projects directly to STN, bypassing the striatum entirely. STN excites GPi/SNr broadly, producing global suppression of all thalamic targets. this occurs BEFORE the striatal pathways can act (STN activation latency ~10 ms vs striatal ~20-30 ms). the computational logic: first suppress everything (hyperdirect), then selectively release the winner (direct), while maintaining suppression on losers (indirect).

ML analog: global suppression before selective facilitation resembles the structure of [[selective_attention]] with masking -- first zero out all candidates, then unmask the selected ones.

## dopamine modulation

dopamine from SNc modulates the Go/NoGo balance through D1 and D2 receptors with opposing effects:

**D1 receptors** (on direct pathway MSNs): dopamine binding increases excitability. high dopamine strengthens the Go pathway, facilitating action execution. D1 activation also promotes **LTP** (long-term potentiation) at corticostriatal synapses, reinforcing the cortical input patterns that led to reward.

**D2 receptors** (on indirect pathway MSNs): dopamine binding decreases excitability. high dopamine weakens the NoGo pathway, reducing suppression of the selected action. D2 activation promotes **LTD** (long-term depression) at corticostriatal synapses, weakening input patterns associated with non-rewarded actions.

the net effect: dopamine simultaneously strengthens Go and weakens NoGo, biasing the system toward action. dopamine depletion (as in Parkinson's disease) produces the opposite: weakened Go, strengthened NoGo, resulting in akinesia (inability to initiate movement).

the phasic dopamine signal encodes the reward prediction error (Schultz et al. 1997): positive RPE (unexpected reward) produces a dopamine burst that reinforces the action just taken (D1 LTP + D2 LTD), while negative RPE (expected reward omitted) produces a dopamine dip that weakens the action (D1 LTD + D2 LTP). this implements the TD learning update rule directly in the circuit (see [[dopamine_system]] for the formal correspondence).

ML analog: dopamine RPE is the biological reward signal. in RLHF (reinforcement learning from human feedback), the reward model provides a scalar signal that modulates policy updates -- strengthening outputs that received positive feedback and weakening those that received negative feedback. the D1/D2 opponent mechanism is analogous to the sign-dependent update in policy gradient methods.

## action selection as competition

the basal ganglia solve the **action selection problem** (choosing one action from many candidates in a way that is decisive, context-sensitive, and modifiable by experience). the mechanism is competitive selection via disinhibition:

1. multiple cortical areas simultaneously project action candidates to different striatal populations
2. the direct pathway channels corresponding to each candidate compete to inhibit GPi/SNr
3. the candidate with the strongest corticostriatal drive wins the competition -- its GPi/SNr target is most strongly inhibited (most disinhibited at the thalamus)
4. the indirect and hyperdirect pathways suppress all other candidates
5. the winner is executed via thalamo-cortical feedback

this is a [[winner_take_all]] mechanism implemented through disinhibition rather than lateral inhibition. the key difference from cortical WTA: the basal ganglia select among ACTIONS (distributed cortical representations), not among local neurons. this makes the basal ganglia a centralized selection bottleneck -- all action candidates must pass through this gate regardless of their cortical origin.

ML analog: action selection in the basal ganglia is analogous to token selection in autoregressive generation. the vocabulary logits are the action candidates, softmax/argmax is the selection mechanism, and the selected token is the "disinhibited" action. but the analogy is shallow: the basal ganglia maintain persistent state (tonic inhibition, dopamine levels, learned corticostriatal weights) that biases selection over time, while token selection in transformers is stateless -- each selection is independent given the context.

## working memory gating

O'Reilly and Frank (2006) proposed that the prefrontal-basal ganglia circuit gates updates to working memory. the prefrontal cortex (PFC) maintains information through sustained activity (see [[selective_attention]]). the basal ganglia control WHEN this information is updated:

- **gate open** (direct pathway active): new information enters PFC, overwriting or updating the current contents
- **gate closed** (indirect pathway dominant): PFC maintains its current contents, resisting distraction

dopamine RPE trains the gating policy: gate openings that lead to reward are reinforced; gate openings that lead to errors are suppressed. this learned gating is critical for flexible cognition -- knowing WHEN to update working memory is as important as knowing WHAT to store.

ML analog: this is the closest biological parallel to KDA's beta gate. beta_t in the delta rule layer controls whether the current input is written to the matrix-valued recurrent state. high beta = gate open = update memory. low beta = gate closed = maintain memory. the difference: KDA's beta is a learned linear projection of the input, while biological gating is mediated by a separate anatomical circuit (basal ganglia) with its own plasticity rules (dopamine-modulated).

## sequence learning

Graybiel (1998) showed that the basal ganglia are critical for **chunking** (grouping individual actions into automated sequences that are executed as a unit). during early learning, individual actions are represented separately in the striatum. with practice, the representation shifts: the striatum fires at the beginning and end of the learned sequence, treating the entire chunk as a single action unit. this is the neural basis of habit formation.

the computational advantage of chunking: it reduces the action selection problem from N individual decisions to one decision (execute the chunk or not). this frees the cortex for higher-level planning while the basal ganglia handle routine execution.

## pathology as pathway imbalance

diseases of the basal ganglia reveal the consequences of disrupting the Go/NoGo balance:

**Parkinson's disease**: degeneration of SNc dopamine neurons depletes striatal dopamine. the direct pathway (Go) is weakened (loss of D1 facilitation) and the indirect pathway (NoGo) is strengthened (loss of D2 suppression). the result: excessive thalamic inhibition, producing bradykinesia (slowness), akinesia (inability to initiate movement), and rigidity.

**Huntington's disease**: selective degeneration of indirect pathway (D2) MSNs in the striatum. the NoGo pathway is weakened while the Go pathway is intact. the result: insufficient thalamic inhibition, producing chorea (involuntary, jerky movements) and impulsive behavior. the Go/NoGo imbalance is the mirror image of Parkinson's.

## challenges

1. **the dimensionality problem**: the striatum has ~100 million neurons in humans, but GPi/SNr has only ~150,000. the compression ratio is ~700:1. how can such a narrow bottleneck support the selection of complex, high-dimensional actions? one answer is that the basal ganglia select among action categories, not specific motor commands -- the details are filled in by cortex and cerebellum. but the mapping from striatal representation to GPi/SNr output channels is not well understood.

2. **beyond action selection**: the basal ganglia are involved in cognitive and emotional functions that do not fit the action selection framework cleanly. obsessive-compulsive disorder, addiction, and mood disorders all involve basal ganglia dysfunction, suggesting these circuits do more than select motor actions. the "action" in action selection may need to be broadened to include cognitive operations (updating beliefs, switching strategies), but this broadening risks making the theory unfalsifiable.

3. **the segregation assumption**: the classical model assumes D1-MSNs and D2-MSNs are cleanly segregated into direct and indirect pathways. recent single-cell transcriptomic and optogenetic studies show substantial overlap: many MSNs co-express D1 and D2 receptors, and some MSNs project to both GPe and GPi. this complicates the neat Go/NoGo dichotomy. the pathways may be more of a continuum than two discrete channels.

4. **learning without a teacher**: the actor-critic model (Joel et al. 2002) assigns the critic role to the ventral striatum and the actor role to the dorsal striatum. but the critic must learn the value function before it can generate useful RPE signals. how does the system bootstrap? early learning must rely on unconditioned reward signals, but the transition from innate to learned valuation is not well specified.

5. **the binding problem for actions**: if multiple basal ganglia loops (motor, oculomotor, prefrontal, limbic) operate in parallel, how are their outputs coordinated into coherent behavior? selecting an arm movement, a gaze shift, and a cognitive strategy simultaneously requires cross-loop coordination, but the anatomical evidence for inter-loop communication is sparse.

## key references

- Schultz, W., Dayan, P. & Montague, P. R. (1997). a neural substrate of prediction and reward. Science, 275(5306), 1593-1599.
- Graybiel, A. M. (1998). the basal ganglia and chunking of action repertoires. Neurobiology of Learning and Memory, 70(1-2), 119-136.
- Frank, M. J. (2005). dynamic dopamine modulation in the basal ganglia: a neurocomputational account. Journal of Cognitive Neuroscience, 17(1), 51-72.
- O'Reilly, R. C. & Frank, M. J. (2006). making working memory work: a computational model of learning in the prefrontal cortex and basal ganglia. Neural Computation, 18(2), 283-328.
- Joel, D., Niv, Y. & Ruppin, E. (2002). actor-critic models of the basal ganglia: new anatomical and computational perspectives. Neural Networks, 15(4-6), 535-547.
- O'Doherty, J. et al. (2004). dissociable roles of ventral and dorsal striatum in instrumental conditioning. Science, 304(5669), 452-454.
- Nambu, A. (2004). a new dynamic model of the cortico-basal ganglia loop. Progress in Brain Research, 143, 461-466.
- Redgrave, P., Prescott, T. J. & Gurney, K. (1999). the basal ganglia: a vertebrate solution to the selection problem? Neuroscience, 89(4), 1009-1023.

## see also

- [[dopamine_system]]
- [[neuromodulatory_framework]]
- [[selective_attention]]
- [[winner_take_all]]
- [[hebbian_learning]]
- [[thalamocortical_loops]]
