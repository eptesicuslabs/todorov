# The Brain in One Page

This is the 80/20 overview. It covers the 20% of neuroscience that gives you 80% of the understanding you need to design brain-inspired architectures.

## The Hardware

The human brain has about 86 billion neurons connected by roughly 100 trillion synapses. It runs on about 20 watts -- less than a laptop charger. For comparison, a 300M parameter transformer has 300 million "neurons" and uses about 200 watts per GPU. The brain is 286 times larger and 10 times more power-efficient.

This efficiency gap is not about cleverness in software. It is about physics. Biological computation uses 10 femtojoules per synaptic operation. Silicon uses 4.6 picojoules per floating-point multiply at 45nm (and 0.3 picojoules at 5nm). Ternary operations cost about 0.001 picojoules at 5nm -- closing the gap by 200-350x.

## The Neuron

A neuron is a cell that receives electrical inputs, integrates them over time, and fires a spike if the total exceeds a threshold.

Think of a ReLU unit in a neural network. It takes a weighted sum of inputs and outputs zero if the sum is negative, or the sum itself if positive. A biological neuron does something similar but with two critical differences. First, it has a temporal state called the **membrane potential** (the voltage across the cell membrane, typically around -65 millivolts at rest). Inputs accumulate in this state over time, like a leaky bucket that fills with each input and slowly drains between inputs. Second, its output is all-or-nothing: when the membrane potential crosses a threshold, the neuron fires a **spike** (also called an **action potential**), a brief ~1ms electrical pulse. Then the membrane potential resets and the neuron is briefly unable to fire again. The simplest mathematical model of this behavior is the leaky integrate-and-fire (LIF) neuron.

The spike is the fundamental unit of neural communication. It is binary: fire or not fire. The neuron does not output a continuous value like a transformer activation. Information is carried by which neurons spike, when they spike, and how often they spike. In Todorov, ternary spikes {-1, 0, +1} extend this to three states: inhibit, silent, or excite.

A neuron has three main parts. The **dendrites** (tree-like branches) receive inputs from other neurons. The **soma** (cell body) integrates these inputs and decides whether to spike. The **axon** (long cable) carries the spike to other neurons. Recent research shows that dendrites are not passive cables -- they perform their own local computations, acting like a small neural network within the neuron.

For ML engineers, the key takeaway: a biological neuron is an RNN cell with binary output. It has persistent state (membrane potential), a nonlinear threshold (spike), and temporal dynamics (leak, refractory period). This is fundamentally different from a feedforward ReLU unit, which has no memory of previous inputs.

## The Synapse

A **synapse** is the connection point between two neurons. One neuron (presynaptic) sends a spike down its axon. When the spike arrives at the synapse, it triggers the release of chemical **neurotransmitters** (signaling molecules). These cross a tiny gap and bind to receptors on the receiving neuron (postsynaptic), causing a change in its membrane potential.

In ML terms, a synapse is a weight in a weight matrix. But biological weights are different in two ways. First, they change based on activity during inference, not just during training. **Short-term plasticity** (lasting milliseconds to minutes) alters synaptic strength based on recent firing patterns. A synapse can get temporarily weaker (depression) or stronger (facilitation) depending on how fast the presynaptic neuron has been firing. Second, long-term weight changes follow local rules, not global backpropagation. The most important is **Hebbian learning**: "neurons that fire together wire together." If neuron A repeatedly causes neuron B to fire, the synapse from A to B gets stronger. Mathematically, this is an outer product: Delta_w = eta * x_pre * x_post.

Synapses come in two types. **Excitatory** synapses (using glutamate as their neurotransmitter) push the postsynaptic neuron toward firing. **Inhibitory** synapses (using GABA) push it away from firing. A neuron is either excitatory or inhibitory -- it cannot switch. This is called Dale's law.

## The Cortex

The **cerebral cortex** is the thin (2-4 mm) wrinkled sheet of neurons covering the brain. It handles perception, language, planning, motor control, and most of what we call cognition. About 80% of cortical neurons are excitatory and 20% are inhibitory. This 80/20 ratio is remarkably consistent across brain regions and species.

The cortex is organized in six layers (L1-L6), stacked from the surface inward. Each layer has different cell types and connectivity patterns. Layer 4 receives sensory input from the thalamus. Layers 2/3 process and relay information between cortical areas. Layer 5 sends output to subcortical structures. Layer 6 sends feedback to the thalamus. This layered architecture is fundamentally different from transformer layers, which are functionally identical.

Neurons in the cortex are organized in **cortical columns** (vertical groups of neurons spanning all 6 layers). Neurons within a column tend to respond to similar features. The column is sometimes called the "repeating unit" of cortical computation, though this analogy is debated.

The cortex also has a repeating circuit motif called the **canonical microcircuit** (Douglas & Martin 1989). It consists of a recurrent excitatory loop between layers, stabilized by inhibitory neurons. This circuit amplifies weak inputs (like a gain control) while keeping activity bounded.

## How Information Flows

Sensory information enters the brain through specialized receptors (eyes, ears, skin). It passes through the **thalamus** (a relay station that also gates and filters signals) and reaches primary sensory cortex (V1 for vision, A1 for hearing). From there it flows through a hierarchy of cortical areas, from simple feature detectors to complex object representations.

The brain is bidirectional. Feedforward connections carry sensory data upward through the hierarchy. Feedback connections carry predictions and contextual information downward. The **predictive coding** framework (Rao & Ballard 1999) proposes that feedforward connections carry prediction errors, not raw data. Only the surprise -- the difference between what was predicted and what was received -- propagates upward.

In a transformer, the residual stream is the sole channel for inter-layer communication. Information flows forward through the layers and accumulates additively. There is no feedback, no prediction error, and no gating by the thalamus. The residual stream is a shared bus, not a predictive hierarchy.

## What Makes the Brain Different from a Transformer

**Sparsity.** At any given moment, only about 1-5% of cortical neurons are active. The rest are silent. This extreme sparsity is an energy constraint: each spike costs energy, so the brain minimizes the number of spikes. In a dense transformer layer, 100% of units are active for every input.

**Communication.** Neurons communicate with spikes -- discrete, all-or-nothing events. Transformers use continuous-valued activations. Spikes carry information in their timing and their identity (which neuron fired), not in their amplitude. This forces a different kind of information encoding.

**Memory.** Synaptic weights change during processing, not just during training. Short-term plasticity adapts synapses on millisecond timescales. The brain effectively rewrites its program while running. Transformer weights are fixed at inference time.

**Learning.** The brain uses local learning rules. Each synapse adjusts based on the activity of its two connected neurons, plus a global neuromodulatory signal (like dopamine for reward). There is no backpropagation of error gradients through the entire network.

**Power.** 20 watts for 86 billion neurons versus 200+ watts per GPU for 300 million parameters. The gap is real but narrowing: ternary operations on modern silicon approach biological efficiency per operation. The bottleneck is data movement, not compute.
