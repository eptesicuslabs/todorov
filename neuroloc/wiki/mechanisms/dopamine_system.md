# dopamine system

**why this matters**: the dopamine reward prediction error signal is the most direct biological implementation of the TD error from reinforcement learning -- it demonstrates that brains use a scalar broadcast signal for credit assignment, fundamentally different from backpropagation's per-parameter gradients.

## summary

**dopamine** (a catecholamine neurotransmitter that modulates reward processing, motivation, and motor control) neurons in the **ventral tegmental area** (**VTA**, a midbrain nucleus and primary source of dopamine for reward circuits) and **substantia nigra pars compacta** (**SNc**, the largest midbrain dopamine nucleus, critical for motor control) encode the **reward prediction error** (**RPE**, the difference between received and expected reward). Schultz, Dayan, and Montague (1997) demonstrated that phasic dopamine activity follows the formal structure of the temporal-difference (TD) error signal in reinforcement learning:

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

where r_t is the received reward, gamma is the discount factor, V(s) is the estimated value of state s, and delta_t is the RPE. dopamine neurons fire above baseline for positive RPE (unexpected reward), at baseline for zero RPE (expected reward), and below baseline for negative RPE (expected reward omitted). this three-way response profile is the most precisely quantified neuromodulatory signal in the brain.

the dopamine system projects broadly to the **striatum** (via the nigrostriatal pathway from SNc), **prefrontal cortex** (via the mesocortical pathway from VTA), and **nucleus accumbens** (via the mesolimbic pathway from VTA). through **D1** and **D2 receptor subtypes** (two classes of dopamine receptors with opposing effects on intracellular signaling), dopamine modulates synaptic plasticity, action selection, and motivational state across these targets.

## anatomy

### source nuclei

**ventral tegmental area (VTA)**: ~400,000-600,000 dopamine neurons in humans. projects primarily to nucleus accumbens (mesolimbic pathway) and prefrontal cortex (mesocortical pathway). encodes RPE for appetitive stimuli, aversion, and salience. recent connectomic work shows VTA also projects to dorsal striatum, blurring the traditional pathway boundaries.

**substantia nigra pars compacta (SNc)**: the largest midbrain dopamine nucleus. projects primarily to dorsal striatum (nigrostriatal pathway). critical for motor control and habit formation. degeneration of SNc neurons causes Parkinson's disease. SNc neurons also encode RPE but with more motor-related tuning than VTA neurons.

### receptor subtypes

**D1 receptors**: coupled to stimulatory G proteins (Gs/Golf). increase cAMP and PKA signaling. enhance synaptic potentiation (**LTP**, long-term potentiation, a lasting increase in synaptic strength) at active synapses. predominantly expressed on direct pathway **medium spiny neurons** (**MSNs**, the principal projection neurons of the striatum) in the striatum. these project to the internal globus pallidus / substantia nigra pars reticulata (GPi/SNr). activation of D1-expressing MSNs disinhibits thalamus, facilitating action execution.

**D2 receptors**: coupled to inhibitory G proteins (Gi). decrease cAMP signaling. facilitate synaptic depression (**LTD**, long-term depression, a lasting decrease in synaptic strength) at active synapses. predominantly expressed on indirect pathway MSNs, which project to the external globus pallidus (GPe). activation of D2-expressing MSNs increases inhibition of thalamus, suppressing competing actions.

the D1/D2 segregation creates an opponent system: dopamine simultaneously strengthens the winning action (via D1 LTP in direct pathway) and weakens competing actions (via D2 LTD in indirect pathway).

ML analog: this opponent D1/D2 system is the neurobiological substrate of reinforcement learning's policy update -- strengthening the chosen action while suppressing alternatives, analogous to softmax cross-entropy loss.

## mechanism

### the Schultz experiment (1997)

Schultz, Dayan, and Montague (1997) recorded from dopamine neurons in behaving macaques during a classical conditioning paradigm. the key findings:

**before learning** (no prediction): a juice reward elicits a phasic burst of dopamine activity. the animal has no prediction, so delta = r - 0 = r > 0.

**after learning** (correct prediction): the conditioned stimulus (CS, e.g., a light) that predicts the reward now elicits the phasic dopamine burst, while the reward itself elicits no response. the prediction has shifted to the CS: delta = r - V(s) = 0 at the time of reward delivery, but delta = V(s') - V(s) > 0 at the time of the CS.

**omission** (violated prediction): if the CS appears but the reward is omitted, dopamine activity drops BELOW baseline at the expected time of reward. delta = 0 - V(s) < 0.

this three-way pattern is precisely the TD error signal:

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

positive delta -> phasic burst -> strengthen actions and stimuli that preceded the reward
zero delta -> baseline activity -> no learning signal
negative delta -> phasic depression -> weaken actions and stimuli that preceded the omission

### temporal-difference learning

the TD(0) value update rule is:

    V(s_t) <- V(s_t) + eta * delta_t

where eta is the learning rate. dopamine RPE directly implements delta_t. the D1/D2 receptor system implements the sign-dependent effect:

- positive delta (dopamine burst): D1 receptors activate, potentiating synapses in the direct pathway (the action that led to reward is reinforced)
- negative delta (dopamine dip): D2 receptors dominate, depressing synapses in the indirect pathway (the action that led to omission is suppressed)

the value function V(s) is believed to be represented in the ventral striatum (nucleus accumbens) and orbitofrontal cortex. the policy (action selection) is computed in the dorsal striatum.

### phasic vs tonic dopamine

**phasic dopamine**: brief bursts (100-500 ms) encoding RPE. the teaching signal. operates on individual trials, driving credit assignment and policy updates.

**tonic dopamine**: sustained background level. sets the motivational state and response vigor. high tonic dopamine = high motivation, vigorous responding. low tonic dopamine = amotivation, slowing. tonic levels are modulated by prefrontal cortex and other inputs to VTA/SNc.

### distributional coding

Dabney et al. (2020) showed that individual dopamine neurons encode different quantiles of the reward distribution, not just the mean RPE. some neurons have low reversal points (optimistic -- they signal positive RPE for most outcomes). others have high reversal points (pessimistic -- they signal negative RPE for most outcomes). the population collectively encodes the full distribution of possible outcomes.

ML analog: this maps directly to **distributional reinforcement learning** (Bellemare et al. 2017), where the agent learns the full return distribution rather than just the expected value.

## plasticity modulation

dopamine gates [[hebbian_learning]] via **three-factor learning rules** (plasticity rules requiring presynaptic activity, postsynaptic activity, AND a neuromodulatory signal):

    Delta_w = eta * x_pre * x_post * DA

where DA is the dopamine signal. without dopamine, co-activation of pre and post neurons produces no lasting weight change. with dopamine, the standard Hebbian association is strengthened (DA > 0) or weakened (DA < 0). this solves the **credit assignment problem** (determining which synapses among millions were responsible for a given outcome): only synapses that were recently active AND that led to reward (dopamine burst) are strengthened.

ML analog: three-factor learning is analogous to reward-weighted policy gradient updates, where the gradient is scaled by a scalar reward signal.

the timing constraint is critical: dopamine must arrive within a narrow window (~1-2 seconds) after synaptic activity for the plasticity gate to work. this window is mediated by molecular **eligibility traces** (molecular tags at recently active synapses that mark them as eligible for modification) at the synapse. the eligibility trace decays exponentially, creating a temporal credit assignment mechanism.

## relationship to todorov

todorov has NO reward prediction error signal. the architecture is trained by backpropagation with cross-entropy loss, which provides a global error gradient distributed through all parameters simultaneously. this is fundamentally different from dopamine-gated plasticity, where:

1. dopamine is a SCALAR broadcast signal, not a per-parameter gradient
2. dopamine gates LOCAL plasticity (Hebbian * DA), not global parameter updates
3. dopamine signals REWARD prediction error, not sensory prediction error (though see [[predictive_coding]] for the precision-weighting interpretation)

the closest todorov analog to dopamine is the training loss itself: a scalar signal that modulates all parameter updates. but the loss is not used DURING inference (it has no forward-pass effect), whereas dopamine affects ongoing neural computation (via D1/D2 receptor modulation of excitability and synaptic transmission).

see [[neuromodulation_to_learning_and_gating]] for a detailed analysis of what a dopamine-like signal could look like in todorov.

## challenges

the RPE theory is the most successful computational account of any neuromodulatory system, but it has significant gaps:

1. **ramping dopamine**: some dopamine neurons show sustained ramping activity as an animal approaches a reward, not just a phasic burst at the moment of prediction error. this is inconsistent with simple TD error but may reflect a motivational or temporal proximity signal.

2. **aversive responses**: some VTA neurons respond to aversive stimuli (punishment, threat). this is debated: is it salience coding (unsigned prediction error) or a separate aversive pathway? the picture is messier than "dopamine = reward."

3. **movement signals**: SNc dopamine neurons respond during movement initiation even in the absence of reward. this motor function is difficult to reconcile with a pure RPE account.

4. **heterogeneity**: the VTA and SNc contain multiple neuronal subtypes with different projection targets, molecular markers, and response profiles. the "dopamine = RPE" story applies most cleanly to a specific population, not all dopamine neurons.

## key references

- Schultz, W., Dayan, P. & Montague, P. R. (1997). A neural substrate of prediction and reward. Science, 275(5306), 1593-1599.
- Schultz, W. (2016). Dopamine reward prediction error coding. Dialogues in Clinical Neuroscience, 18(1), 23-32.
- Dabney, W. et al. (2020). A distributional code for value in dopamine-based reinforcement learning. Nature, 577, 671-675.
- Sutton, R. S. & Barto, A. G. (2018). Reinforcement Learning: An Introduction. 2nd ed. MIT Press.
- Berridge, K. C. & Robinson, T. E. (1998). What is the role of dopamine in reward: hedonic impact, reward learning, or incentive salience? Brain Research Reviews, 28(3), 309-369.
- Frank, M. J. (2005). Dynamic dopamine modulation in the basal ganglia: a neurocomputational account of cognitive deficits in medicated and nonmedicated Parkinsonism. Journal of Cognitive Neuroscience, 17(1), 51-72.

## see also

- [[neuromodulatory_framework]]
- [[neuromodulation_to_learning_and_gating]]
- [[norepinephrine_system]]
- [[acetylcholine_system]]
- [[precision_weighting]]
- [[hebbian_learning]]
- [[stdp]]
