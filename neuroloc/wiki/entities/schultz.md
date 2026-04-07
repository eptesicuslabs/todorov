# Wolfram Schultz

## biographical summary

Wolfram Schultz (b. 1944) is a German-British neuroscientist and Professor of Neuroscience at the University of Cambridge. he is best known for discovering that dopamine neurons encode the reward prediction error (RPE), the foundational experimental finding linking biological neural activity to the temporal-difference learning algorithm from reinforcement learning.

Schultz received his medical degree from the University of Heidelberg in 1972 and completed postdoctoral work with Otto Creutzfeldt (Max Planck Institute, Gottingen), John Eccles (SUNY Buffalo), and Urban Ungerstedt (Karolinska Institute). he worked at the University of Fribourg, Switzerland from 1977 to 2001 before moving to Cambridge.

## the RPE discovery

during the 1980s and 1990s, Schultz recorded from dopamine neurons in behaving macaques during Pavlovian conditioning tasks. the key insight, published in the landmark 1997 Science paper with Peter Dayan and P. Read Montague, was that phasic dopamine activity obeys:

    dopamine response = reward received - reward predicted

this precisely matches the TD error signal delta_t = r_t + gamma * V(s_{t+1}) - V(s_t) from reinforcement learning theory. the three conditions -- burst for unexpected reward, baseline for expected reward, depression for omitted reward -- established dopamine as a biological teaching signal.

Schultz subsequently extended this work to distributional coding (with DeepMind: Dabney et al. 2020), neuroeconomics, and decision-making under risk, studying dopamine neurons, orbitofrontal cortex, striatum, and amygdala.

## recognition

- the Brain Prize (Lundbeck Foundation, 2017, shared with Peter Dayan and Ray Dolan)
- Gruber Prize in Neuroscience (2018)
- Fellow of the Royal Society (FRS)
- Member of the Academia Europaea
- Wellcome Principal Research Fellow (2001-2023)
- past president of the European Brain and Behaviour Society

## relevance to todorov

the dopamine RPE signal is the most precisely quantified neuromodulatory signal in the brain and the strongest evidence for Doya's [[neuromodulatory_framework]]. todorov has no RPE analog -- see [[dopamine_system]] and [[neuromodulation_to_learning_and_gating]].

## key references

- Schultz, W., Dayan, P. & Montague, P. R. (1997). A neural substrate of prediction and reward. Science, 275(5306), 1593-1599.
- Schultz, W. (2016). Dopamine reward prediction error coding. Dialogues in Clinical Neuroscience, 18(1), 23-32.
- Schultz, W. (2015). Neuronal reward and decision signals: from theories to data. Physiological Reviews, 95(3), 853-951.
