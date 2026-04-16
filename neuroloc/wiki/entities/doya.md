# Kenji Doya

status: definitional. last fact-checked 2026-04-16.

## biographical summary

Kenji Doya is a Japanese computational neuroscientist and Professor at the Okinawa Institute of Science and Technology (OIST), where he leads the Neural Computation Unit. he received his PhD in Engineering from the University of Tokyo and conducted postdoctoral research at the University of California San Diego and the Salk Institute. before OIST, he worked at the Advanced Telecommunications Research Institute (ATR) in Kyoto on autonomous learning robots and neural mechanisms of learning.

## metalearning and neuromodulation

Doya's most influential contribution is the 2002 metalearning framework, which maps each major neuromodulatory system to a meta-parameter of reinforcement learning:

- dopamine -> reward signal (TD error)
- serotonin -> temporal discount factor (gamma)
- norepinephrine -> exploration rate (temperature)
- acetylcholine -> learning rate (eta)

this framework reframes neuromodulation as online hyperparameter optimization: the neuromodulatory systems do not compute or learn directly, but adjust the parameters that govern how cortical and striatal circuits learn. the insight that the brain has a two-level control system -- learning and metalearning -- has been deeply influential in both computational neuroscience and biologically-inspired AI.

## other contributions

- complementary roles of basal ganglia (reward-based), cerebellum (error-based), and cortex (model-based) learning systems (Doya 2000)
- Bayesian inference and reinforcement learning in the brain (OIST Neural Computation Unit ongoing research)
- integration of theoretical and experimental approaches to study neuromodulator systems

## recognition

- JSPS Prize
- Tsukahara Memorial Award
- Minister of Education, Culture, Sports, Science and Technology Award (Japan)
- Donald O. Hebb Award (International Neural Network Society)
- Tateishi Special Prize (9th, Tateishi Science and Technology Foundation)

## relevance to todorov

Doya's framework provides the computational vocabulary for analyzing what todorov is missing: the architecture has no online modulation of learning rate (ACh), no reward signal (DA), no temporal discounting (5-HT), and no exploration mechanism (NE). all meta-parameters are fixed before training. see [[neuromodulatory_framework]] and [[neuromodulation_to_learning_and_gating]].

## key references

- Doya, K. (2002). Metalearning and neuromodulation. Neural Networks, 15(4-6), 495-506.
- Doya, K. (2000). Complementary roles of basal ganglia and cerebellum in learning and motor control. Current Opinion in Neurobiology, 10(6), 732-739.
- Doya, K. (2008). Modulators of decision making. Nature Neuroscience, 11(4), 410-416.
