# executive function and cognitive control

## prefrontal working memory maintenance

miller, e. k. & cohen, j. d. (2001). an integrative theory of prefrontal cortex function. *annual review of neuroscience*, 24, 167-202.

key finding: pfc maintains task rules as persistent neural activity patterns that bias processing in posterior cortex. pfc doesn't compute the answer -- it holds the context that shapes how other regions compute. mixed selectivity (fusi et al. 2016 Nature): individual pfc neurons encode nonlinear combinations of task variables, enabling exponentially more task states than neurons.

relevance: a neural machine needs a control layer that holds goals and biases processing without doing the processing itself. this maps to a gating mechanism on the residual stream.

confidence: high.

## basal ganglia gating of working memory

o'reilly, r. c. & frank, m. j. (2006). making working memory work: a computational model of learning in the prefrontal cortex and basal ganglia. *neural computation*, 18(2), 283-328.

key finding: the basal ganglia gate updates to pfc working memory. the go pathway (D1) opens the gate, allowing new information in. the nogo pathway (D2) keeps the gate closed, maintaining current contents. dopamine prediction errors train which inputs should be gated in. this solves the maintenance-vs-update dilemma: hold what's useful, update when something better arrives.

relevance: kda's beta gate (sigmoid(beta_proj(x))) is the closest analog -- it gates what enters recurrent state. but beta is data-dependent, not reward-dependent. a neural machine could add a learned "importance" signal that modulates beta based on downstream utility.

confidence: high.

## conflict monitoring

botvinick, m. m., braver, t. s., barch, d. m., carter, c. s. & cohen, j. d. (2001). conflict monitoring and cognitive control. *psychological review*, 108(3), 624-652.

key finding: the anterior cingulate cortex detects when multiple competing responses are simultaneously active. high conflict triggers increased cognitive control (more pfc engagement, slower more careful processing). this is a self-monitoring mechanism: the system detects its own uncertainty and adjusts effort.

relevance: a neural machine that monitors its own state norm, attention entropy, or spike conflict could dynamically allocate more computation to uncertain inputs. the bcm-like adaptive alpha is a primitive version of this.

confidence: high.

## metacognition

fleming, s. m. & dolan, r. j. (2012). the neural basis of metacognitive ability. *philosophical transactions of the royal society b*, 367(1594), 1338-1349.

key finding: the anterior prefrontal cortex (area 10) tracks confidence in one's own decisions. gray matter volume in this region correlates with metacognitive accuracy across individuals. metacognition is dissociable from task performance: you can be good at a task but bad at knowing how good you are.

relevance: a neural machine that outputs a calibrated confidence signal alongside its computation is fundamentally more useful than one that doesn't. confidence = the machine knowing what it knows and what it doesn't.

confidence: high.

## planning and model-based control

daw, n. d., niv, y. & dayan, p. (2005). uncertainty-based competition between prefrontal and dorsolateral striatal systems for behavioral control. *nature neuroscience*, 8(12), 1704-1711.

key finding: the brain arbitrates between model-based (deliberative, pfc-driven) and model-free (habitual, striatal) control based on uncertainty. when the world is predictable, habits dominate (cheap). when it's uncertain, deliberation takes over (expensive but accurate). the arbitration itself is learned.

relevance: a neural machine could have fast (single-pass, kda recurrence) and slow (multi-iteration, within-layer recurrence) processing modes, with automatic arbitration based on input complexity.

confidence: high.

## task switching

monsell, s. (2003). task switching. *trends in cognitive sciences*, 7(3), 134-140.

key finding: switching between tasks incurs a measurable cost (200-500ms) even when fully predictable. the cost reflects reconfiguration of processing pathways: updating task rules in pfc, inhibiting the previous task set, activating the new one. residual interference from the old task set persists for several trials.

relevance: a neural machine that processes different input types (text, image, structured data) needs a reconfiguration mechanism. the layer schedule could be dynamically adjusted based on input modality.

confidence: high.

## see also

- [[basal_ganglia]]
- [[dopamine_system]]
- [[predictive_coding]]
- [[selective_attention]]
