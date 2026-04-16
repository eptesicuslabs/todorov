# motor control and forward models

status: current (as of 2026-04-16).

## the cerebellum as forward model

wolpert, d. m., miall, r. c. & kawato, m. (1998). internal models in the cerebellum. *trends in cognitive sciences*, 2(9), 338-347.

key finding: the cerebellum builds forward models that predict sensory consequences of motor commands 100-200ms ahead. this prediction compensates for neural transmission delays. the cerebellum receives efference copies (copies of motor commands) and predicts what the sensory result will be. if prediction matches reality, processing continues. if there's a mismatch, the error signal (climbing fiber) triggers learning.

relevance: a neural machine that generates structured output (sequences, actions, responses) needs to predict the consequences of its own outputs before committing. the forward model is imagination applied to action -- the simplest, most validated form of internal simulation.

confidence: high.

## motor cortex as dynamical system

churchland, m. m. et al. (2012). neural population dynamics during reaching. *nature*, 487, 51-56.

key finding: motor cortex activity during reaching is best described as a dynamical system -- a low-dimensional trajectory in neural state space -- not as a muscle-by-muscle controller. the preparatory activity sets the initial condition; the dynamics unfold autonomously once triggered. the same circuit can produce different movements by starting from different initial conditions.

relevance: the crbr recurrent state (kda S_t) evolves as a dynamical system. the motor cortex finding validates that useful computation can emerge from the dynamics of a recurrent state, not just from the static mapping at each step.

confidence: high.

## motor sequence chunking

graybiel, a. m. (1998). the basal ganglia and chunking of action repertoires. *neurobiology of learning and memory*, 70, 119-136.

key finding: the basal ganglia compress frequently repeated action sequences into single "chunks" that can be triggered as a unit. once chunked, the sequence executes automatically without step-by-step control. this converts deliberate (expensive) sequential processing into automatic (cheap) pattern triggering.

relevance: chunking is compression applied to time. a neural machine that detects repeated subsequences and compresses them into single retrievable units would dramatically reduce computation for familiar inputs. this connects to the working memory 4-chunk limit (cowan 2001).

confidence: high.

## optimal feedback control

todorov, e. & jordan, m. i. (2002). optimal feedback control as a theory of motor coordination. *nature neuroscience*, 5(11), 1226-1235.

key finding: the brain doesn't plan complete trajectories. it specifies a cost function (minimize endpoint error + minimize effort) and lets the sensorimotor system find the optimal solution online via feedback. task-relevant variability is corrected; task-irrelevant variability is allowed. this is fundamentally different from planning then executing.

relevance: a neural machine could specify objectives rather than steps. instead of "process token 1, then token 2, then token 3," specify "minimize prediction error across the sequence" and let the recurrent dynamics find the solution.

confidence: high.

## active inference

friston, k. j. (2011). what is optimal about motor control? *neuron*, 72(3), 488-498.

key finding: action is prediction error minimization. you don't send motor commands -- you send proprioceptive predictions that the body fulfills by moving to match. action and perception are the same process: both minimize the difference between prediction and reality. the difference is which side adjusts -- perception updates the model, action updates the world.

relevance: a neural machine that acts by predicting its own output and then matching it is fundamentally different from one that computes an output and emits it. the prediction-first architecture makes the output predictable, controllable, and inspectable before commitment.

confidence: high theoretically. the active inference framework is mathematically elegant but empirically less validated than forward models.

## see also

- [[cerebellum_research]]
- [[predictive_coding]]
- [[basal_ganglia]]
- [[free_energy_principle]]
