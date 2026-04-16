# Rao & Ballard

status: definitional. last fact-checked 2026-04-16.

## Rajesh P. N. Rao

born 1970 in Madras, India. professor of computer science and engineering and electrical and computer engineering at the University of Washington, Seattle. director of the Center for Neurotechnology. Cherng Jia and Elizabeth Yun Hwang endowed professor.

education: B.S. summa cum laude in computer science/mathematics from Angelo State University (1992). M.S. (1994) and Ph.D. (1998) in computer science from the University of Rochester. Sloan postdoctoral fellow at the Salk Institute for Biological Studies.

primary research areas: computational neuroscience, brain-computer interfacing, artificial intelligence. with Dana Ballard, proposed the hierarchical [[predictive_coding]] model of cortical function (1999). in brain-computer interfacing, his group demonstrated the first direct brain control of a humanoid robot (2007) and the first human brain-to-brain communication via the internet (2013).

key publications:
- Rao, R. P. N. & Ballard, D. H. (1999). Predictive coding in the visual cortex. Nature Neuroscience, 2(1), 79-87.
- Huang, Y. & Rao, R. P. N. (2011). Predictive coding. WIREs Cognitive Science, 2(5), 580-593.
- Rao, R. P. N. (2005). Bayesian inference and attentional modulation in the visual cortex. NeuroReport, 16(16), 1843-1848.

books: Brain-Computer Interfacing (Cambridge University Press, 2013). co-editor of Probabilistic Models of the Brain (MIT Press, 2002) and Bayesian Brain (MIT Press, 2007).

awards: Guggenheim Fellowship, Fulbright Scholar, NSF CAREER Award, ONR Young Investigator Award, Sloan Faculty Fellowship, David and Lucile Packard Fellowship. Google Scholar citations: >29,000.

## Dana H. Ballard (1946-2022)

born October 15, 1946. professor of computer science at the University of Texas at Austin (2006-2022). previously professor at the University of Rochester (1975-2006).

education: B.S. in aeronautics and astronautics from MIT (1967). M.S. in information and control engineering from the University of Michigan (1970). Ph.D. in information engineering from UC Irvine (1974).

primary research areas: computer vision, active perception, computational neuroscience, embodied cognition. pioneered the concept of active vision: the idea that biological visual processing is inseparable from motor behavior (eye movements, head movements, locomotion). with Christopher Brown, wrote the foundational textbook Computer Vision (Prentice Hall, 1982). his 2015 book Brain Computation as Hierarchical Abstraction (MIT Press) synthesized decades of work on hierarchical processing.

Ballard's key insight in the 1999 predictive coding paper was connecting efficient coding to feedback circuitry: if the cortex uses top-down feedback to predict sensory input, then feedforward connections need only transmit the prediction error -- achieving redundancy reduction through hierarchical prediction rather than through lateral inhibition alone.

key publications:
- Rao, R. P. N. & Ballard, D. H. (1999). Predictive coding in the visual cortex. Nature Neuroscience, 2(1), 79-87.
- Ballard, D. H., Hayhoe, M. M., Pook, P. K. & Rao, R. P. N. (1997). Deictic codes for the embodiment of cognition. Behavioral and Brain Sciences, 20(4), 723-742.
- Ballard, D. H. (2015). Brain Computation as Hierarchical Abstraction. MIT Press.

## the 1999 paper

Rao & Ballard (1999) is the foundational computational model of [[predictive_coding]]. the model describes visual processing in which feedback connections from higher to lower cortical areas carry predictions of lower-level neural activity, while feedforward connections carry the residual errors between predictions and actual activity. when trained on natural images, the model developed:
- simple-cell-like receptive fields in the representation units
- end-stopping and other extra-classical receptive field effects in the error units

these results were significant because end-stopping had been attributed to feedforward mechanisms. the predictive coding model showed it could arise from cortical feedback, suggesting that extra-classical receptive field effects are a consequence of hierarchical prediction rather than purely local inhibition.

the paper has been cited over 5,200 times and is the direct predecessor to Friston's [[free_energy_principle]] framework.

## see also

- [[predictive_coding]]
- [[free_energy_principle]]
- [[efficient_coding]]
- [[friston]]
