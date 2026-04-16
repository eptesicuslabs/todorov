# Matteo Carandini and David Heeger

status: definitional. last fact-checked 2026-04-16.

## identity

Matteo Carandini: italian-born neuroscientist, professor at University College London (UCL), GlaxoSmithKline/Fight for Sight chair of visual neuroscience. works on visual processing, cortical computation, and normalization.

David Heeger: american computational neuroscientist, professor at New York University (NYU), department of psychology and center for neural science. works on computational models of visual cortex, normalization, attention, and fMRI methodology.

## key contribution: normalization as a canonical computation (2012)

Carandini and Heeger's 2012 review in Nature Reviews Neuroscience synthesized two decades of evidence that divisive normalization -- the operation of dividing a neuron's response by the pooled activity of a population -- is one of a small number of canonical neural computations, alongside linear filtering, thresholding, and recurrence.

see [[divisive_normalization]] for full treatment.

the review unified findings from:
- early vision (contrast gain control, cross-orientation suppression, surround suppression)
- attention (Reynolds and Heeger 2009 normalization model of attention)
- olfaction (concentration-invariant odor coding in Drosophila)
- multisensory integration (inverse effectiveness)
- decision making (value normalization in LIP)

the argument: the same mathematical equation fits neural responses across these diverse systems, suggesting a common computational principle. the biophysical mechanisms differ (synaptic depression, shunting inhibition, recurrent circuits), but the computation is the same.

## Heeger's earlier work

Heeger proposed divisive normalization for V1 in his 1992 paper "normalization of cell responses in cat striate cortex." the model explained cross-orientation suppression: why adding a non-preferred grating to a preferred grating suppresses V1 responses, even though the non-preferred grating alone does not drive the cell. the normalization model predicts divisive suppression, contrast dependence, and non-specificity for mask orientation -- all confirmed experimentally.

Heeger also co-developed the Reynolds-Heeger normalization model of attention (2009) with John Reynolds, which explained how attention interacts with contrast gain control through a multiplicative attention field that modulates the stimulus drive before normalization.

## Carandini's contributions

Carandini has contributed extensive experimental work on normalization in visual cortex, including quantitative measurements of contrast gain control, surround suppression, and adaptation. his work established that normalization is not merely a descriptive model but makes quantitative predictions about neural responses that can be tested and confirmed.

## legacy

the normalization framework has become one of the most widely used models in computational neuroscience. it provides a common language for describing gain control across sensory systems, attention, and decision making. the 2012 review has been cited over 3000 times.

the concept of "canonical neural computations" -- a small set of elementary operations that are reused across brain regions and species -- has influenced thinking about how to build biologically inspired artificial neural networks. divisive normalization has been incorporated into deep learning architectures for image compression (generalized divisive normalization, GDN) and image segmentation.

## key references

- heeger, d. j. (1992). normalization of cell responses in cat striate cortex. visual neuroscience, 9(2), 181-197.
- carandini, m. & heeger, d. j. (2012). normalization as a canonical neural computation. nature reviews neuroscience, 13(1), 51-62.
- reynolds, j. h. & heeger, d. j. (2009). the normalization model of attention. neuron, 61(2), 168-185.

## see also

- [[divisive_normalization]]
- [[lateral_inhibition]]
