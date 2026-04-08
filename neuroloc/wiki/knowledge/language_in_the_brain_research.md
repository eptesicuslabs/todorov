# language in the brain research

curated peer-reviewed research on how the brain processes language. language is the domain where todorov is evaluated (next-token prediction), making this the most directly relevant knowledge domain. the brain's language system is distributed, left-lateralized, dissociable from executive function, and organized as dual processing streams. it exploits statistical regularities, generates predictions, and is modality-independent (spoken, signed, written).

## dual-stream architecture

### the dual-stream model of speech processing

hickok, g. & poeppel, d. (2007). the cortical organization of speech processing. *nature reviews neuroscience*, 8(5), 393-402.

key finding: speech processing is organized into two parallel streams originating from bilateral auditory cortex. the ventral stream (superior and middle temporal gyrus, projecting anteriorly) maps sound to meaning -- it performs spectrotemporal analysis, phonological processing, and lexical-semantic access. the dorsal stream (posterior temporal regions projecting to inferior frontal and premotor cortex via the arcuate fasciculus) maps sound to articulation -- it maintains sensorimotor representations for speech production and supports phonological working memory. the ventral stream is bilateral (either hemisphere suffices for comprehension); the dorsal stream is strongly left-lateralized.

relevance to neural computer: the dual-stream architecture separates what language means (ventral/semantic) from how it sounds (dorsal/articulatory). todorov processes language through a single unified pathway -- there is no separation between semantic and articulatory processing. this may not be a limitation for text-based language modeling (where articulatory representation is unnecessary), but it means the architecture cannot capture the dissociations that characterize human language processing (e.g., patients who understand but cannot repeat, or vice versa). the bilateral ventral / lateralized dorsal distinction also suggests that redundancy is valuable for the robust pathway (comprehension) but not for the specialized pathway (production).

confidence: high. the dual-stream model is supported by decades of lesion data, neuroimaging, and electrophysiology. caveat: the boundary between streams is not sharp -- there is substantial interaction, and some models propose a third stream.

## the language network

### fedorenko language network

fedorenko, e., hsieh, p. j., nieto-castanon, a., whitfield-gabrieli, s., & kanwisher, n. (2010). new method for fmri investigations of language: defining rois functionally in individual subjects. *journal of neurophysiology*, 104(2), 1177-1194.

fedorenko, e. & blank, i. a. (2020). broca's area is not a natural kind. *trends in cognitive sciences*, 24(4), 270-284.

fedorenko, e., ivanova, a. a., & regev, t. I. (2024). the language network as a natural kind within the broader landscape of the human brain. *nature reviews neuroscience*, 25, 289-312.

key finding: using individual-subject functional localization (rather than group-averaged activation maps), fedorenko identified a distributed left-lateralized language network comprising regions in inferior frontal gyrus (including but not limited to "broca's area"), superior and middle temporal gyrus, and angular gyrus. this network responds selectively to linguistic input (sentences > word lists > nonwords) and is dissociable from the multiple-demand (executive function) network, the theory of mind network, and the default mode network. critically, broca's area as traditionally defined is not a unitary functional region -- it contains interleaved patches of language-selective and domain-general tissue.

relevance to neural computer: the dissociation between language processing and executive function is important. in todorov, all processing (language, reasoning, planning) flows through the same layers -- there is no functional specialization. the biological finding that language has its own dedicated network suggests that language processing benefits from specialized circuitry rather than general-purpose computation. however, todorov achieves strong language performance with general-purpose layers, suggesting that the biological specialization may reflect developmental/evolutionary constraints rather than computational necessity. the death of "broca's area" as a monolithic language region is also relevant -- it cautions against localizing function too precisely in any architecture.

confidence: high. individual-subject functional localization is now the gold standard for language neuroscience. the dissociation from executive function is robust across dozens of studies. caveat: the language network definition is based on sentence processing; whether it captures all aspects of language (pragmatics, discourse, prosody) is less clear.

## prediction in language

### the n400 as prediction violation

kutas, m. & hillyard, s. a. (1980). reading senseless sentences: brain potentials reflect semantic anomaly. *science*, 207(4427), 203-205.

key finding: semantically anomalous words in sentences (e.g., "he spread the warm bread with socks") elicit a large negative-going erp component peaking at ~400 ms (the n400). the amplitude of the n400 is inversely proportional to the word's predictability (cloze probability) in its context -- highly predictable words elicit almost no n400, while unpredictable words elicit a large n400 regardless of whether they are semantically anomalous or merely unlikely. this established that the brain generates real-time predictions about upcoming words and that the n400 indexes the degree of prediction violation.

relevance to neural computer: the n400 is the biological analog of prediction error in next-token prediction. todorov's training objective (minimize cross-entropy loss on the next token) is functionally equivalent to minimizing the n400 amplitude -- both measure the discrepancy between the predicted and actual next word. the ~400 ms latency of the n400 reflects the time required for full semantic integration; todorov processes each token in a single forward pass with no analog of this temporal unfolding. the finding that n400 amplitude tracks cloze probability (not just anomaly) means the brain computes graded predictions, not just binary accept/reject -- consistent with todorov's softmax output distribution.

confidence: high. the n400 is one of the most replicated findings in cognitive neuroscience (thousands of studies since 1980). the relationship between n400 amplitude and cloze probability is robust. caveat: the n400 may reflect multiple processes (prediction error, lexical access difficulty, semantic integration cost) rather than pure prediction violation.

## statistical learning

### statistical learning in language acquisition

saffran, j. r., aslin, r. n., & newport, e. L. (1996). statistical learning by 8-month-old infants. *science*, 274(5294), 1926-1929.

key finding: 8-month-old infants exposed to a continuous stream of synthesized syllables (with no pauses or stress cues) for just 2 minutes learned to segment the stream into "words" based solely on transitional probabilities between syllables. syllable pairs that always co-occurred (within-word transitions, probability = 1.0) were distinguished from pairs that occurred by chance across word boundaries (between-word transitions, probability = 0.33). this demonstrated that the human brain performs statistical learning over sequential input from the earliest stages of development, without explicit supervision or linguistic knowledge.

relevance to neural computer: next-token prediction is statistical learning. saffran et al. showed that this is not just a useful training objective but is the actual mechanism used by biological brains to acquire language structure. the speed of learning (2 minutes of exposure) and the age of learners (8 months) suggest that statistical sequence learning is a primitive, hardwired capability -- not something learned through experience. todorov's architecture is explicitly designed for this computation. the transitional probability framework maps directly to bigram and trigram statistics that emerge naturally in autoregressive language models.

confidence: high. replicated extensively across ages, modalities (auditory, visual), and stimulus types. extended to non-adjacent dependencies and hierarchical structure. caveat: statistical learning is necessary but not sufficient for language acquisition -- it provides the raw material (segmented units) but not the grammar.

## motor theory and embodiment

### somatotopic activation during language

[authors]. motor theory of speech perception posits that understanding speech requires activating motor representations of how those speech sounds are produced. neuroimaging studies consistently show somatotopic activation in motor and premotor cortex during language comprehension -- hearing action words like "kick" activates leg motor cortex, "pick" activates hand motor cortex. however, patients with motor cortex lesions or motor neuron disease (als) can still comprehend language, demonstrating that motor activation is real but not necessary for comprehension. the motor activation may reflect automatic simulation that enriches but does not constitute understanding.

relevance to neural computer: the motor theory debate informs whether language understanding requires embodied simulation or can be achieved through purely distributional/statistical processing. todorov achieves functional language competence without any embodied or motor component, which aligns with the finding that motor activation is not necessary for comprehension. however, the richness of human language understanding (including metaphor, spatial language, action descriptions) may partially depend on sensorimotor grounding that distributional models cannot fully capture.

confidence: medium. the activation findings are robust; the causal role is clearly not necessary for basic comprehension. caveat: the debate between embodied and distributional theories of meaning is ongoing and may be a false dichotomy -- both may contribute.

## modality independence

### sign language uses the same network

petitto, l. a., zatorre, r. j., gauna, k., nikelski, e. j., dostie, d., & evans, a. c. (2000). speech-like cerebral activity in profoundly deaf people processing signed languages: implications for the neural basis of human language. *proceedings of the national academy of sciences*, 97(25), 13961-13966.

key finding: deaf native signers processing sign language activate the same left-lateralized language network (inferior frontal gyrus, superior temporal regions) as hearing individuals processing spoken language. critically, right hemisphere regions associated with spatial processing are not preferentially recruited for sign language despite its visuospatial modality. this demonstrates that the language network is modality-independent -- it processes linguistic structure regardless of whether that structure is conveyed through sound, gesture, or (by extension) text.

relevance to neural computer: the modality independence of the language network supports todorov's text-only approach to language modeling. if the same neural circuitry processes language regardless of input modality, then the computational principles underlying language are abstract and not tied to auditory or visual processing. this means a text-based model is not fundamentally limited by its lack of auditory or visual input -- it can capture the core linguistic computation. however, the multimodal aspects of language (prosody, gesture, facial expression in sign) that are processed by non-language networks may still be missed.

confidence: high. replicated across multiple sign languages (asl, bsl, lsf) and with multiple neuroimaging methods. caveat: "same network" is defined at the resolution of fmri (~2mm); finer-grained differences in neural coding between modalities may exist.

## linguistic relativity

### color perception and language

winawer, j., witthoft, n., frank, m. c., wu, l., wade, a. r., & boroditsky, l. (2007). russian blues: effects of language on color discrimination. *proceedings of the national academy of sciences*, 104(19), 7780-7785.

key finding: russian speakers, whose language obligatorily distinguishes between light blue (goluboy) and dark blue (siniy), showed faster discrimination of colors that crossed the goluboy-siniy boundary compared to colors within either category. english speakers, who use a single term (blue), showed no such advantage. the effect was present only when performing a concurrent verbal task but disappeared with a concurrent spatial task, confirming that the effect is mediated by online linguistic processing rather than permanent perceptual restructuring. this demonstrates that language can influence low-level perceptual discrimination in real time, but the effect is narrow, task-dependent, and mediated by active language processing.

relevance to neural computer: the sapir-whorf finding is relevant to tokenization. todorov's tokenizer creates categorical boundaries in the input space (each token is a discrete category). the russian blues result suggests that these categorical boundaries can influence downstream processing -- what the model can easily distinguish depends partly on where the tokenizer draws boundaries. a tokenizer that splits a concept across two tokens creates a categorical boundary that may enhance discrimination of that concept's variants, analogous to how having two color words enhances color discrimination.

confidence: high. well-controlled experiment with specific interference conditions. replicated with other language pairs and other domains (spatial relations, time). caveat: the effects are small (reaction time differences of ~100 ms), narrow (specific to cross-category comparisons), and dependent on concurrent verbal processing. strong versions of sapir-whorf (language determines thought) are not supported.

## relevance to todorov

### validated connections
- next-token prediction is the biological language acquisition mechanism (saffran statistical learning)
- the n400 is the biological analog of cross-entropy loss -- both measure prediction violation graded by probability
- modality independence (petitto) supports text-only modeling as capturing core linguistic computation
- the language network's dissociation from executive function (fedorenko) suggests specialized processing benefits, but todorov achieves strong results with general-purpose layers

### challenged assumptions
- todorov has no dual-stream separation (semantic vs articulatory) -- all processing is unified
- no temporal unfolding of semantic integration (n400 takes ~400 ms; todorov processes each token in a single forward pass)
- no embodied grounding -- todorov lacks sensorimotor representations that may enrich language understanding
- tokenization creates categorical boundaries that may not align with optimal perceptual/conceptual distinctions

### future phases
- dual-stream processing: separate semantic and form-based pathways that can be selectively impaired or enhanced (phase 6+)
- prediction-as-training-signal: auxiliary n400-like prediction error losses at intermediate layers
- tokenizer-aware processing: mechanisms that can bridge or merge token boundaries for concepts that span multiple tokens

## see also

- [[predictive_coding]]
- [[selective_attention]]
- [[global_workspace_theory]]
- [[complementary_learning_systems]]
- [[perception_and_consciousness_research]]
- [[memory_systems_research]]
